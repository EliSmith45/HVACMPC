using LinearAlgebra, ComponentArrays, DataInterpolations
using DifferentialEquations, SciMLSensitivity
using Optimization, OptimizationNLopt, Optim, NLopt
using ReverseDiff, Zygote
using ForwardDiff
using Random
using Gadfly

#= structs =#

struct ZoneDef
    id::Symbol
    cAir::Float64
    cMass::Float64
    rAirAm::Float64
    rAirMass::Float64
    α::Float64 #solar heat gain coefficient to thermal mass
    has_solar::Bool
    has_infiltration::Bool #heat transfer from ambient direct to air, bypassing walls
end

function ZoneDef(id::Symbol; cAir=100.0, cMass=1000.0, rAirAm=0.001, rAirMass = 0.0001, α=0.0)
    return ZoneDef(id, cAir, cMass, rAirAm, rAirMass, α, α != 0.0, rAirAm != 0.0)
end

struct WallDef
    id::Symbol
    zone_a::Symbol
    zone_b::Symbol
    num_nodes::Int
    c::Vector{Float64}
    r::Vector{Float64}
    α_a::Float64 #solar to the wall capacitor on side a
    α_b::Float64 #solar to the wall capacitor on side b
end

function WallDef(id::Symbol, zone_a::Symbol, zone_b::Symbol, num_nodes::Int; c=fill(100.0, num_nodes), r=fill(0.001, num_nodes + 1), α_a=0.0, α_b=0.0)
    @assert length(c) == num_nodes "Length of c must equal num_nodes"
    @assert length(r) == num_nodes + 1 "Length of r must equal num_nodes + 1"
   
    return WallDef(id, zone_a, zone_b, num_nodes, c, r, α_a, α_b)
end

#= Graph model topology =#

struct BuildingGraphModel{T_OA, T_SOL, T_INT, T_HVAC}
    num_nodes::Int
    node_names::Vector{Symbol}

    #internal conduction graph edges: edge e connects edge_src[e] to edge_dst[e]
    edge_src::Vector{Int}
    edge_dst::Vector{Int}

    #external ambient connections: node ambient_nodes[k] connects to T_oa
    ambient_nodes::Vector{Int}

    #solar connections: node solar_nodes[k] connects to T_sol
    solar_nodes::Vector{Int}

    #internal gains and HVAC node list
    hvac_nodes::Vector{Int}
    
    #differentiable array parameters
    params::ComponentArray

    #interpolators for time-varying inputs
    T_oa::T_OA
    I_solar::T_SOL
    Q_int::T_INT
    Q_hvac::T_HVAC
    
end

function BuildingGraphModel(zones::Vector{ZoneDef}, walls::Vector{WallDef}, 
    T_oa, I_solar, Q_int, Q_hvac)

    # create a flat node registry
    node_names = Symbol[]
    capacitances = Float64[]

    zone_air_idx = Dict{Symbol, Int}()
    zone_mass_idx = Dict{Symbol, Int}()

    # track node indices for zones
    for z in zones
        push!(node_names, Symbol(z.id, "_air"))
        push!(capacitances, z.cAir)
        zone_air_idx[z.id] = length(node_names)

        push!(node_names, Symbol(z.id, "_mass"))
        push!(capacitances, z.cMass)
        zone_mass_idx[z.id] = length(node_names)
    end

    # track node indices for walls
    wall_node_indices = Dict{Symbol, Vector{Int}}()
    for w in walls
        indices = Int[]
        for n in 1:w.num_nodes
            push!(node_names, Symbol("$(w.id)_n$(n)"))
            push!(capacitances, w.c[n])
            push!(indices, length(node_names))
        end
        wall_node_indices[w.id] = indices
    end

    total_nodes = length(node_names)

    # build edge lists for conduction, ambient, solar, and hvac connections 
    edge_src = Int[]
    edge_dst = Int[]
    r_internal = Float64[]

    ambient_nodes = Int[]
    r_ambient = Float64[]

    solar_nodes = Int[]
    alpha_solar = Float64[]

    hvac_nodes = Int[]

    #connect zones
    for z in zones
        air_node = zone_air_idx[z.id]
        mass_node = zone_mass_idx[z.id]

        # air to mass connection
        push!(edge_src, air_node)
        push!(edge_dst, mass_node)
        push!(r_internal, z.rAirMass)

        # Infiltration
        if z.has_infiltration
            push!(ambient_nodes, air_node)
            push!(r_ambient, z.rAirAm)
        end

        # Solar
        if z.has_solar
            push!(solar_nodes, mass_node)
            push!(alpha_solar, z.α)
        end

        # HVAC and internal gains
        push!(hvac_nodes, air_node)
    end

    # Connect walls
    for w in walls
        w_nodes = wall_node_indices[w.id]
        K = w.num_nodes

        # Side a boundary
        if w.zone_a == :ambient
            push!(ambient_nodes, w_nodes[1])
            push!(r_ambient, w.r[1])
        else
            
            push!(edge_src, w_nodes[1])
            push!(edge_dst, zone_air_idx[w.zone_a])
            push!(r_internal, w.r[1])
        end

        # Solar on side a
        if w.α_a > 0.0
            push!(solar_nodes, w_nodes[1])
            push!(alpha_solar, w.α_a)
        end

        # Internal wall nodes
        for k in 1:(K-1)
            push!(edge_src, w_nodes[k])
            push!(edge_dst, w_nodes[k+1])
            push!(r_internal, w.r[k+1])
        end

        # Side b boundary
        if w.zone_b == :ambient
            push!(ambient_nodes, w_nodes[K])
            push!(r_ambient, w.r[K+1])
        else
            push!(edge_src, w_nodes[K])
            push!(edge_dst, zone_air_idx[w.zone_b])
            push!(r_internal, w.r[K+1])
        end

        # Solar on side b
        if w.α_b > 0.0
            push!(solar_nodes, w_nodes[K])
            push!(alpha_solar, w.α_b)
        end
    end

    # create component array for parameters
    params = ComponentArray(
        C = capacitances,
        R_internal = r_internal,
        R_ambient = r_ambient,
        α_solar = alpha_solar
    )

    return BuildingGraphModel(
        total_nodes,
        node_names,
        edge_src,
        edge_dst,
        ambient_nodes,
        solar_nodes,
        hvac_nodes,
        params,
        T_oa,
        I_solar,
        Q_int,
        Q_hvac
    )
end

# In-place right-hand side shared by the callable struct and the `ODEFunction`
# wrapper below.  The model is a linear RC network, so the state Jacobian and
# the parameter Jacobian are computed analytically (see `_jac!`/`_paramjac!`).
@inline function _rhs!(du, u, p, t, m::BuildingGraphModel)
    t_oa = m.T_oa(t)
    i_solar = m.I_solar(t)

    du .= 0.0

    @inbounds for (i, k) in enumerate(m.ambient_nodes)
        du[k] += (t_oa - u[k]) * exp(p.R_ambient[i])
    end

    @inbounds for (i, k) in enumerate(m.solar_nodes)
        du[k] += i_solar * exp(p.α_solar[i])
    end

    @inbounds for (i, k) in enumerate(m.hvac_nodes)
        du[k] += m.Q_hvac[i](t) + m.Q_int[i](t)
    end

    @inbounds for i in eachindex(m.edge_src)
        g = exp(p.R_internal[i])
        du[m.edge_dst[i]] += (u[m.edge_src[i]] - u[m.edge_dst[i]]) * g
        du[m.edge_src[i]] += (u[m.edge_dst[i]] - u[m.edge_src[i]]) * g
    end

    @inbounds for i in eachindex(du)
        du[i] *= exp(p.C[i])
    end

    return nothing
end

function (m::BuildingGraphModel)(du, u, p, t)
    _rhs!(du, u, p, t, m)
end

# Analytical state Jacobian ∂f/∂u for the linear RC model.  The Jacobian is
# independent of `u` (and of `t`), and only depends on the conductance/capacitance
# parameters `p`.
@inline function _jac!(J, u, p, t, m::BuildingGraphModel)
    fill!(J, 0.0)
    n = m.num_nodes

    @inbounds begin
        for (i, k) in enumerate(m.ambient_nodes)
            J[k, k] -= exp(p.R_ambient[i])
        end

        for i in eachindex(m.edge_src)
            g = exp(p.R_internal[i])
            s = m.edge_src[i]
            d = m.edge_dst[i]
            J[d, d] -= g
            J[s, s] -= g
            J[d, s] += g
            J[s, d] += g
        end

        # `du .= unscaled_du .* exp.(p.C)`: each row is scaled by its capacitance.
        for i in 1:n
            ci = exp(p.C[i])
            for j in 1:n
                J[i, j] *= ci
            end
        end
    end

    return nothing
end

# Analytical parameter Jacobian ∂f/∂p.  `offC`, `offRi`, `offRa` and `offAs`
# are the 1-based column offsets of each parameter block in the flat
# `ComponentArray` data vector (`C`, `R_internal`, `R_ambient`, `α_solar`).
@inline function _paramjac!(pJ, u, p, t, m::BuildingGraphModel,
                            scratch, offC, offRi, offRa, offAs)
    fill!(pJ, 0.0)
    db = scratch
    fill!(db, 0.0)

    t_oa = m.T_oa(t)
    i_solar = m.I_solar(t)

    @inbounds begin
        # Unscaled RHS, before the final per-node `* exp(p.C)` scaling.
        for (i, k) in enumerate(m.ambient_nodes)
            db[k] += (t_oa - u[k]) * exp(p.R_ambient[i])
        end
        for (i, k) in enumerate(m.solar_nodes)
            db[k] += i_solar * exp(p.α_solar[i])
        end
        for (i, k) in enumerate(m.hvac_nodes)
            db[k] += m.Q_hvac[i](t) + m.Q_int[i](t)
        end
        for i in eachindex(m.edge_src)
            g = exp(p.R_internal[i])
            s = m.edge_src[i]
            d = m.edge_dst[i]
            db[d] += (u[s] - u[d]) * g
            db[s] += (u[d] - u[s]) * g
        end

        # ∂/∂(log C_k) of `db_k * exp(C_k)` is `db_k * exp(C_k)`.
        for k in 1:m.num_nodes
            pJ[k, offC + k - 1] = db[k] * exp(p.C[k])
        end

        for (i, k) in enumerate(m.ambient_nodes)
            pJ[k, offRa + i - 1] = (t_oa - u[k]) * exp(p.R_ambient[i]) * exp(p.C[k])
        end
        for (i, k) in enumerate(m.solar_nodes)
            pJ[k, offAs + i - 1] = i_solar * exp(p.α_solar[i]) * exp(p.C[k])
        end
        for i in eachindex(m.edge_src)
            g = exp(p.R_internal[i])
            s = m.edge_src[i]
            d = m.edge_dst[i]
            pJ[d, offRi + i - 1] = (u[s] - u[d]) * g * exp(p.C[d])
            pJ[s, offRi + i - 1] = (u[d] - u[s]) * g * exp(p.C[s])
        end
    end

    return nothing
end

# Build an `ODEFunction` with analytical state/parameter Jacobians so that
# implicit solvers and `InterpolatingAdjoint(autojacvec = false)` can avoid
# automatic differentiation of the right-hand side altogether.
function _odef(m::BuildingGraphModel)
    n = m.num_nodes
    nC = length(m.params.C)
    nRi = length(m.params.R_internal)
    nRa = length(m.params.R_ambient)
    nAs = length(m.params.α_solar)
    @assert nC == n
    @assert nC + nRi + nRa + nAs == length(m.params)

    offC = 1
    offRi = offC + nC
    offRa = offRi + nRi
    offAs = offRa + nRa

    scratch = Vector{Float64}(undef, n)

    f = (du, u, p, t) -> _rhs!(du, u, p, t, m)
    jac = (J, u, p, t) -> _jac!(J, u, p, t, m)
    paramjac = (pJ, u, p, t) -> _paramjac!(pJ, u, p, t, m, scratch,
                                           offC, offRi, offRa, offAs)

    return ODEFunction{true}(f; jac = jac, paramjac = paramjac,
                             jac_prototype = zeros(n, n))
end

function generate_sample_exogenous_inputs(tspan, steps, num_hvac_nodes;
                                          meanTemp=72.0, varTemp=10.0, amp_mod_amp=15.0, amp_mod_phase=π, amp_mod_freq=1/(3 * tspan[2]),
                                          peakSolar = 500.0,
                                          occInternalGains = fill(5.0, num_hvac_nodes), occ_start = 8.0, occ_end = 18.0,
                                          hvac_freqs = fill(1.0, num_hvac_nodes), hvacPeakCapacities = fill(10.0, num_hvac_nodes))
    t_steps = collect(range(tspan[1], tspan[2], length=steps))
    T_oa_data = (meanTemp .+ (varTemp .* sin.(2π .* t_steps ./ 24.0 .- π/2))) .+ (amp_mod_amp .* sin.(2π .* amp_mod_freq .* t_steps .+ amp_mod_phase))
    I_solar_data = (peakSolar .* max.(0.0, sin.(2π .* t_steps ./ 24.0 .- π/2)))
    Q_int_data = [[occ_start < mod(t, 24.0) < occ_end ? zg : 0.0 for t in t_steps] for zg in occInternalGains]
    Q_hvac_data = [[hp * max(0.0, sin(2π * freq * t)) for t in t_steps] for (hp, freq) in zip(hvacPeakCapacities, hvac_freqs)]

    T_oa = LinearInterpolation(T_oa_data, t_steps)
    I_solar = LinearInterpolation(I_solar_data, t_steps)
    Q_int = Tuple([LinearInterpolation(q, t_steps) for q in Q_int_data])
    Q_hvac = Tuple([LinearInterpolation(q, t_steps) for q in Q_hvac_data])

    return T_oa, I_solar, Q_int, Q_hvac
end

function update_exo_inputs(model::BuildingGraphModel, T_oa, I_solar, Q_int, Q_hvac)
    return BuildingGraphModel(
        model.node_names,
        model.edge_src,
        model.edge_dst,
        model.ambient_nodes,
        model.solar_nodes,
        model.hvac_nodes,
        model.params,
        T_oa,
        I_solar,
        Q_int,
        Q_hvac
    )
end

# -----------------------------------------------------------------------------
# Parameter-recovery objective for the air/HVAC nodes, differentiated with
# SciMLSensitivity.adjoint_sensitivities (interpolating/continuous adjoint).
#
#   loss(p) = Σᵢ wᵢ Σₖ (uₖ(tᵢ) - data_air[i,k])² ,
#
# where k runs over `topo.hvac_nodes` (the zone air nodes), the wᵢ are
# trapezoidal quadrature weights on `tgrid`, `data_air` is an
# `n_t × n_air` matrix of observed air-node temperatures, and the forward model
# is `ODEProblem(topo, u0, tspan, p)`.  The parameter vector p is assumed to be
# in the same space the model RHS expects (log-space for this model).
# -----------------------------------------------------------------------------

function _air_node_quadrature_weights(tgrid)
    n_t = length(tgrid)
    n_t > 1 || throw(ArgumentError("tgrid must contain at least two points"))

    dtv = diff(tgrid)
    w = zeros(n_t)
    w[1] = dtv[1] / 2
    w[end] = dtv[end] / 2
    for i in 2:(n_t - 1)
        w[i] = (dtv[i - 1] + dtv[i]) / 2
    end
    return w
end

function _check_air_node_data(air_idx, data_air, n_t)
    n_air = length(air_idx)
    size(data_air) == (n_t, n_air) || throw(DimensionMismatch(
        "data_air must have size (length(tgrid), n_air) = ($n_t, $n_air), got $(size(data_air))"))
    return nothing
end

function _air_node_residual_loss(sol, air_idx, data_air, w, tgrid)
    u = zero(sol.u[1])
    loss = 0.0
    for (i, ti) in enumerate(tgrid)
        sol(u, ti)
        all(isfinite, u) || return Inf
        for (k, idx) in enumerate(air_idx)
            r = u[idx] - data_air[i, k]
            loss += w[i] * r * r
        end
    end
    return loss
end

function air_node_residual_loss(topo::BuildingGraphModel, u0, tspan, p, data_air, tgrid;
                                alg = Tsit5(), reltol = 1e-8, abstol = 1e-10)
    air_idx = topo.hvac_nodes
    _check_air_node_data(air_idx, data_air, length(tgrid))
    w = _air_node_quadrature_weights(tgrid)

    # Dense forward solve (no `saveat`): `saveat` disables the dense interpolant,
    # which would force the adjoint to checkpoint/re-solve forward intervals.
    sol = solve(ODEProblem(_odef(topo), u0, tspan, p), alg;
                reltol = reltol, abstol = abstol)
    return _air_node_residual_loss(sol, air_idx, data_air, w, tgrid)
end

function air_node_residual_loss_and_gradient(topo::BuildingGraphModel, u0, tspan, p,
                                             data_air, tgrid;
                                             alg = Tsit5(),
                                             sensealg = InterpolatingAdjoint(autojacvec = false),
                                             reltol = 1e-8, abstol = 1e-10)
    air_idx = topo.hvac_nodes
    _check_air_node_data(air_idx, data_air, length(tgrid))
    w = _air_node_quadrature_weights(tgrid)

    # Dense forward solve (see `air_node_residual_loss`): this avoids adjoint
    # checkpointing and its large allocation overhead.
    sol = solve(ODEProblem(_odef(topo), u0, tspan, p), alg;
                reltol = reltol, abstol = abstol)
    loss = _air_node_residual_loss(sol, air_idx, data_air, w, tgrid)
    isfinite(loss) || return (Inf, zero(p))

    # discrete cost gradient at each data time: ∂/∂u [wᵢ ‖u_air − data_air[i,:]‖²]
    function dgdu_discrete(out, u, p_, t, i)
        fill!(out, 0.0)
        for (k, idx) in enumerate(air_idx)
            out[idx] = 2 * w[i] * (u[idx] - data_air[i, k])
        end
        nothing
    end

    _, dp = adjoint_sensitivities(sol, alg; t = tgrid, dgdu_discrete = dgdu_discrete,
                                  sensealg = sensealg, reltol = reltol, abstol = abstol)

    grad = p isa ComponentArray ? ComponentArray(vec(dp), getaxes(p)) : vec(dp)
    return loss, grad
end

air_node_residual_gradient(topo::BuildingGraphModel, u0, tspan, p, data_air, tgrid; kwargs...) =
    air_node_residual_loss_and_gradient(topo, u0, tspan, p, data_air, tgrid; kwargs...)[2]

spAir = 0.24
spBrick = 0.22
densityAir = .075
densityBrick = 120.0

z1sqft = 5000
z2sqft = 10000

floorToRoof = 14.0
exteriorWallThickness = 0.5
interiorWallThickness = 0.5
peakTonPerSqftEst = 1/500.0
z1PeakCoolingLoad = z1sqft * peakTonPerSqftEst 
z2PeakCoolingLoad = z2sqft * peakTonPerSqftEst

z1ExteriorWallft3 = (sqrt(z1sqft) * floorToRoof * exteriorWallThickness) * 3.0 + (sqrt(z1sqft) * exteriorWallThickness) 
z1InteriorWallft3 = (sqrt(z1sqft) * floorToRoof * interiorWallThickness) 
z2ExteriorWallft3 = (sqrt(z2sqft) * floorToRoof * exteriorWallThickness) * 3.0 + (sqrt(z2sqft) * exteriorWallThickness)

c1Air = 1 / (z1sqft * floorToRoof * densityAir * spAir / 12000.0)
c1Brick = 1 / (z1ExteriorWallft3 * densityBrick * spBrick / 12000.0)
c1InteriorWall = 1 / (z1InteriorWallft3 * densityBrick * spBrick / 12000.0)
c2Air = 1 / (z2sqft * floorToRoof * densityAir * spAir / 12000.0)
c2Brick = 1 / (z2ExteriorWallft3 * densityBrick * spBrick / 12000.0)

z1 = ZoneDef(:z1; cAir = c1Air, cMass = c1Air / 5.0, rAirAm = 0.25, rAirMass = 0.1, α = 0.01)
z2 = ZoneDef(:z2; cAir = c2Air, cMass = c2Air / 5.0, rAirAm = 0.25, rAirMass = 0.1, α = 0.0)

w1 = WallDef(:w1, :z1, :z2, 2; c = [c1InteriorWall * 2.0, c1InteriorWall * 2.0], r = [0.1, 0.1, 0.1], α_a = 0.0, α_b = 0.0)
w2 = WallDef(:w2, :z1, :ambient, 2; c = [c1Brick * 2.0, c1Brick * 2.0], r = [1.0, 1.0, 1.0], α_a = 0.0, α_b = 0.0)
w3 = WallDef(:w3, :z2, :ambient, 2; c = [c2Brick * 2.0, c2Brick * 2.0], r = [1.0, 1.0, 1.0], α_a = 0.0, α_b = 0.0)

# 72 hours of synthetic data
t_sim = 72
n_steps_interp = 600
n_steps_modeled = 100

t_steps = collect(range(0, t_sim, length=n_steps_modeled))

n_zones = 2
T_oa, I_solar, Q_int, Q_hvac = generate_sample_exogenous_inputs((0.0, t_sim), n_steps_interp, n_zones;
    meanTemp = 72.0, varTemp = 10.0, amp_mod_amp = 15.0, amp_mod_phase = π, amp_mod_freq = 1/(3 * t_sim))

plot(layer(x = t_steps, y = [T_oa(t) for t in t_steps], Geom.line, Theme(default_color = colorant"blue")),
     layer(x = t_steps, y = [I_solar(t) for t in t_steps], Geom.line, Theme(default_color = colorant"orange")),
     Guide.xlabel("Time (hours)"), Guide.ylabel("Exogenous Inputs"), Guide.title("Outdoor Temperature and Solar Irradiance"))
plot(layer(x = t_steps, y = [Q_int[1](t) for t in t_steps], Geom.line, Theme(default_color = colorant"blue")),
     layer(x = t_steps, y = [Q_hvac[1](t) for t in t_steps], Geom.line, Theme(default_color = colorant"orange")),
     Guide.xlabel("Time (hours)"), Guide.ylabel("Exogenous Inputs"), Guide.title("Outdoor Temperature and Solar Irradiance"))

topo = BuildingGraphModel([z1, z2], [w1, w2, w3], T_oa, I_solar, Q_int, Q_hvac)



# initial conditions 
u0_true = copy(topo.params.C)
u0_true .= 72.0
p_true_log = log.(topo.params)

prob = ODEProblem(topo, u0_true, (0.0, t_sim), p_true_log)

@time sol_true = solve(prob, Tsit5(), reltol = 1e-6, abstol = 1e-6, saveat = t_steps)
#@time sol_true = solve(prob, KenCarp4(); reltol = 1e-6, abstol = 1e-6, saveat = t_steps)

# plot the air temps (sol.u[1] and sol.u[3]) 

plot(layer(x = t_steps, y = [sol_true.u[i][1] for i in 1:length(t_steps)], Geom.line, Theme(default_color = colorant"blue")),
     layer(x = t_steps, y = [sol_true.u[i][3] for i in 1:length(t_steps)], Geom.line, Theme(default_color = colorant"orange")),
     Guide.xlabel("Time (hours)"), Guide.ylabel("Zone Air Temperatures (°F)"), Guide.title("Zone Air Temperatures Over Time"))






#=
Model parameter recovery
=#


# observed air-node trajectory from the truth solve ----------------------------
air_idx = topo.hvac_nodes
n_air   = length(air_idx)
data_air = permutedims(reduce(hcat, [u[air_idx] for u in sol_true.u]))   # n_t ×  n_air

k = 50 # first ~8 h of the saved grid
tchk = t_steps[1:k]
dchk = data_air[1:k, :]


# generate intial guess for parameters in log-space
Random.seed!(1)
p_init_log = p_true_log .+ 0.2 .* randn(length(p_true_log))


loss_short(p) = air_node_residual_loss(topo, u0_true, (0.0, tchk[end]), p, dchk, tchk;
                                       reltol = 1e-10, abstol = 1e-12)

@time g_adj = air_node_residual_gradient(topo, u0_true, (0.0, tchk[end]), p_init_log, dchk, tchk;
                                   reltol = 1e-10, abstol = 1e-12)
@time g_fwd = ForwardDiff.gradient(loss_short, p_init_log)

println("=== adjoint_sensitivities vs ForwardDiff (short window) ===")
println("max |g_adj - g_fwd| = ", round(maximum(abs.(vec(g_adj) .- g_fwd)), sigdigits=4))
println("relative L2 error   = ", round(norm(vec(g_adj) - g_fwd) / norm(g_fwd), sigdigits=4))


# -----------------------------------------------------------------------------
# Solve the full trajectory-matching problem with Optimization.jl.  The adjoint
# gradient from `air_node_residual_loss_and_gradient` is supplied as both `grad`
# and `fg`, so LBFGS needs only one forward+adjoint solve per iteration.
# -----------------------------------------------------------------------------
param_axes = getaxes(p_true_log)
pack(s)    = ComponentArray(s, param_axes)

function opt_objective(u, p)
    try
        loss = air_node_residual_loss(topo, u0_true, (0.0, t_sim), pack(u), data_air, t_steps)
        return isfinite(loss) ? loss : 1e12
    catch
        return 1e12
    end
end

function opt_grad(G, u, p)
    try
        _, g = air_node_residual_loss_and_gradient(topo, u0_true, (0.0, t_sim),
                                                   pack(u), data_air, t_steps)
        all(isfinite, vec(g)) ? (G .= vec(g)) : fill!(G, 0.0)
    catch
        fill!(G, 0.0)
    end
    return nothing
end

function opt_fg(G, u, p)
    local loss, g
    try
        loss, g = air_node_residual_loss_and_gradient(topo, u0_true, (0.0, t_sim),
                                                      pack(u), data_air, t_steps)
    catch
        G !== nothing && fill!(G, 0.0)
        return 1e12
    end

    if isfinite(loss) && all(isfinite, vec(g))
        G !== nothing && (G .= vec(g))
        return loss
    else
        G !== nothing && fill!(G, 0.0)
        return 1e12
    end
end

optf = OptimizationFunction(opt_objective, Optimization.NoAD(); grad = opt_grad, fg = opt_fg)
u0_opt = collect(p_init_log)             # plain Vector{Float64} for the solver

# Keep the log-parameters inside a physically safe box and let the objective
# return a large finite sentinel for any trial point where the forward solve
# fails, so the line search backtracks instead of aborting.
lb_opt = fill(-6.0, length(u0_opt))
ub_opt = fill(3.0, length(u0_opt))
optprob = OptimizationProblem(optf, u0_opt, nothing; lb = lb_opt, ub = ub_opt)

println("\ninitial loss = ", round(opt_objective(u0_opt, nothing), sigdigits = 5))

# Stop once the air-node residual has been driven to ~1e-3 (a near-perfect fit);
# asking NLopt to refine further makes its LBFGS line search stall at numerical
# precision and report a generic failure.

nlopt_alg = NLopt.Opt(:LD_LBFGS, length(u0_opt))
NLopt.maxeval!(nlopt_alg, 300)
NLopt.stopval!(nlopt_alg, .005)

@time opt_sol = solve(optprob, nlopt_alg; verbose = true, show_trace = true, show_every = 1)


p_rec_log = pack(opt_sol.u)
println("\n=== Optimization.jl (NLopt LD_LBFGS) recovery ===")
println("retcode               = ", opt_sol.retcode)
println("final loss            = ", round(opt_objective(opt_sol.u, nothing), sigdigits = 5))
println("||p - p_true|| (log)  = ", round(norm(p_rec_log .- p_true_log), sigdigits = 5))
println("max |p - p_true|      = ", round(maximum(abs.(p_rec_log .- p_true_log)), sigdigits = 5))