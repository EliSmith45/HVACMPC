using LinearAlgebra, ComponentArrays, DataInterpolations
using DifferentialEquations, SciMLSensitivity
using Optimization, OptimizationNLopt, Optim, NLopt
using ReverseDiff, Zygote
using ForwardDiff
using Random
using Gadfly

#= structs =#

# Number of hourly knots in a solar-gain profile: one coefficient per hour of
# day (0:00–23:00), interpolated periodically across the day boundary.
const SOLAR_PROFILE_LENGTH = 24

# Accept either a scalar (constant all day) or a 24-element hourly vector.
_as_solar_profile(α::Real) = fill(Float64(α), SOLAR_PROFILE_LENGTH)
function _as_solar_profile(α::AbstractVector)
    length(α) == SOLAR_PROFILE_LENGTH || throw(DimensionMismatch(
        "solar gain profile must have $(SOLAR_PROFILE_LENGTH) hourly entries, got $(length(α))"))
    return collect(Float64.(α))
end

struct ZoneDef
    id::Symbol
    cAir::Float64
    cMass::Float64
    rAirAm::Float64
    rAirMass::Float64
    α::Vector{Float64} #hourly solar heat gain coefficients to thermal mass (24 h)
    has_solar::Bool
    has_infiltration::Bool #heat transfer from ambient direct to air, bypassing walls
end

function ZoneDef(id::Symbol; cAir=100.0, cMass=1000.0, rAirAm=0.001, rAirMass = 0.0001, α=0.0)
    α = _as_solar_profile(α)
    return ZoneDef(id, cAir, cMass, rAirAm, rAirMass, α, any(!iszero, α), rAirAm != 0.0)
end

struct WallDef
    id::Symbol
    zone_a::Symbol
    zone_b::Symbol
    num_nodes::Int
    c::Vector{Float64}
    r::Vector{Float64}
    α_a::Vector{Float64} #hourly solar to the wall capacitor on side a (24 h)
    α_b::Vector{Float64} #hourly solar to the wall capacitor on side b (24 h)
end

function WallDef(id::Symbol, zone_a::Symbol, zone_b::Symbol, num_nodes::Int; c=fill(100.0, num_nodes), r=fill(0.001, num_nodes + 1), α_a=0.0, α_b=0.0)
    @assert length(c) == num_nodes "Length of c must equal num_nodes"
    @assert length(r) == num_nodes + 1 "Length of r must equal num_nodes + 1"

    α_a = _as_solar_profile(α_a)
    α_b = _as_solar_profile(α_b)
    return WallDef(id, zone_a, zone_b, num_nodes, c, r, α_a, α_b)
end

#= Graph model topology =#

struct BuildingGraphModel{T_OA, T_SOL, T_HVAC}
    num_nodes::Int
    node_names::Vector{Symbol}

    #internal conduction graph edges: edge e connects edge_src[e] to edge_dst[e]
    edge_src::Vector{Int}
    edge_dst::Vector{Int}

    #external ambient connections: node ambient_nodes[k] connects to T_oa
    ambient_nodes::Vector{Int}

    #solar connections: node solar_nodes[k] connects to T_sol
    solar_nodes::Vector{Int}

    #HVAC/air node list (one per zone; the air node receives Q_hvac and Q_int)
    hvac_nodes::Vector{Int}

    #daily occupancy schedule, one entry per zone (hour of day on a 24 h clock)
    occ_start::Vector{Float64}
    occ_end::Vector{Float64}

    #differentiable array parameters, stored in ABSOLUTE (physical) units:
    #  C, R_internal, R_ambient : structural RC parameters
    #  α_solar                  : hourly solar gain coefficients (n_solar × 24)
    #  Q_int_occ, Q_int_unocc   : internal heat gains (per zone)
    # The ODE right-hand side works in log-space; `train!` maps absolute ↔ log
    # and writes learned values back here in absolute form.
    params::ComponentArray

    #known, time-varying exogenous inputs (NOT learned parameters)
    T_oa::T_OA      # outdoor air temperature
    I_solar::T_SOL  # solar irradiance
    Q_hvac::T_HVAC  # delivered HVAC load (control input), one interpolator per zone
    
end

function BuildingGraphModel(zones::Vector{ZoneDef}, walls::Vector{WallDef}, 
    T_oa, I_solar, Q_hvac, occ_start, occ_end)

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
    alpha_solar = Matrix{Float64}(undef, 0, SOLAR_PROFILE_LENGTH)

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
            alpha_solar = vcat(alpha_solar, reshape(z.α, 1, SOLAR_PROFILE_LENGTH))
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
        if any(!iszero, w.α_a)
            push!(solar_nodes, w_nodes[1])
            alpha_solar = vcat(alpha_solar, reshape(w.α_a, 1, SOLAR_PROFILE_LENGTH))
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
        if any(!iszero, w.α_b)
            push!(solar_nodes, w_nodes[K])
            alpha_solar = vcat(alpha_solar, reshape(w.α_b, 1, SOLAR_PROFILE_LENGTH))
        end
    end

    # create component array for parameters
    n_zones = length(hvac_nodes)
    @assert length(occ_start) == n_zones "occ_start must have one entry per zone"
    @assert length(occ_end) == n_zones "occ_end must have one entry per zone"

    params = ComponentArray(
        C = capacitances,
        R_internal = r_internal,
        R_ambient = r_ambient,
        α_solar = alpha_solar,
        Q_int_occ = fill(1.0, n_zones),     # absolute nominal gain (log 0) until trained
        Q_int_unocc = fill(1.0, n_zones)    # absolute nominal gain (log 0) until trained
    )

    return BuildingGraphModel(
        total_nodes,
        node_names,
        edge_src,
        edge_dst,
        ambient_nodes,
        solar_nodes,
        hvac_nodes,
        occ_start,
        occ_end,
        params,
        T_oa,
        I_solar,
        Q_hvac
    )
end

# Daily occupancy indicator: 1.0 during [occ_start, occ_end) hours of each day,
# 0.0 otherwise.  `occ_start`/`occ_end` are hours on a 24 h clock and the window
# may wrap across midnight (occ_start >= occ_end).
@inline function _occupancy(t, occ_start, occ_end)
    tod = mod(t, 24.0)
    if occ_start < occ_end
        return (occ_start <= tod < occ_end) ? 1.0 : 0.0
    else
        return (tod >= occ_start || tod < occ_end) ? 1.0 : 0.0
    end
end

# Hour-of-day lookup for the solar-gain profile: the 24 hourly knots live at
# hours 0..23 and the profile repeats with period 24 h.
@inline function _solar_hour(t)
    tod = mod(t, 24.0)
    h = floor(Int, tod)
    frac = tod - h
    return h, mod(h + 1, SOLAR_PROFILE_LENGTH), frac
end

# Periodic piecewise-linear interpolation of an hourly solar-gain profile row
# (knots at hours 0..23, wrapping across the day boundary).
@inline function _solar_coeff_interp(α, i, t)
    h0, h1, frac = _solar_hour(t)
    @inbounds return α[i, h0 + 1] * (1.0 - frac) + α[i, h1 + 1] * frac
end

# Absolute heat-gain coefficient from a LOG-space profile (`α_log` holds
# log-coefficients; the interpolated log value is exponentiated).
@inline _solar_coefficient(α_log, i, t) = exp(_solar_coeff_interp(α_log, i, t))

# Absolute heat-gain coefficient from an ABSOLUTE-space profile (no `exp`).
@inline _solar_coefficient_abs(α_abs, i, t) = _solar_coeff_interp(α_abs, i, t)

# In-place right-hand side in LOG-space.  `p` holds parameters in log-space
# (`exp.(p.X)` recovers the absolute values); `model.params` stores the absolute
# values and `train!` performs the mapping.
#
# Sign convention: `Q_hvac > 0` and `Q_int > 0` add heat to the zone air node
# (raise its temperature).  Cooling must therefore be supplied as a negative
# `Q_hvac`; the units of `Q_hvac`, `Q_int` and the model capacitances must agree.
@inline function _rhs_log!(du, u, p, t, m::BuildingGraphModel)
    t_oa = m.T_oa(t)
    i_solar = m.I_solar(t)

    du .= 0.0

    @inbounds for (i, k) in enumerate(m.ambient_nodes)
        du[k] += (t_oa - u[k]) * exp(p.R_ambient[i])
    end

    @inbounds for (i, k) in enumerate(m.solar_nodes)
        du[k] += i_solar * _solar_coefficient(p.α_solar, i, t)
    end

    @inbounds for (i, k) in enumerate(m.hvac_nodes)
        occ = _occupancy(t, m.occ_start[i], m.occ_end[i])
        q_int = exp(p.Q_int_occ[i]) * occ + exp(p.Q_int_unocc[i]) * (1.0 - occ)
        du[k] += m.Q_hvac[i](t) + q_int
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

# In-place right-hand side in ABSOLUTE space: `p` holds physical parameter values
# directly, with no per-step `exp`.  This is the cheaper forward-simulation form
# (used e.g. in MPC loops) and can also be selected for training via
# `train!(...; log_space = false)`.
@inline function _rhs_abs!(du, u, p, t, m::BuildingGraphModel)
    t_oa = m.T_oa(t)
    i_solar = m.I_solar(t)

    du .= 0.0

    @inbounds for (i, k) in enumerate(m.ambient_nodes)
        du[k] += (t_oa - u[k]) * p.R_ambient[i]
    end

    @inbounds for (i, k) in enumerate(m.solar_nodes)
        du[k] += i_solar * _solar_coefficient_abs(p.α_solar, i, t)
    end

    @inbounds for (i, k) in enumerate(m.hvac_nodes)
        occ = _occupancy(t, m.occ_start[i], m.occ_end[i])
        q_int = p.Q_int_occ[i] * occ + p.Q_int_unocc[i] * (1.0 - occ)
        du[k] += m.Q_hvac[i](t) + q_int
    end

    @inbounds for i in eachindex(m.edge_src)
        g = p.R_internal[i]
        du[m.edge_dst[i]] += (u[m.edge_src[i]] - u[m.edge_dst[i]]) * g
        du[m.edge_src[i]] += (u[m.edge_dst[i]] - u[m.edge_src[i]]) * g
    end

    @inbounds for i in eachindex(du)
        du[i] *= p.C[i]
    end

    return nothing
end

# Compile-time parameterization dispatch for the right-hand side.
@inline _rhs!(du, u, p, t, m::BuildingGraphModel, ::Val{true}) = _rhs_log!(du, u, p, t, m)
@inline _rhs!(du, u, p, t, m::BuildingGraphModel, ::Val{false}) = _rhs_abs!(du, u, p, t, m)

# The callable struct is a log-space convenience for interactive use; the
# parameterization for an `ODEProblem` is chosen at `_odef`/`train!` construction
# time instead (so the struct needs no parameterization type parameter).
function (m::BuildingGraphModel)(du, u, p, t)
    _rhs_log!(du, u, p, t, m)
end

# Analytical state Jacobian ∂f/∂u for the linear RC model.  The Jacobian is
# independent of `u` (and of `t`), and only depends on the conductance/capacitance
# parameters `p`.
@inline function _jac_log!(J, u, p, t, m::BuildingGraphModel)
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

# Absolute-space state Jacobian: identical structure to `_jac_log!` but with
# `p.C`/`p.R_*` used directly (no `exp`).
@inline function _jac_abs!(J, u, p, t, m::BuildingGraphModel)
    fill!(J, 0.0)
    n = m.num_nodes

    @inbounds begin
        for (i, k) in enumerate(m.ambient_nodes)
            J[k, k] -= p.R_ambient[i]
        end

        for i in eachindex(m.edge_src)
            g = p.R_internal[i]
            s = m.edge_src[i]
            d = m.edge_dst[i]
            J[d, d] -= g
            J[s, s] -= g
            J[d, s] += g
            J[s, d] += g
        end

        for i in 1:n
            ci = p.C[i]
            for j in 1:n
                J[i, j] *= ci
            end
        end
    end

    return nothing
end

@inline _jac!(J, u, p, t, m::BuildingGraphModel, ::Val{true}) = _jac_log!(J, u, p, t, m)
@inline _jac!(J, u, p, t, m::BuildingGraphModel, ::Val{false}) = _jac_abs!(J, u, p, t, m)

# Analytical parameter Jacobian ∂f/∂p in LOG-space.  `offC`, `offRi`, `offRa`,
# `offAs`, `offQo`, `offQu` are the 1-based column offsets of each parameter
# block in the flat `ComponentArray` data vector (C, R_internal, R_ambient,
# α_solar, Q_int_occ, Q_int_unocc).  `α_solar` is an n_solar × 24 matrix laid
# out column-major in the flat vector, so column (i, h) sits at
# `offAs + (i - 1) + h * n_solar` (h is 0-based).
@inline function _paramjac_log!(pJ, u, p, t, m::BuildingGraphModel,
                                scratch, offC, offRi, offRa, offAs, offQo, offQu)
    fill!(pJ, 0.0)
    db = scratch
    fill!(db, 0.0)

    t_oa = m.T_oa(t)
    i_solar = m.I_solar(t)
    n_solar = length(m.solar_nodes)

    @inbounds begin
        # Unscaled RHS, before the final per-node `* exp(p.C)` scaling.
        for (i, k) in enumerate(m.ambient_nodes)
            db[k] += (t_oa - u[k]) * exp(p.R_ambient[i])
        end
        for (i, k) in enumerate(m.solar_nodes)
            db[k] += i_solar * _solar_coefficient(p.α_solar, i, t)
        end
        for (i, k) in enumerate(m.hvac_nodes)
            occ = _occupancy(t, m.occ_start[i], m.occ_end[i])
            db[k] += m.Q_hvac[i](t) + exp(p.Q_int_occ[i]) * occ +
                     exp(p.Q_int_unocc[i]) * (1.0 - occ)
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
        # ∂/∂(log α_solar[i, h]) = I_solar * α(t) * hat_h(t) * exp(C_k), where
        # hat_h is the piecewise-linear interpolation basis for hour h.
        for (i, k) in enumerate(m.solar_nodes)
            h0, h1, frac = _solar_hour(t)
            α = _solar_coefficient(p.α_solar, i, t)
            pJ[k, offAs + (i - 1) + h0 * n_solar] = i_solar * α * (1.0 - frac) * exp(p.C[k])
            pJ[k, offAs + (i - 1) + h1 * n_solar] = i_solar * α * frac * exp(p.C[k])
        end
        for i in eachindex(m.edge_src)
            g = exp(p.R_internal[i])
            s = m.edge_src[i]
            d = m.edge_dst[i]
            pJ[d, offRi + i - 1] = (u[s] - u[d]) * g * exp(p.C[d])
            pJ[s, offRi + i - 1] = (u[d] - u[s]) * g * exp(p.C[s])
        end
        for (i, k) in enumerate(m.hvac_nodes)
            occ = _occupancy(t, m.occ_start[i], m.occ_end[i])
            pJ[k, offQo + i - 1] = exp(p.Q_int_occ[i]) * occ * exp(p.C[k])
            pJ[k, offQu + i - 1] = exp(p.Q_int_unocc[i]) * (1.0 - occ) * exp(p.C[k])
        end
    end

    return nothing
end

# Analytical parameter Jacobian ∂f/∂p in ABSOLUTE space (same layout as above,
# but with the chain-rule `exp` factors dropped).
@inline function _paramjac_abs!(pJ, u, p, t, m::BuildingGraphModel,
                                scratch, offC, offRi, offRa, offAs, offQo, offQu)
    fill!(pJ, 0.0)
    db = scratch
    fill!(db, 0.0)

    t_oa = m.T_oa(t)
    i_solar = m.I_solar(t)
    n_solar = length(m.solar_nodes)

    @inbounds begin
        # Unscaled RHS, before the final per-node `* p.C` scaling.
        for (i, k) in enumerate(m.ambient_nodes)
            db[k] += (t_oa - u[k]) * p.R_ambient[i]
        end
        for (i, k) in enumerate(m.solar_nodes)
            db[k] += i_solar * _solar_coefficient_abs(p.α_solar, i, t)
        end
        for (i, k) in enumerate(m.hvac_nodes)
            occ = _occupancy(t, m.occ_start[i], m.occ_end[i])
            db[k] += m.Q_hvac[i](t) + p.Q_int_occ[i] * occ +
                     p.Q_int_unocc[i] * (1.0 - occ)
        end
        for i in eachindex(m.edge_src)
            g = p.R_internal[i]
            s = m.edge_src[i]
            d = m.edge_dst[i]
            db[d] += (u[s] - u[d]) * g
            db[s] += (u[d] - u[s]) * g
        end

        # ∂/∂C_k of `db_k * C_k` is `db_k`.
        for k in 1:m.num_nodes
            pJ[k, offC + k - 1] = db[k]
        end

        for (i, k) in enumerate(m.ambient_nodes)
            pJ[k, offRa + i - 1] = (t_oa - u[k]) * p.C[k]
        end
        # ∂/∂α_solar[i, h] = I_solar * hat_h(t) * C_k.
        for (i, k) in enumerate(m.solar_nodes)
            h0, h1, frac = _solar_hour(t)
            pJ[k, offAs + (i - 1) + h0 * n_solar] = i_solar * (1.0 - frac) * p.C[k]
            pJ[k, offAs + (i - 1) + h1 * n_solar] = i_solar * frac * p.C[k]
        end
        for i in eachindex(m.edge_src)
            s = m.edge_src[i]
            d = m.edge_dst[i]
            pJ[d, offRi + i - 1] = (u[s] - u[d]) * p.C[d]
            pJ[s, offRi + i - 1] = (u[d] - u[s]) * p.C[s]
        end
        for (i, k) in enumerate(m.hvac_nodes)
            occ = _occupancy(t, m.occ_start[i], m.occ_end[i])
            pJ[k, offQo + i - 1] = occ * p.C[k]
            pJ[k, offQu + i - 1] = (1.0 - occ) * p.C[k]
        end
    end

    return nothing
end

@inline function _paramjac!(pJ, u, p, t, m::BuildingGraphModel,
                            scratch, offC, offRi, offRa, offAs, offQo, offQu,
                            ::Val{true})
    _paramjac_log!(pJ, u, p, t, m, scratch, offC, offRi, offRa, offAs, offQo, offQu)
end
@inline function _paramjac!(pJ, u, p, t, m::BuildingGraphModel,
                            scratch, offC, offRi, offRa, offAs, offQo, offQu,
                            ::Val{false})
    _paramjac_abs!(pJ, u, p, t, m, scratch, offC, offRi, offRa, offAs, offQo, offQu)
end

# Build an `ODEFunction` with analytical state/parameter Jacobians so that
# implicit solvers and `InterpolatingAdjoint(autojacvec = false)` can avoid
# automatic differentiation of the right-hand side altogether.  `log_space`
# selects the parameterization used by `p` (true: log-space, false: absolute).
function _odef(m::BuildingGraphModel; log_space::Bool = true)
    n = m.num_nodes
    nC = length(m.params.C)
    nRi = length(m.params.R_internal)
    nRa = length(m.params.R_ambient)
    nAs = length(m.params.α_solar)
    nQo = length(m.params.Q_int_occ)
    nQu = length(m.params.Q_int_unocc)
    @assert nC == n
    @assert nQo == nQu
    @assert nC + nRi + nRa + nAs + nQo + nQu == length(m.params)

    offC = 1
    offRi = offC + nC
    offRa = offRi + nRi
    offAs = offRa + nRa
    offQo = offAs + nAs
    offQu = offQo + nQo

    scratch = Vector{Float64}(undef, n)
    ls = Val(log_space)

    f = (du, u, p, t) -> _rhs!(du, u, p, t, m, ls)
    jac = (J, u, p, t) -> _jac!(J, u, p, t, m, ls)
    paramjac = (pJ, u, p, t) -> _paramjac!(pJ, u, p, t, m, scratch,
                                           offC, offRi, offRa, offAs, offQo, offQu, ls)

    return ODEFunction{true}(f; jac = jac, paramjac = paramjac,
                             jac_prototype = zeros(n, n))
end

# -----------------------------------------------------------------------------
# Forward (non-training) simulation.  For MPC we rebuild the ODE problem every
# loop with fresh weather/HVAC forecasts, so the exogenous inputs are passed
# through `p` rather than closed over in the model struct.  Only the topology
# (node lists, edges, occupancy schedule) is reused from `m`.
# -----------------------------------------------------------------------------

"""
    Exogenous(T_oa, I_solar, Q_hvac)

Bundle of known time-varying inputs carried through `p` for forward simulation.
`T_oa`/`I_solar` are outdoor temperature / solar irradiance callables and
`Q_hvac` is a tuple of per-zone HVAC-load callables.
"""
struct Exogenous{T_OA, T_SOL, T_HVAC}
    T_oa::T_OA
    I_solar::T_SOL
    Q_hvac::T_HVAC
end

# Absolute-space right-hand side reading parameters and exogenous inputs from
# `p = (; params, exo)` (a NamedTuple).  No `exp`, and no reliance on the
# interpolators stored in `m`.
@inline function _rhs_forward!(du, u, p, t, m::BuildingGraphModel)
    prm = p.params
    exo = p.exo
    t_oa = exo.T_oa(t)
    i_solar = exo.I_solar(t)

    du .= 0.0

    @inbounds for (i, k) in enumerate(m.ambient_nodes)
        du[k] += (t_oa - u[k]) * prm.R_ambient[i]
    end

    @inbounds for (i, k) in enumerate(m.solar_nodes)
        du[k] += i_solar * _solar_coefficient_abs(prm.α_solar, i, t)
    end

    @inbounds for (i, k) in enumerate(m.hvac_nodes)
        occ = _occupancy(t, m.occ_start[i], m.occ_end[i])
        q_int = prm.Q_int_occ[i] * occ + prm.Q_int_unocc[i] * (1.0 - occ)
        du[k] += exo.Q_hvac[i](t) + q_int
    end

    @inbounds for i in eachindex(m.edge_src)
        g = prm.R_internal[i]
        du[m.edge_dst[i]] += (u[m.edge_src[i]] - u[m.edge_dst[i]]) * g
        du[m.edge_src[i]] += (u[m.edge_dst[i]] - u[m.edge_src[i]]) * g
    end

    @inbounds for i in eachindex(du)
        du[i] *= prm.C[i]
    end

    return nothing
end

"""
    make_forward_odefun(m::BuildingGraphModel) -> ODEFunction

Build an `ODEFunction` for pure forward simulation.  Its `p` must be
`(; params, exo)` with `params` in **absolute** units and `exo` an
[`Exogenous`](@ref).  The state Jacobian is supplied analytically; no parameter
Jacobian is provided (forward simulation does not differentiate `p`).
"""
function make_forward_odefun(m::BuildingGraphModel)
    f = (du, u, p, t) -> _rhs_forward!(du, u, p, t, m)
    jac = (J, u, p, t) -> _jac_abs!(J, u, p.params, t, m)
    return ODEFunction{true}(f; jac = jac,
                             jac_prototype = zeros(m.num_nodes, m.num_nodes))
end

"""
    make_forward_problem(m, u0, tspan, params_abs, T_oa, I_solar, Q_hvac) -> ODEProblem

Build a forward-simulation `ODEProblem` whose `p` carries the (absolute) model
parameters and the exogenous inputs.  Reuse this with fresh forecasts in each
MPC iteration without rebuilding the model struct.
"""
function make_forward_problem(m::BuildingGraphModel, u0, tspan, params_abs,
                              T_oa, I_solar, Q_hvac)
    p = (; params = params_abs, exo = Exogenous(T_oa, I_solar, Q_hvac))
    return ODEProblem(make_forward_odefun(m), u0, tspan, p)
end

# -----------------------------------------------------------------------------
# Exogenous input generation.  Weather (T_oa, I_solar) and the HVAC control
# input (Q_hvac) are generated separately: weather is an uncontrollable external
# disturbance, while Q_hvac is a *known* control input that we choose.  Internal
# gains Q_int are NOT generated here — they are learned parameters (occupancy
# coefficients) living in `BuildingGraphModel.params`.
# -----------------------------------------------------------------------------

"""
    generate_sample_weather(tspan, steps; kwargs...) -> (T_oa, I_solar)

Synthetic outdoor-air temperature and solar irradiance, for testing before real
OpenMeteo weather is wired in.  Returns `LinearInterpolation` objects.
"""
function generate_sample_weather(tspan, steps;
                                 meanTemp=72.0, varTemp=10.0,
                                 amp_mod_amp=15.0, amp_mod_phase=π,
                                 amp_mod_freq=1/(3 * tspan[2]),
                                 peakSolar=500.0)
    t_steps = collect(range(tspan[1], tspan[2], length=steps))
    T_oa_data = (meanTemp .+ (varTemp .* sin.(2π .* t_steps ./ 24.0 .- π/2))) .+
                (amp_mod_amp .* sin.(2π .* amp_mod_freq .* t_steps .+ amp_mod_phase))
    I_solar_data = peakSolar .* max.(0.0, sin.(2π .* t_steps ./ 24.0 .- π/2))

    T_oa = LinearInterpolation(T_oa_data, t_steps)
    I_solar = LinearInterpolation(I_solar_data, t_steps)
    return T_oa, I_solar
end

"""
    generate_sample_control(tspan, steps, num_zones; kwargs...) -> Q_hvac

Synthetic open-loop HVAC control input (one interpolator per zone).  For
parameter identifiability the input should be persistently exciting; a
sinusoid at a single frequency is a weak probe, so this uses a random
piecewise-constant (PRBS-like) signal with a new level every `hold_hours`.
`Q_hvac > 0` adds heat to the zone (see `_rhs!`); cooling would be negative.
"""
function generate_sample_control(tspan, steps, num_zones;
                                 hold_hours=2.0,
                                 amplitudes=fill(10.0, num_zones),
                                 seed=1234)
    t_steps = collect(range(tspan[1], tspan[2], length=steps))
    Q_hvac_data = Vector{Vector{Float64}}(undef, num_zones)
    for z in 1:num_zones
        rng = Random.MersenneTwister(seed + 1000 * z)
        amp = amplitudes[z]
        q = Vector{Float64}(undef, steps)
        last_bin = typemin(Int)
        level = 0.0
        for (j, t) in enumerate(t_steps)
            bin = floor(Int, t / hold_hours)
            if bin != last_bin
                # deterministic per-interval random level in [-amp, amp]
                level = (2.0 * rand(rng) - 1.0) * amp
                last_bin = bin
            end
            q[j] = level
        end
        Q_hvac_data[z] = q
    end
    return Tuple([LinearInterpolation(q, t_steps) for q in Q_hvac_data])
end

function update_exo_inputs(model::BuildingGraphModel, T_oa, I_solar, Q_hvac)
    return BuildingGraphModel(
        model.node_names,
        model.edge_src,
        model.edge_dst,
        model.ambient_nodes,
        model.solar_nodes,
        model.hvac_nodes,
        model.occ_start,
        model.occ_end,
        model.params,
        T_oa,
        I_solar,
        Q_hvac
    )
end

# -----------------------------------------------------------------------------
# Observation handling.  Observed air-node temperatures may come from a forward
# solve (synthetic) or from real sensors with arbitrary, irregular sampling
# times.  In both cases they are represented as one interpolator (or callable)
# per air node, so the loss/gradient machinery never needs the observation times
# to align with any internally chosen grid.
# -----------------------------------------------------------------------------

"""
    observation_interpolators(data_air, tobs) -> (T_obs, ts_sorted)

Build one `LinearInterpolation` per air node from an `n_t × n_air` matrix of
observations sampled at `tobs`.  Returns the tuple of interpolators together
with the sorted observation times (interpolators require sorted times).
"""
function observation_interpolators(data_air::AbstractMatrix, tobs)
    n_t, n_air = size(data_air)
    n_t == length(tobs) || throw(DimensionMismatch(
        "data_air has $(n_t) rows but $(length(tobs)) observation times were given"))
    p_ = sortperm(collect(tobs))
    ts = collect(tobs)[p_]
    T_obs = Tuple(LinearInterpolation(data_air[p_, k], ts) for k in 1:n_air)
    return T_obs, ts
end

# -----------------------------------------------------------------------------
# Parameter-recovery objective for the air/HVAC nodes, differentiated with
# SciMLSensitivity.adjoint_sensitivities (interpolating/continuous adjoint).
#
#   loss(u0, p) = Σᵢ wᵢ Σₖ (uₖ(tᵢ) - T_obsₖ(tᵢ))² ,
#
# where k runs over `topo.hvac_nodes` (the zone air nodes), the `tᵢ` are the
# (arbitrarily spaced) observation times, the `wᵢ` are trapezoidal quadrature
# weights on those times, and `T_obsₖ` is the observed temperature of node k as
# an interpolator or other callable.  Both the initial condition `u0` and the
# parameter vector `p` are estimated; `p` is in log-space (see `_rhs!`).
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

# Sort observation times and compute trapezoidal quadrature weights for them.
function _observation_weights(tobs)
    ts = sort(collect(tobs))
    return _air_node_quadrature_weights(ts), ts
end

function _check_observation_data(air_idx, T_obs)
    length(T_obs) == length(air_idx) || throw(DimensionMismatch(
        "T_obs must have one interpolator per air node ($(length(air_idx))), got $(length(T_obs))"))
    return nothing
end

function _air_node_residual_loss(sol, air_idx, T_obs, w, tobs)
    u = zero(sol.u[1])
    loss = 0.0
    for (i, ti) in enumerate(tobs)
        sol(u, ti)
        all(isfinite, u) || return Inf
        for (k, idx) in enumerate(air_idx)
            r = u[idx] - T_obs[k](ti)
            loss += w[i] * r * r
        end
    end
    return loss
end

# Canonical method: observations as a tuple of per-node interpolators/callables.
function air_node_residual_loss(topo::BuildingGraphModel, u0, tspan, p,
                                T_obs::Tuple, tobs;
                                alg = Tsit5(), reltol = 1e-8, abstol = 1e-10,
                                log_space = true)
    air_idx = topo.hvac_nodes
    _check_observation_data(air_idx, T_obs)
    w, ts = _observation_weights(tobs)

    # Dense forward solve (no `saveat`): `saveat` disables the dense interpolant,
    # which would force the adjoint to checkpoint/re-solve forward intervals.
    sol = solve(ODEProblem(_odef(topo; log_space = log_space), u0, tspan, p), alg;
                reltol = reltol, abstol = abstol)
    return _air_node_residual_loss(sol, air_idx, T_obs, w, ts)
end

# Convenience method: observations as an (n_t × n_air) matrix aligned with `tobs`.
function air_node_residual_loss(topo::BuildingGraphModel, u0, tspan, p,
                                data_air::AbstractMatrix, tobs;
                                kwargs...)
    T_obs, ts = observation_interpolators(data_air, tobs)
    return air_node_residual_loss(topo, u0, tspan, p, T_obs, ts; kwargs...)
end

"""
    air_node_residual_loss_and_gradients(topo, u0, tspan, p, T_obs, tobs; kwargs...)

Returns `(loss, du0, dp)` where `du0 = ∂loss/∂u0` and `dp = ∂loss/∂p`, both
computed with one forward + adjoint solve.  `T_obs` is a tuple of per-air-node
observation interpolators (or callables) and `tobs` are the (arbitrary)
observation times.  This is the entry point for joint `(u0, p)` estimation.
"""
function air_node_residual_loss_and_gradients(topo::BuildingGraphModel, u0, tspan, p,
                                              T_obs::Tuple, tobs;
                                              alg = Tsit5(),
                                              sensealg = InterpolatingAdjoint(autojacvec = false),
                                              reltol = 1e-8, abstol = 1e-10,
                                              log_space = true)
    air_idx = topo.hvac_nodes
    _check_observation_data(air_idx, T_obs)
    w, ts = _observation_weights(tobs)

    # Dense forward solve (see `air_node_residual_loss`): this avoids adjoint
    # checkpointing and its large allocation overhead.
    sol = solve(ODEProblem(_odef(topo; log_space = log_space), u0, tspan, p), alg;
                reltol = reltol, abstol = abstol)
    loss = _air_node_residual_loss(sol, air_idx, T_obs, w, ts)
    isfinite(loss) || return (Inf, zero(u0), zero(p))

    # discrete cost gradient at each data time: ∂/∂u [wᵢ ‖u_air − T_obs(tᵢ)‖²]
    function dgdu_discrete(out, u, p_, t, i)
        fill!(out, 0.0)
        for (k, idx) in enumerate(air_idx)
            out[idx] = 2 * w[i] * (u[idx] - T_obs[k](t))
        end
        nothing
    end

    du0, dp = adjoint_sensitivities(sol, alg; t = ts, dgdu_discrete = dgdu_discrete,
                                    sensealg = sensealg, reltol = reltol, abstol = abstol)

    grad_u0 = u0 isa ComponentArray ? ComponentArray(vec(du0), getaxes(u0)) : vec(du0)
    grad_p  = p  isa ComponentArray ? ComponentArray(vec(dp),  getaxes(p))  : vec(dp)
    return loss, grad_u0, grad_p
end

function air_node_residual_loss_and_gradients(topo::BuildingGraphModel, u0, tspan, p,
                                              data_air::AbstractMatrix, tobs;
                                              kwargs...)
    T_obs, ts = observation_interpolators(data_air, tobs)
    return air_node_residual_loss_and_gradients(topo, u0, tspan, p, T_obs, ts; kwargs...)
end

function air_node_residual_loss_and_gradient(topo::BuildingGraphModel, u0, tspan, p,
                                             T_obs, tobs;
                                             kwargs...)
    loss, _, grad_p = air_node_residual_loss_and_gradients(
        topo, u0, tspan, p, T_obs, tobs; kwargs...)
    return loss, grad_p
end

air_node_residual_gradient(topo::BuildingGraphModel, u0, tspan, p, T_obs, tobs; kwargs...) =
    air_node_residual_loss_and_gradient(topo, u0, tspan, p, T_obs, tobs; kwargs...)[2]

# -----------------------------------------------------------------------------
# Training interface: learn (u0, p) from measured air-node temperatures and
# store the learned parameters back into the model in absolute units.
# -----------------------------------------------------------------------------

"""
    train!(model::BuildingGraphModel, T_obs, tobs; kwargs...) -> (; u0, loss, retcode)

Learn the model parameters (and initial state `u0`) by minimizing the air-node
trajectory-matching loss against observed temperatures over `[tobs[1], tobs[end]]`.

`T_obs` is a tuple of per-air-node observation interpolators (or an
`n_t × n_air` matrix of observations aligned with `tobs`).  The exogenous
inputs stored in `model` (`T_oa`, `I_solar`, `Q_hvac`) must cover the training
window.

The model's parameters are stored in **absolute** units.  With `log_space=true`
(default) the optimizer works in log-space and maps back with `exp`; with
`log_space=false` the optimizer works directly in absolute space (no per-step
`exp` in the ODE), with positive lower bounds keeping parameters physical.
Either way the learned values are written back into `model.params` in absolute
units.  Returns `(; u0, loss, retcode)`.
"""
function train!(model::BuildingGraphModel, T_obs::Tuple, tobs;
                u0_init = nothing,
                p_init_perturb = 0.2,
                rng_seed = 1,
                alg = Tsit5(),
                sensealg = InterpolatingAdjoint(autojacvec = false),
                reltol = 1e-8, abstol = 1e-10,
                u0_lb = 40.0, u0_ub = 110.0,
                p_lb = nothing, p_ub = nothing,
                maxeval = 400, stopval = 1e-3,
                verbose = false,
                log_space = true)
    air_idx = model.hvac_nodes
    _check_observation_data(air_idx, T_obs)
    ts = sort(collect(tobs))
    tspan = (ts[1], ts[end])

    n_u = model.num_nodes
    n_p = length(model.params)

    if log_space
        all(>(0.0), model.params) || throw(ArgumentError(
            "log-space training requires strictly positive absolute parameters " *
            "(they are mapped through `log`)"))
    end

    # initial-state guess: if none given, start every node at the mean observed
    # air temperature.
    if u0_init === nothing
        μ = 0.0
        n_obs = 0
        for k in 1:length(air_idx)
            for t in ts
                μ += T_obs[k](t)
                n_obs += 1
            end
        end
        u0_init = fill(μ / n_obs, n_u)
    end
    length(u0_init) == n_u || throw(DimensionMismatch(
        "u0_init must have length $(n_u) (one entry per node)"))

    # Parameter initial guess in the chosen space.  The perturbation is a
    # *relative* perturbation in both regimes (multiplicative in absolute space,
    # additive in log-space), so it is scale-invariant.
    p_init = log_space ? log.(model.params) : copy(model.params)
    if p_init_perturb != 0
        rng = Random.MersenneTwister(rng_seed)
        if log_space
            p_init .+= p_init_perturb .* randn(rng, n_p)
        else
            p_init .*= exp.(p_init_perturb .* randn(rng, n_p))
        end
    end

    param_axes = getaxes(model.params)
    pack(s) = ComponentArray(s, param_axes)

    z0 = vcat(collect(u0_init), vec(p_init))

    unpack_z(z) = (z[1:n_u], pack(z[(n_u + 1):end]))

    function objective(z, _)
        try
            u0v, pv = unpack_z(z)
            loss = air_node_residual_loss(model, u0v, tspan, pv, T_obs, ts;
                                          alg = alg, reltol = reltol, abstol = abstol,
                                          log_space = log_space)
            return isfinite(loss) ? loss : 1e12
        catch
            return 1e12
        end
    end

    function grad!(G, z, _)
        try
            u0v, pv = unpack_z(z)
            loss, gu0, gp = air_node_residual_loss_and_gradients(
                model, u0v, tspan, pv, T_obs, ts;
                alg = alg, sensealg = sensealg, reltol = reltol, abstol = abstol,
                log_space = log_space)
            if isfinite(loss) && all(isfinite, vec(gu0)) && all(isfinite, vec(gp))
                G[1:n_u] .= vec(gu0)
                G[(n_u + 1):end] .= vec(gp)
            else
                fill!(G, 0.0)
            end
        catch
            fill!(G, 0.0)
        end
        return nothing
    end

    function fg!(G, z, _)
        local loss, gu0, gp
        try
            u0v, pv = unpack_z(z)
            loss, gu0, gp = air_node_residual_loss_and_gradients(
                model, u0v, tspan, pv, T_obs, ts;
                alg = alg, sensealg = sensealg, reltol = reltol, abstol = abstol,
                log_space = log_space)
        catch
            G !== nothing && fill!(G, 0.0)
            return 1e12
        end

        if isfinite(loss) && all(isfinite, vec(gu0)) && all(isfinite, vec(gp))
            if G !== nothing
                G[1:n_u] .= vec(gu0)
                G[(n_u + 1):end] .= vec(gp)
            end
            return loss
        else
            G !== nothing && fill!(G, 0.0)
            return 1e12
        end
    end

    # Parameter bounds, interpreted in the chosen space.
    _p_lb = p_lb === nothing ? (log_space ? -10.0 : 1e-8) : p_lb
    _p_ub = p_ub === nothing ? (log_space ? 5.0 : 1e4) : p_ub

    optf = OptimizationFunction(objective, Optimization.NoAD(); grad = grad!, fg = fg!)
    lb = vcat(fill(u0_lb, n_u), fill(_p_lb, n_p))
    ub = vcat(fill(u0_ub, n_u), fill(_p_ub, n_p))
    optprob = OptimizationProblem(optf, z0, nothing; lb = lb, ub = ub)

    nlopt_alg = NLopt.Opt(:LD_LBFGS, length(z0))
    NLopt.maxeval!(nlopt_alg, maxeval)
    NLopt.stopval!(nlopt_alg, stopval)

    opt_sol = solve(optprob, nlopt_alg; verbose = verbose)

    u0_rec = opt_sol.u[1:n_u]
    p_rec  = pack(opt_sol.u[(n_u + 1):end])

    # store learned parameters in the model, in absolute units
    model.params .= log_space ? exp.(p_rec) : p_rec

    return (; u0 = u0_rec, loss = objective(opt_sol.u, nothing),
            retcode = opt_sol.retcode)
end

# Convenience method: observations as an (n_t × n_air) matrix aligned with `tobs`.
function train!(model::BuildingGraphModel, data_air::AbstractMatrix, tobs; kwargs...)
    T_obs, ts = observation_interpolators(data_air, tobs)
    return train!(model, T_obs, ts; kwargs...)
end

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

# Hourly solar-gain profiles.  The true coefficients vary over the day (a
# midday peak for the zone mass, a morning peak for the east exterior wall).
# Night hours use a tiny positive floor because parameters are stored/log-mapped
# in log-space (which cannot represent an exact zero).
_day_solar_profile = [1e-4, 1e-4, 1e-4, 1e-4, 1e-4, 1e-4,
                      0.002, 0.005, 0.010, 0.015, 0.020, 0.024,
                      0.025, 0.024, 0.020, 0.015, 0.010, 0.005,
                      0.002, 1e-4, 1e-4, 1e-4, 1e-4, 1e-4]
_morning_solar_profile = [1e-4, 1e-4, 1e-4, 1e-4, 1e-4, 1e-4,
                          0.008, 0.020, 0.030, 0.025, 0.015, 0.008,
                          0.002, 1e-4, 1e-4, 1e-4, 1e-4, 1e-4,
                          1e-4, 1e-4, 1e-4, 1e-4, 1e-4, 1e-4]

z1 = ZoneDef(:z1; cAir = c1Air, cMass = c1Air / 5.0, rAirAm = 0.25, rAirMass = 0.1, α = _day_solar_profile)
z2 = ZoneDef(:z2; cAir = c2Air, cMass = c2Air / 5.0, rAirAm = 0.25, rAirMass = 0.1, α = 0.0)

w1 = WallDef(:w1, :z1, :z2, 2; c = [c1InteriorWall * 2.0, c1InteriorWall * 2.0], r = [0.1, 0.1, 0.1], α_a = 0.0, α_b = 0.0)
w2 = WallDef(:w2, :z1, :ambient, 2; c = [c1Brick * 2.0, c1Brick * 2.0], r = [1.0, 1.0, 1.0], α_a = _morning_solar_profile, α_b = 0.0)
w3 = WallDef(:w3, :z2, :ambient, 2; c = [c2Brick * 2.0, c2Brick * 2.0], r = [1.0, 1.0, 1.0], α_a = 0.0, α_b = 0.0)

# 72 hours of synthetic data
t_sim = 72
n_steps_interp = 600
n_steps_modeled = 100

t_steps = collect(range(0, t_sim, length=n_steps_modeled))

n_zones = 2

# occupancy schedule: each zone occupied 8:00–18:00 every day (24 h clock)
occ_start = [8.0, 8.0]
occ_end   = [18.0, 18.0]

# Weather (uncontrolled disturbance) and HVAC control (known, open-loop) are
# generated separately; internal gains Q_int are learned through the parameters.
T_oa, I_solar = generate_sample_weather((0.0, t_sim), n_steps_interp;
    meanTemp = 72.0, varTemp = 10.0, amp_mod_amp = 15.0, amp_mod_phase = π,
    amp_mod_freq = 1/(3 * t_sim))
Q_hvac = generate_sample_control((0.0, t_sim), n_steps_interp, n_zones;
    amplitudes = [z1PeakCoolingLoad, z2PeakCoolingLoad])

plot(layer(x = t_steps, y = [T_oa(t) for t in t_steps], Geom.line, Theme(default_color = colorant"blue")),
     layer(x = t_steps, y = [I_solar(t) for t in t_steps], Geom.line, Theme(default_color = colorant"orange")),
     Guide.xlabel("Time (hours)"), Guide.ylabel("Exogenous Inputs"), Guide.title("Outdoor Temperature and Solar Irradiance"))
plot(layer(x = t_steps, y = [Q_hvac[1](t) for t in t_steps], Geom.line, Theme(default_color = colorant"blue")),
     layer(x = t_steps, y = [Q_hvac[2](t) for t in t_steps], Geom.line, Theme(default_color = colorant"orange")),
     Guide.xlabel("Time (hours)"), Guide.ylabel("Q_hvac (tons)"), Guide.title("Zone HVAC Control Inputs"))

topo = BuildingGraphModel([z1, z2], [w1, w2, w3], T_oa, I_solar, Q_hvac, occ_start, occ_end)



# initial conditions and true parameters (absolute units in `topo.params`,
# mapped to log-space for the truth solve)
u0_true = fill(72.0, topo.num_nodes)
topo.params.Q_int_occ   .= [5.0, 5.0]   # true occupied internal gain per zone
topo.params.Q_int_unocc .= [0.5, 0.5]   # true unoccupied internal gain per zone
p_abs_true = copy(topo.params)          # absolute truth, for reporting
p_true_log = log.(topo.params)          # log-space, for the ODE

prob = ODEProblem(_odef(topo), u0_true, (0.0, t_sim), p_true_log)

# Dense truth solve (no `saveat`) so it can be sampled at arbitrary times below.
@time sol_true = solve(prob, Tsit5(), reltol = 1e-6, abstol = 1e-6)

# plot the air temps (sol_true.u[1] and sol_true.u[3]) 

plot(layer(x = t_steps, y = [sol_true(t)[1] for t in t_steps], Geom.line, Theme(default_color = colorant"blue")),
     layer(x = t_steps, y = [sol_true(t)[3] for t in t_steps], Geom.line, Theme(default_color = colorant"orange")),
     Guide.xlabel("Time (hours)"), Guide.ylabel("Zone Air Temperatures (°F)"), Guide.title("Zone Air Temperatures Over Time"))






#=
Model parameter recovery
=#


# observed air-node trajectory from the truth solve ----------------------------
# Sample the truth solution at *irregular* times (as real sensors would produce)
# and wrap the observations in one interpolator per air node.
air_idx = topo.hvac_nodes
n_air   = length(air_idx)

t_obs = sort(t_steps .+ 0.19 .* sin.(3.7 .* t_steps))   # deterministic non-uniform offsets
t_obs[1] = t_steps[1]
t_obs[end] = t_steps[end]
t_obs = unique(t_obs)

data_air = permutedims(reduce(hcat, [sol_true(t)[air_idx] for t in t_obs]))   # n_t × n_air
T_obs, t_obs = observation_interpolators(data_air, t_obs)

# short validation window (first ~8 h of observation time)
tchk = t_obs[t_obs .<= 8.0]

# generate intial guess for parameters in log-space
Random.seed!(1)
p_init_log = p_true_log .+ 0.2 .* randn(length(p_true_log))


# NB: the same interpolators are reused for the short window — no matrix slicing
# needed, since observations are queried as T_obs[k](t).
loss_short(u0v, pv) = air_node_residual_loss(topo, u0v, (0.0, tchk[end]), pv, T_obs, tchk;
                                       reltol = 1e-10, abstol = 1e-12)

@time _, g_u0_adj, g_p_adj = air_node_residual_loss_and_gradients(
    topo, u0_true, (0.0, tchk[end]), p_init_log, T_obs, tchk;
    reltol = 1e-8, abstol = 1e-8)
@time g_p_fwd  = ForwardDiff.gradient(pv -> loss_short(u0_true, pv), p_init_log)
@time g_u0_fwd = ForwardDiff.gradient(u0v -> loss_short(u0v, p_init_log), u0_true)

println("=== adjoint_sensitivities vs ForwardDiff (short window) ===")
println("max |g_p_adj  - g_p_fwd|  = ", round(maximum(abs.(vec(g_p_adj) .- g_p_fwd)), sigdigits=4))
println("relative L2 error (p)     = ", round(norm(vec(g_p_adj) - g_p_fwd) / norm(g_p_fwd), sigdigits=4))
println("max |g_u0_adj - g_u0_fwd| = ", round(maximum(abs.(vec(g_u0_adj) .- g_u0_fwd)), sigdigits=4))
println("relative L2 error (u0)    = ", round(norm(vec(g_u0_adj) - g_u0_fwd) / norm(g_u0_fwd), sigdigits=4))


# -----------------------------------------------------------------------------
# Learn (u0, p) with train!.  Perturb the true parameters to form the initial
# guess (what "a model architecture + approximate parameters" looks like), then
# let train! store the learned absolute parameters back into `topo`.
# -----------------------------------------------------------------------------
Random.seed!(1)
topo_learned = deepcopy(topo)
topo_learned.params .= p_abs_true .* exp.(0.2 .* randn(length(p_abs_true)))   # initial guess

@time result = train!(topo_learned, T_obs, t_obs;
                      u0_init = fill(75.0, topo_learned.num_nodes),
                      p_init_perturb = 0.0,
                      maxeval = 500)

p_abs_rec = topo_learned.params   # learned, stored in absolute units

println("\n=== train! (u0, p) recovery ===")
println("retcode                 = ", result.retcode)
println("final loss              = ", round(result.loss, sigdigits = 5))
println("||u0 - u0_true||        = ", round(norm(result.u0 .- u0_true), sigdigits = 5))
println("||p - p_true|| (log)    = ", round(norm(log.(p_abs_rec) - log.(p_abs_true)), sigdigits = 5))
println("Q_int_occ recovered     = ", round.(p_abs_rec.Q_int_occ, sigdigits = 4),
        "  (true ", round.(p_abs_true.Q_int_occ, sigdigits = 4), ")")
println("Q_int_unocc recovered   = ", round.(p_abs_rec.Q_int_unocc, sigdigits = 4),
        "  (true ", round.(p_abs_true.Q_int_unocc, sigdigits = 4), ")")

# Report the identifiable (daytime) solar-gain profile for the first solar node.
# Hours without solar irradiance carry no information, so those coefficients stay
# near their initial guess.
println("α_solar[1,:] recovered  = ", round.(p_abs_rec.α_solar[1, :], sigdigits = 3))
println("α_solar[1,:] true       = ", round.(p_abs_true.α_solar[1, :], sigdigits = 3))