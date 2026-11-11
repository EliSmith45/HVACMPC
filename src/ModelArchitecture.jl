module ModelArchitecture

using LinearAlgebra, ComponentArrays, DataInterpolations
using DifferentialEquations, OrdinaryDiffEqSDIRK
using Random

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

"""
    simulate_forward(m::BuildingGraphModel, u0, tspan; kwargs...) -> ODESolution

Simulate the model forward in **absolute** units and return the `ODESolution`.
`u0` is the initial state and `tspan` the time span; `params_abs`, `T_oa`,
`I_solar` and `Q_hvac` default to the values stored in `m` (pass fresh
forecasts instead to reuse the same topology in an MPC loop).

The ODE is solved with the stiff `KenCarp4` method by default and an analytic
state Jacobian (see [`make_forward_odefun`](@ref)).  Any extra keyword arguments
(`reltol`, `abstol`, `saveat`, `dtmax`, …) are forwarded to `solve`.
"""
function simulate_forward(m::BuildingGraphModel, u0, tspan;
                          params_abs = m.params,
                          T_oa = m.T_oa,
                          I_solar = m.I_solar,
                          Q_hvac = m.Q_hvac,
                          alg = KenCarp4(),
                          kwargs...)
    prob = make_forward_problem(m, u0, tspan, params_abs, T_oa, I_solar, Q_hvac)
    return solve(prob, alg; kwargs...)
end


# -----------------------------------------------------------------------------
# HVAC control input generation.  Q_hvac is a *known* control input chosen by
# the user; internal gains Q_int are learned parameters living in
# `BuildingGraphModel.params`.
# -----------------------------------------------------------------------------

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
        model.num_nodes,
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


export ZoneDef, WallDef, BuildingGraphModel, Exogenous,
       make_forward_odefun, make_forward_problem, simulate_forward,
       generate_sample_control, update_exo_inputs

end
