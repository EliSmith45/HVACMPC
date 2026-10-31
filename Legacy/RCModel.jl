using DifferentialEquations
using DiffEqCallbacks
using NLsolve
using DataFrames
using Gadfly, Cairo
using DataInterpolations
using Random
using ComponentArrays
using LinearAlgebra
using SciMLSensitivity



#= 
Plugin to get weather data
=#
include("openmeteo_weather.jl")

addr = "4716 Liberty Ave, Pittsburgh, PA 15224"
w = get_weather(addr, Date(2023, 6, 1), Date(2023, 6, 7), 7)

wl = w.location
hist = w.historical
fcst = w.forecast


hist.time[1]
aa = Date("2024-01-01")

aa=DateTime("2024-01-01T00:00:00")
Dates.value(aa)
Dates.value(hist.time[1])

#=
# Model definition
=#

struct InterpInputs{F1, F2, F3, F4}
    T_oa::F1
    I_sol::F2
    Q_int::F3
    Q_hvac::F4
end

struct ZoneDef
    id::Symbol
    has_solar::Bool
    params::Vector{Float64} # [C_air, C_mass, R_am] (+ A_solar when has_solar)
end

struct WallDef
    id::Symbol
    zone_a::Symbol # zone symbol or :ambient
    zone_b::Symbol # zone symbol or :ambient
    C_wall::Vector{Float64} # capacitance of each node; num_nodes = length(C_wall)
    R_wall::Vector{Float64} # resistors; length(R_wall) == length(C_wall) + 1
end

# defines the structure of the building model as a set of zones and walls linking zones
struct BuildingTopology
    zones::Vector{ZoneDef}
    walls::Vector{WallDef}
    zone_indices::Dict{Symbol, Int}
    state_names::Vector{Symbol}
    param_names::Vector{Symbol}
end

function BuildingTopology(zones::Vector{ZoneDef}, walls::Vector{WallDef})
    z_map = Dict(z.id => i for (i, z) in enumerate(zones))
    state_names = Symbol[]
    param_names = Symbol[]

    # zone air and internal mass states and parameters
    for z in zones
        @assert length(z.params) == (z.has_solar ? 4 : 3) "zone $(z.id): params must be [C_air, C_mass, R_am$(z.has_solar ? ", A_solar" : "")]"

        push!(state_names, Symbol("T_air_", z.id))
        push!(state_names, Symbol("T_mass_", z.id))
        push!(param_names, Symbol("C_air_", z.id))
        push!(param_names, Symbol("C_mass_", z.id))
        push!(param_names, Symbol("R_am_", z.id))

        if z.has_solar
            push!(param_names, Symbol("A_solar_", z.id))
        end
    end

    # wall parameters and states
    for w in walls
        @assert length(w.R_wall) == length(w.C_wall) + 1 "wall $(w.id): need length(R_wall) == length(C_wall) + 1"

        for n in 1:length(w.C_wall)
            push!(state_names, Symbol("T_wall_", w.id, "_n", n))
            push!(param_names, Symbol("C_wall_", w.id, "_n", n))
        end

        # need n + 1 resistors for a wall with n capacitors
        for n in 1:length(w.R_wall)
            push!(param_names, Symbol("R_wall_", w.id, "_n", n))
        end
    end

    return BuildingTopology(zones, walls, z_map, state_names, param_names)
end


#=
Assemble the state space model matrices.

State ordering (matches `topo.state_names`):
    zone i air   -> index 2i - 1
    zone i mass  -> index 2i
    wall nodes   -> indices 2*N_zones + 1 : end, in `topo.walls` order, node 1..N_n

Parameter ordering (matches `topo.param_names`):
    per zone: C_air, C_mass, R_am, (A_solar if has_solar)
    per wall: C_wall_1..C_wall_n, R_wall_1..R_wall_{n+1}

The dynamics are C * dx/dt = G * x + B_d_raw * d(t) + B_u_raw * u(t) with
    x    = state temperatures
    d(t) = [T_oa(t), I_sol(t), Q_int_1(t), ..., Q_int_nz(t)]
    u(t) = [Q_hvac_1(t), ..., Q_hvac_nz(t)]
so dividing through by the capacitances gives dx/dt = A*x + B_d*d + B_u*u.

Sign convention: a node with capacitance C coupled to a neighbour by conductance g
satisfies C*dT/dt = sum_j g*(T_j - T), which contributes -g to its diagonal entry
of G and +g to the (node, j) off-diagonal. Ambient temperature and heat-gain
columns enter B_d_raw/B_u_raw with a positive sign because they add heat.
=#

# Build the parameter vector directly from the zone/wall objects, in the same
# order as `topo.param_names`.
function initial_params(topo::BuildingTopology)
    p = Float64[]
    for z in topo.zones
        append!(p, z.params)
    end
    for w in topo.walls
        append!(p, w.C_wall)
        append!(p, w.R_wall)
    end
    return p
end

assemble_matrices(topo::BuildingTopology) = assemble_matrices(topo, initial_params(topo))

function assemble_matrices(topo::BuildingTopology, p::AbstractVector{T}) where {T}
    N_states = length(topo.state_names)
    N_zones = length(topo.zones)

    # Conductance matrix G and inverse capacitance vector
    # C_vec .* dx/dt = G * x + B_d_raw * d(t) + B_u_raw * u(t)
    G = zeros(T, N_states, N_states)
    C_inv = zeros(T, N_states)

    # Inputs: d(t) = [T_oa, I_sol, Q_int_1..Q_int_nz], u(t) = [Q_hvac_1..Q_hvac_nz]
    N_dist = 2 + N_zones
    B_d_raw = zeros(T, N_states, N_dist)
    B_u_raw = zeros(T, N_states, N_zones)

    pi = 1 # cursor into p, advanced in lockstep with the param_names ordering

    for (i, z) in enumerate(topo.zones)
        idx_a = 2i - 1
        idx_m = 2i

        c_a = p[pi]; pi += 1
        c_m = p[pi]; pi += 1
        r_am = p[pi]; pi += 1

        C_inv[idx_a] = 1.0 / c_a
        C_inv[idx_m] = 1.0 / c_m

        # air <-> internal mass conductance
        g_am = 1.0 / r_am
        G[idx_a, idx_a] -= g_am
        G[idx_m, idx_m] -= g_am
        G[idx_a, idx_m] += g_am
        G[idx_m, idx_a] += g_am

        # Solar heat gain to the air node (column 2 is I_sol)
        if z.has_solar
            A_sol = p[pi]; pi += 1
            B_d_raw[idx_a, 2] += A_sol
        end

        # Internal gains (column 2 + i is Q_int_i) and HVAC (column i is Q_hvac_i)
        B_d_raw[idx_a, 2 + i] += 1.0
        B_u_raw[idx_a, i] += 1.0
    end

    # Wall RC meshes
    wall_base = 2 * N_zones
    wall_cursor = 0
    for w in topo.walls
        N_n = length(w.C_wall)
        wall_node_idxs = collect((wall_base + wall_cursor + 1):(wall_base + wall_cursor + N_n))
        wall_cursor += N_n

        # invert capacitances
        for idx in wall_node_idxs
            c_w = p[pi]; pi += 1
            C_inv[idx] = 1.0 / c_w
        end

        # resistances (n + 1 for n capacitors) -> conductances
        resistors = p[pi:(pi + N_n)]
        pi += N_n + 1
        conductances = 1.0 ./ resistors

        # boundary connection to zone_a (or ambient)
        g_first = conductances[1]
        idx_first_node = wall_node_idxs[1]
        G[idx_first_node, idx_first_node] -= g_first

        if w.zone_a == :ambient
            B_d_raw[idx_first_node, 1] += g_first  # column 1 is T_oa
        else
            idx_za = 2 * topo.zone_indices[w.zone_a] - 1
            G[idx_za, idx_first_node] += g_first
            G[idx_first_node, idx_za] += g_first
            G[idx_za, idx_za] -= g_first
        end

        # inner wall inter-node conductances
        for n in 1:(N_n - 1)
            g_inner = conductances[n + 1]
            idx_inner_node = wall_node_idxs[n]
            idx_next_node = wall_node_idxs[n + 1]
            G[idx_inner_node, idx_inner_node] -= g_inner
            G[idx_next_node, idx_next_node] -= g_inner
            G[idx_inner_node, idx_next_node] += g_inner
            G[idx_next_node, idx_inner_node] += g_inner
        end

        # boundary connection to zone_b (or ambient)
        g_last = conductances[end]
        idx_last_node = wall_node_idxs[end]
        G[idx_last_node, idx_last_node] -= g_last

        if w.zone_b == :ambient
            B_d_raw[idx_last_node, 1] += g_last  # column 1 is T_oa
        else
            idx_zb = 2 * topo.zone_indices[w.zone_b] - 1
            G[idx_zb, idx_last_node] += g_last
            G[idx_last_node, idx_zb] += g_last
            G[idx_zb, idx_zb] -= g_last
        end
    end

    A_mat = C_inv .* G
    B_d = C_inv .* B_d_raw
    B_u = C_inv .* B_u_raw

    return A_mat, B_d, B_u
end


#=
# BuildingODEModel: a neural-ODE-style wrapper around the RC building model
=#

"""
    BuildingODEModel

A mutable, neural-ODE-like wrapper around the building RC model.

The learnable parameters are the assembled state-space matrices `A`, `B_d`, `B_u`,
stored as a `ComponentArray` with named matrix views over a single flat vector
(so Zygote/SciMLSensitivity can differentiate through `solve` w.r.t. them).

The exogenous inputs (`T_oa`, `I_sol`, `Q_int`, `Q_hvac` interpolation functions)
are fixed during learning; they are captured by the ODE right-hand side closure
and are *not* part of the differentiable parameter vector.

The ODE right-hand side never assembles matrices itself -- it only consumes the
matrices stored in `p` and evaluates the interpolators at the current time `t`:

    du/dt = A * u + B_d * d(t) + B_u * u(t)

where `d(t) = [T_oa(t), I_sol(t), Q_int_1(t), ..., Q_int_nz(t)]` and
`u(t) = [Q_hvac_1(t), ..., Q_hvac_nz(t)]`.
"""
mutable struct BuildingODEModel
    topo::BuildingTopology
    params::ComponentArray   # named matrix views: .A, .B_d, .B_u over one flat vector
    inputs::InterpInputs     # exogenous interpolators (fixed, not learnable)
    tspan::Tuple             # (t0, tf) simulation window
    u0::Vector{Float64}      # initial state
    prob::ODEProblem         # ODE problem built from the above
end

# Disturbance input vector: d(t) = [T_oa(t), I_sol(t), Q_int_1(t), ..., Q_int_nz(t)]
function disturbance_vector(inputs::InterpInputs, t)
    return vcat(inputs.T_oa(t), inputs.I_sol(t), [q(t) for q in inputs.Q_int]...)
end

# Control input vector: u(t) = [Q_hvac_1(t), ..., Q_hvac_nz(t)]
function control_vector(inputs::InterpInputs, t)
    return [q(t) for q in inputs.Q_hvac]
end

# Out-of-place RHS for the ODE.
# `p` is the ComponentArray of matrices (the learnable parameters);
# `inputs` is captured by the closure built in the constructor, so it is
# treated as a constant by Zygote/SciMLSensitivity adjoint differentiation.
function building_rhs(u, p, t, inputs::InterpInputs)
    return p.A * u + p.B_d * disturbance_vector(inputs, t) + p.B_u * control_vector(inputs, t)
end

"""
    BuildingODEModel(topo, A, B_d, B_u, inputs, tspan, u0)

Construct a `BuildingODEModel` from pre-assembled state-space matrices.

The matrices are wrapped into a `ComponentArray` so they form a single flat,
Zygote-differentiable parameter vector with named views `.A`, `.B_d`, `.B_u`.
The ODE problem is built from an out-of-place right-hand side that closes over
`inputs` (the exogenous interpolators) and reads the matrices from `p`.
"""
function BuildingODEModel(topo::BuildingTopology, A::AbstractMatrix, B_d::AbstractMatrix, B_u::AbstractMatrix, inputs::InterpInputs, tspan::Tuple, u0::AbstractVector)
    params = ComponentArray(A = A, B_d = B_d, B_u = B_u)
    u0v = Float64.(u0)
    f = (u, p, t) -> building_rhs(u, p, t, inputs)
    prob = ODEProblem(f, u0v, tspan, params)
    return BuildingODEModel(topo, params, inputs, tspan, u0v, prob)
end

# NOTE: a `train` method using Zygote reverse-mode AD for adjoint sensitivity
# will be added later. It will differentiate
#     solve(model.prob, p = model.params, ...)
# w.r.t. the matrix parameters (A, B_d, B_u). Not implemented yet.

zones = [
    # [C_air, C_mass, R_am, A_solar]
    ZoneDef(:perimeter, true,  [1000.0, 500.0, 0.0005, 10.0]),
    # [C_air, C_mass, R_am]
    ZoneDef(:core, false,     [1000.0, 500.0, 0.0005])
]

walls = [
    # [C_wall node 1..n], [R_wall resistor 1..n+1]
    WallDef(:ext_perim, :ambient, :perimeter, [5.0e6, 5.0e6], [1.0e-6, 1.0e-6, 0.0001]),
    WallDef(:partition, :perimeter, :core, [5.0e6, 5.0e6, 3.0e6], [1.0e-6, 1.0e-6, 0.0001, 1.0e-6]),
    WallDef(:roof_core, :ambient, :core, [5.0e6, 5.0e6], [1.0e-6, 1.0e-6, 1.0e-6])
]

topo = BuildingTopology(zones, walls)


# get weather data

addr = "4716 Liberty Ave, Pittsburgh, PA 15224"
w = get_weather(addr, Date(2023, 6, 1), Date(2023, 6, 7), 7)

wl = w.location
hist = w.historical
fcst = w.forecast

tref = hist.time[1]
t_sim = 7*86400
tsteps_interp = Dates.value.(hist.time .- tref) ./ 3600000


Q_internal_1 = [8.0 < mod(t, 24) < 16.0 ? 200000 : 500000 for t in Int.(tsteps_interp)]
Q_internal_2 = [8.0 < mod(t, 24) < 16.0 ? 100000 : 250000 for t in Int.(tsteps_interp)]

Q_hvac_1 = [-2500 .* max.(0, sin.(2*pi*t/24)) for t in Int.(tsteps_interp)]
Q_hvac_2 = [-2500 .* max.(0, sin.(2*pi*t/24)) for t in Int.(tsteps_interp)]

interp_inputs = InterpInputs(
    LinearInterpolation(hist.temperature_2m, tsteps_interp), 
    LinearInterpolation(hist.shortwave_radiation, tsteps_interp),
    [LinearInterpolation(Q_internal_1, tsteps_interp), LinearInterpolation(Q_internal_2, tsteps_interp)],
    [LinearInterpolation(Q_hvac_1, tsteps_interp), LinearInterpolation(Q_hvac_2, tsteps_interp)]
)

a, bd, bu = assemble_matrices(topo)
u0 = fill(72, size(a, 1))

rcModel = BuildingODEModel(topo, a, bd, bu, interp_inputs, (0, 165), u0)

sol = solve(rcModel.prob, p = rcModel.params, tspan = rcModel.tspan)