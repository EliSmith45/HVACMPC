module Controls

using OrdinaryDiffEqSDIRK

using ..ModelArchitecture: BuildingGraphModel, simulate_forward
using ..EnergyModel: HVACSystem, clamp_load

# ---------------------------------------------------------------------------
# Controllers
# ---------------------------------------------------------------------------

abstract type AbstractController end

"""
    PIDController(n_zones; kp=1.0, ki=0.0, kd=0.0, setpoint=72.0,
                  q_min=-Inf, q_max=Inf, dt=1.0)

A per-zone discrete-time PID controller.  Gains, setpoint and output limits are
scalars (applied to every zone) or per-zone vectors.

Conventions match the building model: the measured signal is the zone air
temperature (°F) and the control output `Q_hvac` is a thermal load (tons);
`Q > 0` heats and `Q < 0` cools.  The output is clamped to `[q_min, q_max]` and
the integrator uses conditional-integration anti-windup (it only integrates
while the output is unsaturated).
"""
mutable struct PIDController <: AbstractController
    kp::Vector{Float64}
    ki::Vector{Float64}
    kd::Vector{Float64}
    setpoint::Vector{Float64}
    q_min::Vector{Float64}
    q_max::Vector{Float64}
    dt::Float64
    integral::Vector{Float64}
    prev_error::Vector{Float64}
    initialized::Bool
end

_as_vec(x::Real, n, name) = fill(Float64(x), n)
function _as_vec(x, n, name)
    v = collect(x)
    length(v) == n || throw(DimensionMismatch(
        "$name must have one entry per zone ($n), got $(length(v))"))
    return Float64.(v)
end

function PIDController(n_zones::Integer; kp = 1.0, ki = 0.0, kd = 0.0,
                       setpoint = 72.0, q_min = -Inf, q_max = Inf, dt = 1.0)
    n_zones > 0 || throw(ArgumentError("n_zones must be positive"))
    dt > 0 || throw(ArgumentError("dt must be positive"))
    n = Int(n_zones)
    return PIDController(
        _as_vec(kp, n, "kp"),
        _as_vec(ki, n, "ki"),
        _as_vec(kd, n, "kd"),
        _as_vec(setpoint, n, "setpoint"),
        _as_vec(q_min, n, "q_min"),
        _as_vec(q_max, n, "q_max"),
        Float64(dt),
        zeros(n),
        zeros(n),
        false,
    )
end

"""
    reset!(ctrl::PIDController) -> ctrl

Reset the controller's integrator and previous-error memory.
"""
function reset!(ctrl::PIDController)
    fill!(ctrl.integral, 0.0)
    fill!(ctrl.prev_error, 0.0)
    ctrl.initialized = false
    return ctrl
end

"""
    compute_control(ctrl::PIDController, t, y) -> q

Compute the per-zone control action from the measured zone temperatures `y` at
time `t`, updating the controller's internal state.  Returns the saturated
(clamped) output; the integrator is frozen on any zone whose output is
saturated.
"""
function compute_control(ctrl::PIDController, t, y::AbstractVector)
    n = length(ctrl.kp)
    length(y) == n || throw(DimensionMismatch(
        "measured vector has $(length(y)) entries but controller expects $n"))

    e = ctrl.setpoint .- y
    if !ctrl.initialized
        ctrl.prev_error .= e
        ctrl.initialized = true
    end

    u = ctrl.kp .* e .+ ctrl.ki .* ctrl.integral .+
        ctrl.kd .* (e .- ctrl.prev_error) ./ ctrl.dt
    u_lim = clamp.(u, ctrl.q_min, ctrl.q_max)

    @inbounds for i in 1:n
        # Conditional-integration anti-windup: only integrate while unsaturated.
        if ctrl.q_min[i] < u[i] < ctrl.q_max[i]
            ctrl.integral[i] += e[i] * ctrl.dt
        end
        ctrl.prev_error[i] = e[i]
    end

    return u_lim
end

# ---------------------------------------------------------------------------
# Closed-loop simulation
# ---------------------------------------------------------------------------

"""
    ClosedLoopResult(times, states, q_hvac, air_temps)

Result of a closed-loop simulation.  `states` is `num_nodes × n_t`; `q_hvac`
and `air_temps` are `n_zones × n_t`, all sampled at `times` (hours).
"""
struct ClosedLoopResult
    times::Vector{Float64}
    states::Matrix{Float64}
    q_hvac::Matrix{Float64}
    air_temps::Matrix{Float64}
end

"""
    simulate_closed_loop(model, u0, tspan, controller, hvac; kwargs...) -> ClosedLoopResult

Simulate the building under feedback control with a zero-order-hold: every
`dt` hours the controller computes the per-zone load from the current zone air
temperatures, the load is clamped to the HVAC capacity, and the building ODE is
advanced one step with that load held constant.

`hvac` is a single [`HVACSystem`](@ref) applied to every zone, or a
vector/tuple with one unit per zone.  `T_oa`/`I_solar` default to the
interpolators stored in `model`.  Extra keyword arguments (`reltol`, `abstol`)
are forwarded to `simulate_forward`; `alg` defaults to `KenCarp4()`.
"""
function simulate_closed_loop(model::BuildingGraphModel, u0, tspan,
                              controller::AbstractController, hvac;
                              dt = 1.0,
                              alg = KenCarp4(),
                              reltol = 1e-6, abstol = 1e-6,
                              T_oa = model.T_oa, I_solar = model.I_solar,
                              progress = false)
    air_idx = model.hvac_nodes
    n_zones = length(air_idx)

    hvac_list = hvac isa HVACSystem ? fill(hvac, n_zones) : collect(hvac)
    length(hvac_list) == n_zones || throw(DimensionMismatch(
        "one HVAC system per zone is required ($n_zones), got $(length(hvac_list))"))

    t0, t_end = Float64.(tspan)
    dt = Float64(dt)
    u = Float64.(u0)
    length(u) == model.num_nodes || throw(DimensionMismatch(
        "u0 has $(length(u)) entries but the model has $(model.num_nodes) nodes"))

    reset!(controller)

    times = Float64[t0]
    state_list = Vector{Float64}[copy(u)]
    q_list = Vector{Float64}[zeros(n_zones)]

    t = t0
    while t < t_end
        t_next = min(t + dt, t_end)

        q = compute_control(controller, t, u[air_idx])
        for i in 1:n_zones
            q[i] = clamp_load(hvac_list[i], q[i])
        end

        # Hold the load constant over [t, t_next].
        q_const = Vector{Function}(undef, n_zones)
        for i in 1:n_zones
            qi = q[i]
            q_const[i] = tt -> qi
        end

        sol = simulate_forward(model, u, (t, t_next);
                               params_abs = model.params,
                               T_oa = T_oa, I_solar = I_solar,
                               Q_hvac = Tuple(q_const),
                               alg = alg, reltol = reltol, abstol = abstol)

        u = Float64.(sol.u[end])
        t = t_next

        push!(times, t)
        push!(state_list, copy(u))
        push!(q_list, copy(q))

        if progress && (length(times) - 1) % 240 == 0
            println("t = $(round(t, digits = 1)) h, zone air = ",
                    round.(u[air_idx], digits = 2),
                    ", Q = ", round.(q, digits = 3), " tons")
        end
    end

    states = hcat(state_list...)    # num_nodes × n_t
    q_hvac = hcat(q_list...)        # n_zones × n_t
    air_temps = states[air_idx, :]  # n_zones × n_t
    return ClosedLoopResult(times, states, q_hvac, air_temps)
end

export AbstractController, PIDController, reset!, compute_control,
       ClosedLoopResult, simulate_closed_loop

end
