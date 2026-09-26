# =============================================================================
# inverse_problem.jl
# -----------------------------------------------------------------------------
# Recover the 6 physical parameters of the 2-state RC house model
#     p = (c_a, c_m, r_oa, r_am, α_solar, q_internal)
# from the observed trajectories T_air(t), T_mass(t) produced by the forward
# model in exploratory.jl, using CONTINUOUS ADJOINT SENSITIVITY ANALYSIS.
#
# Pipeline
#   1. Run the full hybrid (on/off controller) model to generate "data" and
#      to log the HVAC on/off state (this is information a real controller
#      would record).
#   2. Turn the logged schedule into a known exogenous heat flux q_hvac(t).
#      This decouples the controller from the estimation, so the estimation
#      model is a SMOOTH ODE (the standard adjoint method applies).
#   3. Define the trajectory-matching loss, derive the continuous adjoint by
#      hand, and verify the adjoint gradient against finite differences.
#   4. Minimize the loss with Adam (in log-space, so all 6 very different
#      scales are treated equally) and compare to the truth.
#
# Identifiability: all 6 parameters are identifiable because q_hvac is a KNOWN
# heat flux (BTU/hr) — it provides the absolute scale. The effective parameters
#     θ = (c_a r_oa, c_a r_am, c_a α_solar, c_a q_internal, c_a, c_m r_am)
# enter the ODE linearly, and c_a itself is the coefficient of q_hvac.
# =============================================================================

using DifferentialEquations
using DiffEqCallbacks
using DataInterpolations
using DataFrames
using LinearAlgebra
using Printf
using Statistics

include("openmeteo_weather.jl")

# ---- weather (identical to exploratory.jl) ---------------------------------
addr = "4716 Liberty Ave, Pittsburgh, PA 15224"
w = get_weather(addr, Date(2023, 6, 1), Date(2023, 6, 7), 7)
hist = w.historical
t_ref = hist.time[1]
hist.rel_time = Dates.value.(hist.time .- t_ref) ./ 3600000
temp_interp = CubicSpline(hist.temperature_2m, hist.rel_time)
sw_interp   = CubicSpline(hist.shortwave_radiation, hist.rel_time)

# ---- true physical parameters (the ones we will recover) --------------------
c_a, c_m       = 0.0005, 0.0003
r_oa           = 16000.0 / (90.0 - 70.0)
r_am           = r_oa / 2
α_solar        = 9.9
q_internal     = 6000.0
p_true         = [c_a, c_m, r_oa, r_am, α_solar, q_internal]

# ---- controller parameters (assumed KNOWN) ----------------------------------
setpoint, deadband   = 70.0, 2.0
hvac_capacity        = 36000.0
supply_temp          = 55.0
design_cfm           = 400.0 * hvac_capacity / 12000.0
min_run_time         = 0.25

Tspan = (0.0, 24.0 * 3.0)          # 3 days is plenty of excitation
RT, AT = 1e-8, 1e-10               # tight tolerances: the estimation is sensitive
                                    # to solver error (default reltol=1e-3 ≈ 0.07 °F)

# =============================================================================
# Forward (data-generating) model: the full hybrid system from exploratory.jl
# =============================================================================
mutable struct HVACState
    setpoint::Float64; deadband::Float64; capacity::Float64
    supply_temp::Float64; cfm::Float64; min_run_time::Float64
    on::Bool; t_on::Float64
end

function rc_model!(du, u, p, t)
    T_air, T_mass, run_timer = u
    hvac = p.hvac
    oat_interp, sw_interp_ = p.weather
    c_a_, c_m_, r_oa_, r_am_, α_, qint_ = p.coeffs
    t_out  = oat_interp(t)
    sw_out = sw_interp_(t)
    q_hvac = hvac.on ? -clamp(1.08 * hvac.cfm * (T_air - hvac.supply_temp), 0.0, hvac.capacity) : 0.0
    du[1]  = c_a_ * ((t_out - T_air) * r_oa_ + (T_mass - T_air) * r_am_ + α_ * sw_out + qint_ + q_hvac)
    du[2]  = c_m_ * r_am_ * (T_air - T_mass)
    du[3]  = hvac.on ? 1.0 : 0.0
end

on_condition(u, t, integ) = u[1] - (integ.p.hvac.setpoint + integ.p.hvac.deadband)
function on_affect!(integ)
    integ.p.hvac.on = true
    integ.p.hvac.t_on = integ.t
    integ.u[3] = 0.0
end
function off_condition(u, t, integ)
    hvac = integ.p.hvac
    max(u[1] - (hvac.setpoint - hvac.deadband), hvac.min_run_time - u[3])
end
off_affect!(integ) = (integ.p.hvac.on = false)

sched = SavedValues(Float64, Bool)          # logs the on/off state (a real controller would too)
cb = CallbackSet(ContinuousCallback(on_condition, on_affect!, nothing),
                 ContinuousCallback(off_condition, nothing, off_affect!),
                 SavingCallback((u, t, integ) -> integ.p.hvac.on, sched))

hvac0  = HVACState(setpoint, deadband, hvac_capacity, supply_temp, design_cfm, min_run_time, false, -Inf)
params = (weather=(temp_interp, sw_interp), coeffs=Tuple(p_true), hvac=hvac0)
sol    = solve(ODEProblem(rc_model!, [setpoint, setpoint, 0.0], Tspan, params),
               Tsit5(), callback=cb, reltol=RT, abstol=AT)

# ---- "observed" data on a uniform grid --------------------------------------
dt      = 0.02
tgrid   = collect(0.0:dt:Tspan[2])
Ta_data = [sol(t)[1] for t in tgrid]
Tm_data = [sol(t)[2] for t in tgrid]

# ---- known schedule -> known heat flux q_hvac(t) ----------------------------
function on_at(t)
    idx = searchsortedlast(sched.t, t)
    idx == 0 ? false : sched.saveval[idx]
end
qh_true = [on_at(t) ? -clamp(1.08 * design_cfm * (sol(t)[1] - supply_temp), 0.0, hvac_capacity) : 0.0
           for t in tgrid]

QHVAC = LinearInterpolation(qh_true, tgrid)   # known forcing (time only)
TA_D  = LinearInterpolation(Ta_data, tgrid)   # data interpolants for the adjoint
TM_D  = LinearInterpolation(Tm_data, tgrid)

# =============================================================================
# Estimation model: SMOOTH ODE (the schedule is fixed), params in physical units
#     dT_a/dt = c_a [ r_oa(T_out-T_a) + r_am(T_m-T_a) + α_solar·sw + q_int + q_hvac(t) ]
#     dT_m/dt = c_m r_am (T_a - T_m)
# =============================================================================
function fest!(du, u, p, t)
    T_a, T_m = u
    c_a_, c_m_, r_oa_, r_am_, α_s, qint = p
    t_out = temp_interp(t); sw = sw_interp(t); qh = QHVAC(t)
    du[1] = c_a_ * ((t_out - T_a) * r_oa_ + (T_m - T_a) * r_am_ + α_s * sw + qint + qh)
    du[2] = c_m_ * r_am_ * (T_a - T_m)
    nothing
end

# ---- trajectory-matching loss ----------------------------------------------
function loss(p)
    fsol = solve(ODEProblem(fest!, [setpoint, setpoint], Tspan, p), Tsit5(),
                 reltol=RT, abstol=AT, saveat=tgrid)
    Ta = [u[1] for u in fsol.u]; Tm = [u[2] for u in fsol.u]
    e1 = Ta .- Ta_data; e2 = Tm .- Tm_data
    dtv = diff(tgrid)
    0.5 * sum(dtv .* ((e1[1:end-1].^2 .+ e2[1:end-1].^2 .+ e1[2:end].^2 .+ e2[2:end].^2) .* 0.5))
end

# =============================================================================
# Continuous adjoint.
#
# With g(u,t) = ½[(T_a-Ta_d)² + (T_m-Tm_d)²], the adjoint λ(t) solves
#     dλ/dt = -Jᵀ λ - ∂g/∂u ,   λ(T) = 0,
#     J = [-(θ1+θ2)   θ2 ;  θ6  -θ6 ],   θ1=c_a r_oa, θ2=c_a r_am, θ6=c_m r_am
# and the gradient is
#     ∂L/∂p_j = ∫ λ(t)ᵀ (∂f/∂p_j) dt .
# We integrate backward in one pass with the 6 gradient components appended.
# =============================================================================
function adjoint_gradient(p)
    fsol = solve(ODEProblem(fest!, [setpoint, setpoint], Tspan, p), Tsit5(), reltol=RT, abstol=AT)
    T = Tspan[2]
    function adj!(dz, z, p_, t)
        u = fsol(t); T_a, T_m = u[1], u[2]
        c_a_, c_m_, r_oa_, r_am_, α_s, qint = p_
        λ1, λ2 = z[1], z[2]
        θ1 = c_a_ * r_oa_; θ2 = c_a_ * r_am_; θ6 = c_m_ * r_am_

        # adjoint dynamics
        dz[1] = (θ1 + θ2) * λ1 - θ6 * λ2 - (T_a - TA_D(t))
        dz[2] = -θ2 * λ1 + θ6 * λ2 - (T_m - TM_D(t))

        # accumulate ∫ λᵀ ∂f/∂p_j dt  (∂f/∂p_j below)
        t_out = temp_interp(t); sw = sw_interp(t); qh = QHVAC(t)
        dz[3] = λ1 * ((t_out - T_a) * r_oa_ + (T_m - T_a) * r_am_ + α_s * sw + qint + qh)  # ∂f1/∂c_a
        dz[4] = λ2 * (r_am_ * (T_a - T_m))                                                  # ∂f2/∂c_m
        dz[5] = λ1 * (c_a_ * (t_out - T_a))                                                 # ∂f1/∂r_oa
        dz[6] = λ1 * (c_a_ * (T_m - T_a)) + λ2 * (c_m_ * (T_a - T_m))                       # ∂f/∂r_am
        dz[7] = λ1 * (c_a_ * sw)                                                            # ∂f1/∂α_solar
        dz[8] = λ1 * c_a_                                                                   # ∂f1/∂q_internal
        nothing
    end
    asol = solve(ODEProblem(adj!, zeros(8), (T, 0.0), p), Tsit5(), reltol=RT, abstol=AT)
    -asol.u[end][3:8]          # ∫_T^0 = -∫_0^T
end

# ---- 1) gradient at the truth should vanish --------------------------------
println("adjoint gradient at TRUE params (≈0 in log-space):")
g_true = adjoint_gradient(p_true)
println("   ", round.(g_true .* p_true, sigdigits=3))

# ---- 2) gradient check vs central finite differences ------------------------
p_test = [6.0e-4, 2.5e-4, 900.0, 500.0, 12.0, 5000.0]
g_adj = adjoint_gradient(p_test)
g_fd  = zeros(6)
for j in 1:6
    eps = 1e-6 * (1 + abs(p_test[j]))
    pp = copy(p_test); pp[j] += eps
    pm = copy(p_test); pm[j] -= eps
    g_fd[j] = (loss(pp) - loss(pm)) / (2eps)
end
println("\n=== adjoint vs finite-difference gradient (perturbed point) ===")
println("param:    ", ["c_a", "c_m", "r_oa", "r_am", "α_solar", "q_internal"])
println("adjoint:  ", round.(g_adj, sigdigits=4))
println("finite-diff: ", round.(g_fd, sigdigits=4))

# ---- 3) minimize with Adam in log-space ------------------------------------
function adam!(s, grad_s, m, v, it; lr=0.15)
    β1, β2, ϵ = 0.9, 0.999, 1e-8
    @. m = β1 * m + (1 - β1) * grad_s
    @. v = β2 * v + (1 - β2) * grad_s^2
    m̂ = m ./ (1 - β1^it); v̂ = v ./ (1 - β2^it)
    @. s -= lr * m̂ / (sqrt(v̂) + ϵ)
end

s = log.(p_test)
m = zeros(6); v = zeros(6)
println("\n=== Adam (log-space, adjoint gradients) ===")
println("true:      ", round.(p_true, sigdigits=4))
for it in 1:300
    p = exp.(s)
    adam!(s, adjoint_gradient(p) .* p, m, v, it)
    if it in [10, 50, 100, 200, 300]
        println("it=", rpad(it, 4), " p=", round.(exp.(s), sigdigits=4), "  L=", round(loss(exp.(s)), sigdigits=3))
    end
end
p_rec = exp.(s)
println("\nrecovered: ", round.(p_rec, sigdigits=4))
println("rel err:   ", round.((p_rec .- p_true) ./ p_true, sigdigits=3))

# ---- 4) visualize the fit ---------------------------------------------------
using Gadfly, Cairo
df_fit = DataFrame(time=tgrid, T_air_obs=Ta_data, T_mass_obs=Tm_data)
fsol_rec = solve(ODEProblem(fest!, [setpoint, setpoint], Tspan, p_rec), Tsit5(), reltol=RT, abstol=AT, saveat=tgrid)
df_fit.T_air_fit = [u[1] for u in fsol_rec.u]
df_fit.T_mass_fit = [u[2] for u in fsol_rec.u]

p_fit = plot(df_fit,
    layer(x=:time, y=:T_air_obs, Geom.line, color=["T_air (data)"]),
    layer(x=:time, y=:T_air_fit, Geom.line, color=["T_air (fit)"]),
    layer(x=:time, y=:T_mass_obs, Geom.line, color=["T_mass (data)"]),
    layer(x=:time, y=:T_mass_fit, Geom.line, color=["T_mass (fit)"]),
    Guide.xlabel("Time (hours)"), Guide.ylabel("Temperature (°F)"),
    Guide.title("Data vs. adjoint-estimated model"), Guide.colorkey(title=""))
draw(PNG("inverse_fit.png", 12inch, 5inch), p_fit)   # renders headless
println("saved inverse_fit.png")
