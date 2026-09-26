# =============================================================================
# MWE: continuous-adjoint sensitivity analysis for parameter estimation
# -----------------------------------------------------------------------------
# Minimal scalar ODE:   dx/dt = -a*x + b*u(t),   x(0) = x0
# with a KNOWN forcing u(t) and "observed" trajectory x_data(t).
#
# We estimate (a, b) by minimizing
#     L(a,b) = 1/2 ∫ (x(t) - x_data(t))^2 dt
#
# The continuous adjoint (Lagrange multiplier) method:
#   * forward:   dx/dt   = f(x,p,t),                 x(0) = x0
#   * adjoint:   dλ/dt   = -∂f/∂x · λ - ∂g/∂x,      λ(T) = 0
#   * gradient:  ∇L(p_j) = ∫ λ(t) · ∂f/∂p_j dt
#
# This is exactly the "adjoint sensitivity" machinery behind SciMLSensitivity's
# `BacksolveAdjoint`, written out by hand so every term is visible.
# =============================================================================

using DifferentialEquations
using DataInterpolations
using LinearAlgebra
using Printf

# ---- truth + data ----------------------------------------------------------
a_true, b_true = 0.7, 2.0
x0 = 1.0
Tspan = (0.0, 10.0)

u(t) = sin(2t) + 0.5*cos(t)          # known exogenous forcing

f!(dx, x, p, t) = (dx[1] = -p[1]*x[1] + p[2]*u(t); nothing)

tgrid = collect(range(Tspan[1], Tspan[2], length=500))
sol    = solve(ODEProblem(f!, [x0], Tspan, [a_true, b_true]), Tsit5(), reltol=1e-10, abstol=1e-12)
X_data = [sol(t)[1] for t in tgrid]
X_D    = LinearInterpolation(X_data, tgrid)

# ---- loss ------------------------------------------------------------------
function loss(p)
    s = solve(ODEProblem(f!, [x0], Tspan, p), Tsit5(), reltol=1e-10, abstol=1e-12, saveat=tgrid)
    X = [v[1] for v in s.u]
    0.5 * sum(diff(tgrid) .* ((X[1:end-1] .- X_data[1:end-1]).^2 .+ (X[2:end] .- X_data[2:end]).^2) .* 0.5)
end

# ---- hand-written continuous adjoint ---------------------------------------
function adjoint_gradient(p)
    fsol = solve(ODEProblem(f!, [x0], Tspan, p), Tsit5(), reltol=1e-10, abstol=1e-12)
    T = Tspan[2]
    # augmented state [λ, ga, gb];  ga/gb accumulate ∫ λ ∂f/∂p dt
    function adj!(dz, z, p_, t)
        x = fsol(t)[1]
        λ = z[1]
        dz[1] = p_[1]*λ - (x - X_D(t))      # dλ/dt = -∂f/∂x λ - ∂g/∂x,  ∂f/∂x = -a
        dz[2] = λ*(-x)                       # ∂f/∂a = -x
        dz[3] = λ*u(t)                       # ∂f/∂b = u(t)
        nothing
    end
    asol = solve(ODEProblem(adj!, zeros(3), (T, 0.0), p), Tsit5(), reltol=1e-10, abstol=1e-12)
    -asol.u[end][2:3]                         # integrate backwards -> negate
end

# ---- gradient check vs central differences ---------------------------------
p0 = [1.1, 1.4]                              # a deliberately wrong start
g_adj = adjoint_gradient(p0)
g_fd  = zeros(2)
for j in 1:2
    eps = 1e-6
    pp = copy(p0); pp[j] += eps
    pm = copy(p0); pm[j] -= eps
    g_fd[j] = (loss(pp) - loss(pm)) / (2eps)
end
println("gradient at p0 = $(p0):")
println("  adjoint:      ", round.(g_adj, sigdigits=6))
println("  finite-diff:  ", round.(g_fd, sigdigits=6))

# ---- a few gradient-descent steps ------------------------------------------
p = copy(p0)
println("\ngradient descent:")
for it in 1:500
    g = adjoint_gradient(p)
    global p = p .- 0.1 .* g
    it % 100 == 0 && println("  it=", rpad(it,4), " p=", round.(p, digits=5), "  L=", round(loss(p), sigdigits=3))
end
println("true:   [", a_true, ", ", b_true, "]")
println("recovered: ", round.(p, digits=5))
