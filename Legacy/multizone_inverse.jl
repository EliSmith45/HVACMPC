# =============================================================================
# multizone_inverse.jl
# -----------------------------------------------------------------------------
# Arbitrary-N-zone RC model + parameter estimation using **SciML adjoint
# sensitivity** (Zygote autodiff through SciMLSensitivity), instead of the
# hand-derived continuous adjoint in inverse_problem.jl.
#
# Model (zone i, with air and thermal-mass states):
#   dT_a,i/dt = c_a,i [ r_oa,i (T_out - T_a,i)
#                       + r_am,i (T_m,i - T_a,i)
#                       + Σ_j r_zz,ij (T_a,j - T_a,i)        # adjacent zones
#                       + α_i · SW(t)                         # optional solar
#                       + q_int,i                             # optional internal
#                       + q_hvac,i(t) ]                       # known, per zone
#   dT_m,i/dt = c_m,i · r_am,i (T_a,i - T_m,i)
#
# Design decisions:
#   * The HVAC input is NOT a feedback controller. q_hvac,i(t) is a prescribed,
#     per-zone cooling schedule (a smooth pseudo-random signal within each
#     zone's capacity), chosen to be informative for estimation. This makes the
#     estimation model a *smooth ODE*, so a standard adjoint method applies.
#   * "May or may not have" solar / internal loads / connections is encoded
#     STRUCTURALLY: absent features simply do not appear in the parameter
#     vector. Selection matrices (solar, internal) and an incidence matrix
#     (zone-zone edges) inject the features, so no zero ever enters log-space.
#   * The RHS is out-of-place; the exogenous inputs (weather, HVAC schedule) are
#     analytic in t and constant w.r.t. the parameters, so they are wrapped in
#     Zygote.@ignore (skips their trace under Zygote) but remain ForwardDiff-safe,
#     allowing the gradient check to compare Zygote adjoints vs ForwardDiff.
#   * Parameters are optimized in log-space (all physical params are > 0), so
#     the wildly different scales are treated uniformly.
#
# Parameter vector θ (flat, length nθ = 4n + n_solar + n_qint + n_edges):
#   c_a (n) | c_m (n) | r_oa (n) | r_am (n) | α (n_solar) | q_int (n_qint) | r_zz (n_edges)
# =============================================================================

using DifferentialEquations
using DataInterpolations
using LinearAlgebra
using Statistics
using Random
using Printf

using SciMLSensitivity
using Zygote
using ForwardDiff

# -----------------------------------------------------------------------------
# Model construction
# -----------------------------------------------------------------------------
struct ZoneModel
    nz::Int
    n_solar::Int
    n_qint::Int
    n_edges::Int
    nθ::Int
    solar_idx::Vector{Int}              # zones that have a solar load
    qint_idx::Vector{Int}               # zones that have an internal load
    edges::Vector{Tuple{Int,Int}}       # zone-zone connections (undirected)
    S_solar::Matrix{Float64}            # nz × n_solar selection
    S_qint::Matrix{Float64}             # nz × n_qint selection
    D::Matrix{Float64}                  # nz × n_edges incidence
    T_out::Any                          # outdoor-air temperature (°F)
    sw::Any                             # solar radiation (W/m²)
    q_hvac::Vector{Any}                 # per-zone cooling schedules (BTU/hr)
    rhs::Function
end

function ZoneModel(nz::Int,
                   has_solar::AbstractVector{Bool},
                   has_internal::AbstractVector{Bool},
                   edges::AbstractVector{Tuple{Int,Int}},
                   T_out, sw, q_hvac)
    length(has_solar)    == nz || error("has_solar must have length nz")
    length(has_internal) == nz || error("has_internal must have length nz")
    length(q_hvac)       == nz || error("q_hvac must have length nz")

    solar_idx = findall(has_solar)
    qint_idx  = findall(has_internal)
    n_solar, n_qint, n_edges = length(solar_idx), length(qint_idx), length(edges)

    S_solar = zeros(nz, n_solar)
    for (k, i) in enumerate(solar_idx); S_solar[i, k] = 1.0; end
    S_qint = zeros(nz, n_qint)
    for (k, i) in enumerate(qint_idx); S_qint[i, k] = 1.0; end
    D = zeros(nz, n_edges)
    for (k, (i, j)) in enumerate(edges); D[i, k] = 1.0; D[j, k] = -1.0; end

    # flat parameter index ranges
    i_ca   = 1:nz
    i_cm   = nz+1:2nz
    i_roa  = 2nz+1:3nz
    i_ram  = 3nz+1:4nz
    i_alp  = 4nz+1:4nz+n_solar
    i_qint = 4nz+n_solar+1:4nz+n_solar+n_qint
    i_zz   = 4nz+n_solar+n_qint+1:4nz+n_solar+n_qint+n_edges

    # Out-of-place, Zygote-clean RHS.  State layout is interleaved:
    #   u = [T_a1, T_m1, T_a2, T_m2, ..., T_an, T_mn]
    function rhs(u, p, t)
        Ta = u[1:2:end]
        Tm = u[2:2:end]

        # exogenous forcings: analytic in t (Dual-safe for ForwardDiff) and
        # constant w.r.t. the estimated parameters, so @ignore skips their trace
        # under Zygote (faster) while ForwardDiff still passes straight through.
        t_out = Zygote.@ignore T_out(t)
        sw_t  = Zygote.@ignore sw(t)
        qh    = Zygote.@ignore [f(t) for f in q_hvac]

        c_a   = p[i_ca]
        c_m   = p[i_cm]
        r_oa  = p[i_roa]
        r_am  = p[i_ram]
        α     = p[i_alp]
        qint  = p[i_qint]
        r_zz  = p[i_zz]

        α_full    = S_solar * α          # nz-vector with zeros where no solar
        qint_full = S_qint * qint        # nz-vector with zeros where no internal
        zz        = -D * (r_zz .* (D' * Ta))   # zone-zone conduction

        dTa = c_a .* (r_oa .* (t_out .- Ta) .+ r_am .* (Tm .- Ta) .+ zz
                      .+ α_full .* sw_t .+ qint_full .+ qh)
        dTm = c_m .* r_am .* (Ta .- Tm)

        # interleave [dTa1, dTm1, dTa2, dTm2, ...] without mutation
        return vec(permutedims(hcat(dTa, dTm)))
    end

    nθ = 4nz + n_solar + n_qint + n_edges
    ZoneModel(nz, n_solar, n_qint, n_edges, nθ, solar_idx, qint_idx, collect(edges),
              S_solar, S_qint, D, T_out, sw, q_hvac, rhs)
end

# pack the named pieces into the flat θ vector in the documented order
function packθ(m::ZoneModel, c_a, c_m, r_oa, r_am, α, qint, r_zz)
    vcat(c_a, c_m, r_oa, r_am, α, qint, r_zz)
end

param_labels(m::ZoneModel) = vcat(
    ["c_a[$i]"        for i in 1:m.nz],
    ["c_m[$i]"        for i in 1:m.nz],
    ["r_oa[$i]"       for i in 1:m.nz],
    ["r_am[$i]"       for i in 1:m.nz],
    ["α[$i]"          for i in m.solar_idx],
    ["q_int[$i]"      for i in m.qint_idx],
    ["r_zz[$(a),$(b)]" for (a, b) in m.edges],
)

# -----------------------------------------------------------------------------
# Demo: 3 zones, each with different features and different HVAC input
# -----------------------------------------------------------------------------
nz = 3
has_solar    = [true,  false, true]     # zones 1,3 have solar
has_internal = [true,  true,  false]    # zones 1,2 have internal loads
edges        = [(1, 2), (2, 3)]         # 1–2 and 2–3 connected; 1–3 not

Tspan = (0.0, 36.0)                     # 1.5 days of excitation
tgrid = collect(0.0:0.1:Tspan[2])
RT, AT = 1e-8, 1e-10                    # tight tolerances (see repo notes)

# Synthetic weather (self-contained).  Swap for OpenMeteo interpolants if desired:
#   include("openmeteo_weather.jl"); ... CubicSpline(...)
T_out(t) = 80.0 + 12.0 * sin(2π * t / 24.0)
sw(t)    = max(0.0, 900.0 * sin(π * (mod(t, 24.0) - 6.0) / 12.0))

# Per-zone HVAC: a smooth pseudo-random cooling signal inside each zone's
# capacity.  Each zone gets a DIFFERENT profile (different capacity and random
# phase/amplitude) to make all couplings identifiable.  It is analytic in t so
# both Zygote (reverse) and ForwardDiff (forward) can differentiate the RHS.
function smooth_cooling(rng, capacity, n_harmonics=6)
    freqs  = (2π / Tspan[2]) .* collect(1:n_harmonics)
    phases = 2π .* rand(rng, n_harmonics)
    amps   = rand(rng, n_harmonics)
    amps  .*= 0.5 * capacity / sum(amps)      # peak excursion = capacity/2
    t -> -capacity / 2 - sum(a * sin(f * t + φ) for (a, f, φ) in zip(amps, freqs, phases))
end
caps  = [36000.0, 24000.0, 30000.0]
q_hvac = [smooth_cooling(MersenneTwister(100 + i), caps[i]) for i in 1:nz]

m = ZoneModel(nz, has_solar, has_internal, edges, T_out, sw, q_hvac)

# true parameters (in flat θ order)
θ_true = packθ(m,
    [5.0e-4, 6.0e-4, 4.5e-4],    # c_a   (°F/BTU)
    [3.0e-4, 3.5e-4, 2.8e-4],    # c_m   (°F/BTU)
    [800.0,  900.0,  750.0],      # r_oa  (BTU/(hr·°F))
    [400.0,  450.0,  380.0],      # r_am  (BTU/(hr·°F))
    [9.9, 11.0],                  # α     (zones 1,3)
    [6000.0, 5000.0],             # q_int (zones 1,2)
    [250.0, 300.0])               # r_zz  (edges 1-2, 2-3)

u0 = repeat([72.0, 72.0], nz)

# ---- generate "observed" data (smooth ODE, no controller) -------------------
sol_data = solve(ODEProblem(m.rhs, u0, Tspan, θ_true), Tsit5();
                 reltol=RT, abstol=AT, saveat=tgrid)
Udata    = reduce(hcat, sol_data.u)
Ta_data  = Udata[1:2:end, :]
Tm_data  = Udata[2:2:end, :]

# ---- trajectory-matching loss, differentiated with SciML adjoints -----------
# InterpolatingAdjoint + ZygoteVJP: standard, stable reverse-mode adjoint.
# Alternatives: BacksolveAdjoint(autojacvec=ZygoteVJP())   # O(1) memory
#               QuadratureAdjoint(autojacvec=ZygoteVJP())  # continuous adjoint
sensealg = InterpolatingAdjoint(autojacvec=ZygoteVJP())
dtv      = diff(tgrid)

function loss(s)
    θ   = exp.(s)                      # log-space -> physical (all params > 0)
    sol = solve(ODEProblem(m.rhs, u0, Tspan, θ), Tsit5();
                reltol=RT, abstol=AT, saveat=tgrid, sensealg=sensealg)
    U  = reduce(hcat, sol.u)
    Ta = U[1:2:end, :]
    Tm = U[2:2:end, :]
    e1 = Ta .- Ta_data
    e2 = Tm .- Tm_data
    # trapezoidal quadrature over the uniform grid (sum over zones, then time)
    quad = 0.5 .* (e1[:,1:end-1].^2 .+ e1[:,2:end].^2 .+ e2[:,1:end-1].^2 .+ e2[:,2:end].^2)
    0.5 * dot(dtv, vec(sum(quad, dims=1)))
end

# ---- 1) adjoint gradient check vs ForwardDiff (forward-mode sensitivity) ----
# Done on a short fixed-step solve so the reference (ForwardDiff, forward-mode)
# is cheap and numerically clean; both methods differentiate the SAME problem.
s_true = log.(θ_true)
s0     = s_true .+ 0.2 .* randn(MersenneTwister(7), length(s_true))

t_check   = collect(0.0:0.1:8.0)          # first 8 h of the data
k         = length(t_check)
dtv_check = diff(t_check)

function loss_check(s)
    θ   = exp.(s)
    sol = solve(ODEProblem(m.rhs, u0, (0.0, 8.0), θ), Tsit5();
                adaptive=false, dt=0.05, saveat=t_check, sensealg=sensealg)
    U  = reduce(hcat, sol.u)
    e1 = U[1:2:end, :] .- Ta_data[:, 1:k]
    e2 = U[2:2:end, :] .- Tm_data[:, 1:k]
    quad = 0.5 .* (e1[:,1:end-1].^2 .+ e1[:,2:end].^2 .+ e2[:,1:end-1].^2 .+ e2[:,2:end].^2)
    0.5 * dot(dtv_check, vec(sum(quad, dims=1)))
end

@time g_adj = Zygote.gradient(loss_check, s0)[1]
@time g_fwd = ForwardDiff.gradient(loss_check, s0)

println("=== adjoint (Zygote + InterpolatingAdjoint) vs ForwardDiff (forward-mode) ===")
println("max |g_adj - g_fwd| = ", round(maximum(abs.(g_adj .- g_fwd)), sigdigits=4))
println("relative L2 error   = ", round(norm(g_adj - g_fwd) / norm(g_fwd), sigdigits=4))

# ---- 2) Adam in log-space (any standard optimizer can be dropped in) --------
function adam!(s, g, mt, vt, it; lr=0.15)
    β1, β2, ϵ = 0.9, 0.999, 1e-8
    @. mt = β1 * mt + (1 - β1) * g
    @. vt = β2 * vt + (1 - β2) * g^2
    m̂ = @. mt / (1 - β1^it)
    v̂ = @. vt / (1 - β2^it)
    @. s -= lr * m̂ / (sqrt(v̂) + ϵ)
end

s  = copy(s0)
mt = zeros(length(s)); vt = zeros(length(s))
labels = param_labels(m)

println("\n=== Adam (log-space, SciML adjoint gradients) ===")
for it in 1:400
    g = Zygote.gradient(loss, s)[1]
    adam!(s, g, mt, vt, it)
    if it in [50, 100, 200, 400]
        println("it=", rpad(it, 4), "  L=", round(loss(s), sigdigits=4))
    end
end

θ_rec = exp.(s)
println("\nparameter      true            recovered       rel.err")
for (lbl, t, r) in zip(labels, θ_true, θ_rec)
    println(rpad(lbl, 15), rpad(round(t, sigdigits=5), 16),
            rpad(round(r, sigdigits=5), 16), round((r - t) / t, sigdigits=3))
end

# ---- 3) visualize the fit ---------------------------------------------------
using DataFrames
using Gadfly, Cairo

sol_rec = solve(ODEProblem(m.rhs, u0, Tspan, θ_rec), Tsit5();
                reltol=RT, abstol=AT, saveat=tgrid)
Urec = reduce(hcat, sol_rec.u)
Ta_rec = Urec[1:2:end, :]

# long-form data frame: one row per (time, zone, obs-or-fit)
obs_long = DataFrame(time = repeat(tgrid, nz),
                     zone = repeat(1:nz, inner=length(tgrid)))
obs_long.T  = vec(Ta_data')      # nz × nt -> long (zone-major, time fastest)
obs_long.kind = fill("observed", nrow(obs_long))
fit_long = DataFrame(time = obs_long.time, zone = obs_long.zone)
fit_long.T    = vec(Ta_rec')
fit_long.kind = fill("fit", nrow(fit_long))
df = vcat(obs_long, fit_long)
df.series = string.(df.kind, " · zone ", df.zone)

p = plot(df,
    x=:time, y=:T, color=:series, Geom.line,
    Guide.xlabel("Time (hours)"), Guide.ylabel("T_air (°F)"),
    Guide.title("Observed vs recovered zone air temperature"),
    Guide.colorkey(title=""))
draw(PNG("multizone_fit.png", 12inch, 5inch), p)
println("\nsaved multizone_fit.png")
