module ModelTraining

using LinearAlgebra, ComponentArrays, DataInterpolations
using DifferentialEquations, SciMLSensitivity
using OrdinaryDiffEqSDIRK
using Optimization, OptimizationNLopt, NLopt
using Random

using ..ModelArchitecture: BuildingGraphModel, _odef

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
                alg = KenCarp4(),
                sensealg = InterpolatingAdjoint(autojacvec = false),
                reltol = 1e-8, abstol = 1e-6,
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


export train!, air_node_residual_loss, air_node_residual_loss_and_gradients,
       air_node_residual_loss_and_gradient, air_node_residual_gradient,
       observation_interpolators

end
