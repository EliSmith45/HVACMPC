using HVACMPC
using Plots
using DataFrames
using Dates
using LinearAlgebra
using Random
using ForwardDiff

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


# -----------------------------------------------------------------------------
# Real weather from OpenMeteo (hourly historical archive).  The building model's
# time axis is measured in hours from the first downloaded row (hour 0 = local
# midnight of the first day), so the daily occupancy/solar schedules stay
# aligned with local clock time.
# -----------------------------------------------------------------------------
weather_address = "Pittsburgh, PA"
weather_start   = Date("2024-07-01")
weather_end     = Date("2025-07-01")

weather_loc = geocode_address(weather_address)
println("Weather location: $(weather_loc.matched_address)  (source: $(weather_loc.source))")
println("Coordinates: lat=$(weather_loc.latitude), lon=$(weather_loc.longitude)")

weather_df = download_historical(weather_loc.latitude, weather_loc.longitude,
                                 weather_start, weather_end)
println("Downloaded $(nrow(weather_df)) hourly weather rows ($(weather_start) → $(weather_end))")

T_oa, I_solar = weather_from_openmeteo(weather_df)

T_oa
I_solar

t_start = 0.0
t_end = 24*3.0 # simulate 3 days of temp trajectories
n_steps_modeled = Int(t_end - t_start + 1) # number of time steps for training

t_steps = collect(range(t_start, t_end, length=n_steps_modeled))

n_zones = 2

# occupancy schedule: each zone occupied 8:00–18:00 every day (24 h clock)
occ_start = [8.0, 8.0]
occ_end   = [18.0, 18.0]

# Weather is an uncontrolled disturbance from OpenMeteo; HVAC control (known,
# open-loop) is generated with one hourly knot per downloaded row.  Internal
# gains Q_int are learned through the parameters.
Q_hvac = generate_sample_control((0.0, T_oa.t[end]), nrow(weather_df), n_zones;
    amplitudes = [z1PeakCoolingLoad, z2PeakCoolingLoad])

topo = BuildingGraphModel([z1, z2], [w1, w2, w3], T_oa, I_solar, Q_hvac, occ_start, occ_end)



# initial conditions and true parameters (absolute units in `topo.params`,
# mapped to log-space for the truth solve)
u0_true = fill(72.0, topo.num_nodes)
topo.params.Q_int_occ   .= [5.0, 5.0]   # true occupied internal gain per zone
topo.params.Q_int_unocc .= [0.5, 0.5]   # true unoccupied internal gain per zone
p_abs_true = copy(topo.params)          # absolute truth, for reporting
p_true_log = log.(topo.params)          # log-space, for the ODE


@time sol_true = simulate_forward(topo, u0_true, (t_start, t_end), params_abs = p_abs_true, T_oa = T_oa, I_solar = I_solar, Q_hvac = Q_hvac,
               alg = KenCarp4(), reltol = 1e-8, abstol = 1e-8)


Plots.plot(sol_true, tspan = (t_start, t_end), 
           xlabel = "Time (hours)", ylabel = "Temperature (°F)",
           title = "Node Temperatures Over Time")



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

# short validation window: first two observation times (keeps the interval
# small while guaranteeing ≥2 points for the adjoint-vs-ForwardDiff check)
tchk = t_obs[1:2]

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
initial_params = p_abs_true .* exp.(0.2 .* randn(length(p_abs_true)))   # initial guess
topo_learned.params .= initial_params
topo_guess = deepcopy(topo_learned)   # for reporting the initial guess

@time result_log = train!(topo_learned, T_obs, t_obs;
                      p_init_perturb = 0.0,
                      stopval = 1e-2,
                      maxeval = 100, 
                      log_space = true)


p_abs_true
p_abs_rec = topo_learned.params   # learned, stored in absolute units
p_abs_initial = topo_guess.params   # initial guess, stored in absolute units

println("\n=== train! (u0, p) recovery ===")
println("retcode                 = ", result_log.retcode)
println("final loss              = ", round(result_log.loss, sigdigits = 5))
println("||u0 - u0_true||        = ", round(norm(result_log.u0 .- u0_true), sigdigits = 5))
println("||p - p_true|| (log)    = ", round(norm(log.(p_abs_rec) - log.(p_abs_true)), sigdigits = 5))
println("||p - p_true|| (log)    = ", round(norm(log.(p_abs_initial) - log.(p_abs_true)), sigdigits = 5))
println("||p - p_true|| (absolute)    = ", round(norm(p_abs_rec - p_abs_true), sigdigits = 5))
println("||p - p_true|| (absolute)    = ", round(norm(p_abs_initial - p_abs_true), sigdigits = 5))


println("Q_int_occ recovered     = ", round.(p_abs_rec.Q_int_occ, sigdigits = 4),
        "  (true ", round.(p_abs_true.Q_int_occ, sigdigits = 4), ")")
println("Q_int_unocc recovered   = ", round.(p_abs_rec.Q_int_unocc, sigdigits = 4),
        "  (true ", round.(p_abs_true.Q_int_unocc, sigdigits = 4), ")")

# Report the identifiable (daytime) solar-gain profile for the first solar node.
# Hours without solar irradiance carry no information, so those coefficients stay
# near their initial guess.
println("α_solar[1,:] recovered  = ", round.(p_abs_rec.α_solar[1, :], sigdigits = 3))
println("α_solar[1,:] true       = ", round.(p_abs_true.α_solar[1, :], sigdigits = 3))

sol_recovered = simulate_forward(topo_learned, result_log.u0, (t_start, t_end), params_abs = p_abs_rec, T_oa = T_oa, I_solar = I_solar, Q_hvac = Q_hvac,
               alg = KenCarp4(), reltol = 1e-8, abstol = 1e-8)
sol_initial = simulate_forward(topo_guess, result_log.u0, (t_start, t_end), params_abs = p_abs_initial, T_oa = T_oa, I_solar = I_solar, Q_hvac = Q_hvac,
               alg = KenCarp4(), reltol = 1e-8, abstol = 1e-8)
sol_true = simulate_forward(topo, result_log.u0, (t_start, t_end), params_abs = p_abs_true, T_oa = T_oa, I_solar = I_solar, Q_hvac = Q_hvac,
               alg = KenCarp4(), reltol = 1e-8, abstol = 1e-8)

Plots.plot(sol_recovered, tspan = (t_start, t_end),
           xlabel = "Time (hours)", ylabel = "Temperature (°F)",
           title = "Recovered Node Temperatures Over Time")

# Plot the true vs recovered air-node trajectories over the training window.
Plots.plot(t_obs, data_air[:, 1], label = "z1 true", lw = 2, color = :blue)
Plots.plot!(t_obs, Array(sol_recovered(t_obs))[1, :], label = "z1 recovered", lw = 2, ls = :dash, color = :green)
Plots.plot!(t_obs, Array(sol_initial(t_obs))[1, :], label = "z1 initial guess", lw = 2, ls = :dash, color = :yellow)

Plots.plot!(t_obs, data_air[:, 2], label = "z2 true", lw = 2, color = :green)



Plots.plot!(t_obs, Array(sol_recovered(t_obs))[1, :], label = "z1 recovered", lw = 2, ls = :dash, color = :blue)
Plots.plot!(t_obs, Array(sol_recovered(t_obs))[3, :], label = "z2 recovered", lw = 2, ls = :dash, color = :green)
Plots.xlabel!("Time (hours)")
Plots.ylabel!("Temperature (°F)")
Plots.title!("True vs Recovered Air-Node Temperatures") 


Plots.plot(data_air[:, 1], Array(sol_recovered(t_obs))[1, :], label = "z1", lw = 2, color = :blue)
Plots.plot(data_air[:, 2], Array(sol_recovered(t_obs))[3, :], label = "z2", lw = 2, color = :green)

