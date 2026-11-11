# Example 2: PID temperature control + annual HVAC energy estimation.
#
#   * Build the two-zone RC model with real Pittsburgh weather (full year)
#   * Run a per-zone discrete-time PID controller in closed loop (hourly)
#   * Estimate annual electrical energy from the resulting Q_hvac trajectory
#     using a biquadratic COP vs. lift (T_oa) and part-load ratio model.

using HVACMPC
using Plots
using DataFrames
using Dates
using LinearAlgebra

# ---------------------------------------------------------------------------
# Building model (same two-zone geometry as example1.jl)
# ---------------------------------------------------------------------------
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

# ---------------------------------------------------------------------------
# Real weather (full year of hourly historical data for Pittsburgh, PA)
# ---------------------------------------------------------------------------
weather_address = "Pittsburgh, PA"
weather_start   = Date("2024-07-01")
weather_end     = Date("2025-07-01")

weather_loc = geocode_address(weather_address)
println("Weather location: $(weather_loc.matched_address)")
println("Coordinates: lat=$(weather_loc.latitude), lon=$(weather_loc.longitude)")

weather_df = download_historical(weather_loc.latitude, weather_loc.longitude,
                                 weather_start, weather_end)
println("Downloaded $(nrow(weather_df)) hourly weather rows ($(weather_start) → $(weather_end))")

T_oa, I_solar = weather_from_openmeteo(weather_df)

tspan = (0.0, Float64(nrow(weather_df) - 1))
occ_start = [8.0, 8.0]
occ_end   = [18.0, 18.0]

# The HVAC load is produced by the controller in closed loop; the model only
# needs a placeholder callable at construction time.
Q_dummy = (t -> 0.0, t -> 0.0)
topo = BuildingGraphModel([z1, z2], [w1, w2, w3], T_oa, I_solar, Q_dummy, occ_start, occ_end)

u0 = fill(72.0, topo.num_nodes)
topo.params.Q_int_occ   .= [5.0, 5.0]
topo.params.Q_int_unocc .= [0.5, 0.5]

# ---------------------------------------------------------------------------
# HVAC equipment: biquadratic COP vs. lift (T_oa) and part-load ratio, with a
# maximum capacity that the controller output is clamped to.
# ---------------------------------------------------------------------------
hvac1 = HVACSystem(100)
hvac2 = HVACSystem(100)
hvac  = [hvac1, hvac2]

# ---------------------------------------------------------------------------
# PID controller (one per zone).  Output limits match the HVAC capacities.
# ---------------------------------------------------------------------------
controller = PIDController(2;
                           kp = 1.80, ki = 18.00, kd = 0.0,
                           setpoint = [72.0, 72.0],
                           q_min = [-300.0, -z2PeakCoolingLoad],
                           q_max = [10.0, 10.0],
                           dt = 0.05)

# ---------------------------------------------------------------------------
# Closed-loop simulation over the year (hourly zero-order-hold).
# ---------------------------------------------------------------------------
@time result = simulate_closed_loop(topo, u0, (0.0, 872.0), controller, hvac;
                                    dt = 0.05, progress = true)

Plots.plot(result.times, result.air_temps[1, :], label = "z1 air", lw = 1.5)
Plots.plot!(result.times, topo.T_oa(result.times), label = "T_oa", lw = 1.5, ls = :dash, color = :black)
Plots.plot(result.times, result.q_hvac[1, :], label = "z1 Q_hvac", lw = 1.5)

Plots.plot(result.times, result.air_temps[2, :], label = "z2 air", lw = 1.5)


# ---------------------------------------------------------------------------
# Annual energy consumption.
# ---------------------------------------------------------------------------
energy = estimate_energy(hvac, T_oa, result.times, result.q_hvac)

println("\n=== Annual HVAC energy ===")
for (i, e) in enumerate(energy.per_zone)
    println("zone $i : ", round(e, digits = 1), " kWh  (", round(e / 1000, digits = 2), " MWh)")
end
println("total   : ", round(energy.total, digits = 1), " kWh  (",
        round(energy.total / 1000, digits = 2), " MWh)")

# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------
Plots.plot(result.times, result.air_temps[1, :], label = "z1 air", lw = 1.5)
Plots.plot!(result.times, result.air_temps[2, :], label = "z2 air", lw = 1.5)
Plots.hline!([72.0], label = "setpoint", ls = :dash, color = :black)
Plots.xlabel!("Time (hours)")
Plots.ylabel!("Temperature (°F)")
Plots.title!("Zone Air Temperatures under PID Control")

Plots.plot(result.times, result.q_hvac[1, :], label = "z1 Q_hvac", lw = 1.5)
Plots.plot!(result.times, result.q_hvac[2, :], label = "z2 Q_hvac", lw = 1.5)
Plots.xlabel!("Time (hours)")
Plots.ylabel!("Q_hvac (tons)")
Plots.title!("Delivered HVAC Load")

Plots.plot(result.times, energy.power[1, :], label = "z1 electrical", lw = 1.5)
Plots.plot!(result.times, energy.power[2, :], label = "z2 electrical", lw = 1.5)
Plots.xlabel!("Time (hours)")
Plots.ylabel!("Electrical power (kW)")
Plots.title!("Electrical Power Draw")
