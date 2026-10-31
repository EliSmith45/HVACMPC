using DifferentialEquations
using DiffEqCallbacks
using NLsolve
using DataFrames
using Gadfly, Cairo
using DataInterpolations

include("openmeteo_weather.jl")

addr = "4716 Liberty Ave, Pittsburgh, PA 15224"
w = get_weather(addr, Date(2023, 6, 1), Date(2023, 6, 7), 7)

wl = w.location
hist = w.historical
fcst = w.forecast


# Create a 2-state RC model for a house with a single thermal mass and a single thermal resistance. 
# The model will track the temperature of the air and the thermal mass in the house. The model will 
# be driven by conductive/convective heat transfer from the outside air, conductive heat transfer
# between the air and the thermal mass, the solar radiation, the internal heat gains, and the capacity
# of the HVAC unit. 

# The HVAC cooling unit is a simple on/off controller: it turns on (full capacity) when the air temp rises
# above setpoint + deadband, and turns off when the air temp falls to setpoint - deadband, with a minimum
# run time to prevent short cycling. The unit always tries to make 55 degree supply air (though the supply
# air temp may be greater if return air is too high given the capacity). On/off transitions are handled with
# callbacks so they can occur exactly between saved time steps.

c_a = .0005 # thermal capacitance of the air in the house (°F/BTU) accounting for volume, density, and specific heat. This coefficient says air temp will change by c_a degrees for every BTU of heat added to or removed from the air.
c_m = .0003 # thermal capacitance of the thermal mass in the house (°F/BTU) accounting for volume, density, and specific heat. This coefficient says thermal mass temp will change by c_m degrees for every BTU of heat added to or removed from the thermal mass.
r_oa = 16000/(90 - 70) # BTU/(°F), thermal resistance between the outside air and the air
r_am = r_oa / 2 # BTU/(°F), thermal resistance between the air and the thermal mass
α_solar = 9.9 # coefficient for solar radiation heat gain (BTU/(W/m²)) accounting for surface area and absorptivity of the house
q_internal = 6000 # internal heat gains (BTU/hr) from lights, appliances, and people. This is a constant value for now but could be made time-varying in the future.
t_ref = hist.time[1] # reference time for the weather data. This is the time corresponding to the first ODE time step.

# set up the HVAC controller parameters
setpoint = 70.0 # desired indoor air temperature (°F)
deadband = 2.0 # deadband for the HVAC controller (°F)
hvac_capacity = 36000.0 # maximum cooling capacity of the HVAC unit (BTU/hr)
supply_temp = 55.0 # supply air temperature of the HVAC unit (°F)
design_cfm = 400 * hvac_capacity / 12000 # design airflow rate of the HVAC unit (CFM)
min_run_time = 0.25 # minimum run time of the HVAC unit (hours) to prevent short cycling. This is a constant value for now but could be made time-varying in the future.


# Create differentiable interpolations of the weather data for use in the ODE. The weather data is given at discrete time steps, but the ODE solver requires continuous functions. We will use cubic spline interpolation to create continuous functions for the outdoor air temperature, relative humidity, and solar radiation.
hist.rel_time = Dates.value.(hist.time .- t_ref) ./ 3600000 # convert the time to hours since the reference time
temp_interp = CubicSpline(hist.temperature_2m, hist.rel_time)
rh_interp = CubicSpline(hist.relative_humidity_2m, hist.rel_time)
sw_interp = CubicSpline(hist.shortwave_radiation, hist.rel_time)


# Mutable container for the discrete (on/off) HVAC controller state. The ODE is
# continuous between events; this state is flipped by the callbacks below.
mutable struct HVACState
    setpoint::Float64
    deadband::Float64
    capacity::Float64      # BTU/hr
    supply_temp::Float64   # °F
    cfm::Float64
    min_run_time::Float64  # hours
    on::Bool
    t_on::Float64          # time (hours) at which the current run started
end

# set up the ODE function for the 2-state RC model (u[3] is an on-run timer)
function rc_model!(du, u, p, t)

    T_air, T_mass, run_timer = u # unpack the state variables

    #unpack parameters
    hvac = p.hvac # mutable HVAC controller state
    oat_interp, rh_interp, sw_interp = p.weather
    c_a, c_m, r_oa, r_am, α_solar, q_internal = p.coeffs 

    # get the weather data for the current time step
    t_out = oat_interp(t)
    rh_out = rh_interp(t)
    sw_out = sw_interp(t)

    # get q_hvac from the discrete on/off state (unit runs at full capacity only)
    
    q_hvac = hvac.on ? -clamp(1.08 * hvac.cfm * (T_air - hvac.supply_temp), 0.0, hvac.capacity) : 0.0
 

    q_oa = (t_out - T_air) * r_oa
    q_thermal_mass = (T_mass - T_air) * r_am
    q_solar = α_solar * sw_out
    q_from_air = ((T_air - T_mass) * r_am)

    #calculate derivatives
    du[1] = c_a * (q_oa + q_thermal_mass + q_solar + q_internal + q_hvac) 
    du[2] = c_m * q_from_air
    du[3] = hvac.on ? 1.0 : 0.0  # advances the run timer only while the unit is on

end



# ---- HVAC event handling ---------------------------------------------------
# Turn ON: T_air rises through setpoint + deadband (upcrossing only).
on_condition(u, t, integ) = u[1] - (integ.p.hvac.setpoint + integ.p.hvac.deadband)
function on_affect!(integ)
    integ.p.hvac.on = true
    integ.p.hvac.t_on = integ.t
    integ.u[3] = 0.0   # restart the run timer
end

# Turn OFF: run at least min_run_time AND cool to setpoint - deadband. Writing
# the condition as max(...) makes it cross zero at the LATER of the two, and it
# stays continuous, so rootfinding locates the exact stop time between steps.
function off_condition(u, t, integ)
    hvac = integ.p.hvac
    lower = hvac.setpoint - hvac.deadband
    max(u[1] - lower, hvac.min_run_time - u[3])
end
function off_affect!(integ)
    integ.p.hvac.on = false
end

# Saving callback to store intermediary heat gain values
saved = SavedValues(Float64, Tuple{Float64, Float64, Float64, Float64, Float64}) # save the state variables and time at each callback event
function save_func(u, t, integrator)
    T_air, T_mass, run_timer = u # unpack the state variables

    #unpack parameters
    hvac = integrator.p.hvac # mutable HVAC controller state
    oat_interp, rh_interp, sw_interp = integrator.p.weather
    c_a, c_m, r_oa, r_am, α_solar, q_internal = integrator.p.coeffs 

    # get the weather data for the current time step
    t_out = oat_interp(t)
    rh_out = rh_interp(t)
    sw_out = sw_interp(t)

    # get q_hvac from the discrete on/off state (unit runs at full capacity only)
    
    q_hvac = hvac.on ? -clamp(1.08 * hvac.cfm * (T_air - hvac.supply_temp), 0.0, hvac.capacity) : 0.0
 

    q_oa = (t_out - T_air) * r_oa
    q_thermal_mass = (T_mass - T_air) * r_am
    q_solar = α_solar * sw_out
    q_from_air = ((T_air - T_mass) * r_am)

    (q_oa, q_thermal_mass, q_solar, q_internal, q_hvac)
end

cb_on  = ContinuousCallback(on_condition, on_affect!, nothing)  # upcrossing only
cb_off = ContinuousCallback(off_condition, nothing, off_affect!) # downcrossing only
cb_saving = SavingCallback(save_func, saved)

cb = CallbackSet(cb_on, cb_off, cb_saving)

u0 = [setpoint, setpoint, 0.0]
hvac = HVACState(setpoint, deadband, hvac_capacity, supply_temp, design_cfm, min_run_time, false, -Inf)
params = (weather=(temp_interp, rh_interp, sw_interp), coeffs=(c_a, c_m, r_oa, r_am, α_solar, q_internal), hvac=hvac) # params for the ODE function

ode = ODEProblem(rc_model!, u0, (0.0, 24.0*6.0), params) # solve for 7 days
@time sol = solve(ode, Tsit5(), callback=cb) # save the solution at every step
sol.u[10]
sol.t[10]
saved.t
saved_vals = saved.saveval


# make data frame of the results
df = DataFrame(time=sol.t, T_air=[u[1] for u in sol.u], T_mass=[u[2] for u in sol.u])
df.temperature_2m = temp_interp.(df.time)
df.shortwave_radiation = sw_interp.(df.time)

#interpolate the saved heat gain values to the solution time steps
q_oa_interp = CubicSpline([vals[1] for vals in saved_vals], [t for t in saved.t])
q_thermal_mass_interp = CubicSpline([vals[2] for vals in saved_vals], [t for t in saved.t])
q_solar_interp = CubicSpline([vals[3] for vals in saved_vals], [t for t in saved.t])
q_internal_interp = CubicSpline([vals[4] for vals in saved_vals], [t for t in saved.t])
q_hvac_interp = CubicSpline([vals[5] for vals in saved_vals], [t for t in saved.t])

# add the interpolated heat gain values to the data frame
df.q_oa = q_oa_interp.(df.time)
df.q_thermal_mass = q_thermal_mass_interp.(df.time)
df.q_solar = q_solar_interp.(df.time)
df.q_internal = q_internal_interp.(df.time)
df.q_hvac = q_hvac_interp.(df.time)

# plot 2 time series of air temp and outdoor air temp on the same plot
p1 = plot(df, layer(x=:time, y=:T_air, Geom.line, color = ["Indoor Air"]),
     layer(x = :time, y = :T_mass, Geom.line, color = ["Thermal Mass"]),
     layer(x=:time, y=:temperature_2m, Geom.line, color = ["Outdoor Air"]),
     Guide.xlabel("Time (hours)"),
     Guide.ylabel("Temperature (°F)"),
     Guide.title("Indoor and Outdoor Air Temperature Over Time"),
     Scale.y_continuous(minvalue=50, maxvalue=90),
     Scale.x_continuous(minvalue=0, maxvalue=24*6))

# Plot each of the heat gain components on the same plot.
p2 = plot(df,
    layer(x=:time, y=:q_oa,          Geom.line, color=["Outside Air"]),
    layer(x=:time, y=:q_thermal_mass,Geom.line, color=["Thermal Mass"]),
    layer(x=:time, y=:q_solar,       Geom.line, color=["Solar"]),
    layer(x=:time, y=:q_internal,    Geom.line, color=["Internal"]),
    layer(x=:time, y=:q_hvac,        Geom.line, color=["HVAC"]),
    #Scale.color_discrete_manual("blue", "green", "red", "orange", "purple"),
    Guide.xlabel("Time (hours)"),
    Guide.ylabel("Heat Gain (BTU/hr)"),
    Guide.title("Heat Gain Components Over Time"),
    Scale.y_continuous(minvalue=-40000, maxvalue=40000),
    Scale.x_continuous(minvalue=0, maxvalue=24*6),
    Guide.colorkey(title="Component")
)

