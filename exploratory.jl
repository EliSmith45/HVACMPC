using DifferentialEquations
using NLsolve
using DataFrames
using Gadfly
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

# The HVAC cooling unit will be modeled as a simple on/off controller that turns on when the air temp rises above
# a setpoint and turns off when the air temp falls below the setpoint (plus or minus a deadband). The HVAC
# unit will have a maximum capacity, constant speed fan, and will always try to make 55 degree air (though
# the supply air temp may be greater if return air is too high given the capacity). 

c_a = .005 # thermal capacitance of the air in the house (°F/BTU) accounting for volume, density, and specific heat. This coefficient says air temp will change by c_a degrees for every BTU of heat added to or removed from the air.
c_m = .001 # thermal capacitance of the thermal mass in the house (°F/BTU) accounting for volume, density, and specific heat. This coefficient says thermal mass temp will change by c_m degrees for every BTU of heat added to or removed from the thermal mass.
r_oa = 1000 # BTU/(°F), thermal resistance between the outside air and the air
r_am = 100 # BTU/(°F), thermal resistance between the air and the thermal mass
α_solar = 0.0001 # coefficient for solar radiation heat gain (°F/(W/m²)) accounting for surface area and absorptivity of the house
q_internal = 6000 # internal heat gains (BTU/hr) from lights, appliances, and people. This is a constant value for now but could be made time-varying in the future.
t_ref = hist.time[1] # reference time for the weather data. This is the time corresponding to the first ODE time step.

# set up the HVAC controller parameters
setpoint = 70.0 # desired indoor air temperature (°F)
deadband = 2.0 # deadband for the HVAC controller (°F)
hvac_capacity = 36000.0 # maximum cooling capacity of the HVAC unit (BTU/hr)
supply_temp = 55.0 # supply air temperature of the HVAC unit (°F)
design_cfm = 1200.0 # design airflow rate of the HVAC unit (CFM)
min_run_time = 0.25 # minimum run time of the HVAC unit (hours) to prevent short cycling. This is a constant value for now but could be made time-varying in the future.

# set up the ODE function for the 2-state RC model
function rc_model!(du, u, p, t)

    T_air, T_mass = u # unpack the state variables
    
    #unpack parameters
    oat_interp, rh_interp, sw_interp = p.weather
    c_a, c_m, r_oa, r_am, α_solar, q_internal = p.coeffs 
    setpoint, deadband, hvac_capacity, supply_temp, design_cfm, min_run_time = p.hvac

    # get the weather data for the current time step
    t_out = oat_interp(t)
    rh_out = rh_interp(t)
    sw_out = sw_interp(t)

    # get q_hvac based on the current air temperature and the setpoint/deadband
    if T_air > setpoint - deadband
        q_hvac = -clamp(1.08 * design_cfm * (T_air - supply_temp), 0.0, hvac_capacity) # HVAC is on and providing maximum cooling
    elseif T_air < setpoint + deadband
        q_hvac = 0.0 # HVAC is off
    else
        q_hvac = 0.0 # HVAC is off
    end


    #calculate derivatives
    du[1] = c_a * (((t_out - T_air) * r_oa) + ((T_mass - T_air) * r_am) + (α_solar * sw_out) + q_internal + q_hvac) 
    du[2] = c_m * ((T_air - T_mass) * r_am)

end


# use DataInterpolations.jl instead to create a linear interpolation of the weather data for more efficient lookups
hist.rel_time = Dates.value.(hist.time .- t_ref) ./ 3600000 # convert the time to hours since the reference time
temp_interp = LinearInterpolation(hist.temperature_2m, hist.rel_time)
rh_interp = LinearInterpolation(hist.relative_humidity_2m, hist.rel_time)
sw_interp = LinearInterpolation(hist.shortwave_radiation, hist.rel_time)

u0 = [setpoint, setpoint]
params = (weather=(temp_interp, rh_interp, sw_interp), coeffs=(c_a, c_m, r_oa, r_am, α_solar, q_internal), hvac=(setpoint, deadband, hvac_capacity, supply_temp, design_cfm, min_run_time)) # params for the ODE function

ode = ODEProblem(rc_model!, u0, (0.0, 24.0*6.0), params) # solve for 7 days
sol = solve(ode, Tsit5(), saveat=1.0) # save the solution at every hour
sol.u[10]
sol.t[10]

# plot the results
df = DataFrame(time=sol.t, T_air=[u[1] for u in sol.u], T_mass=[u[2] for u in sol.u])
df.temperature_2m = temp_interp.(df.time)
df.shortwave_radiation = sw_interp.(df.time)

# plot 2 time series of air temp and outdoor air temp on the same plot
plot(df, layer(x=:time, y=:T_air, Geom.line, Theme(default_color="blue")),
     layer(x = :time, y = :T_mass, Geom.line, Theme(default_color="green")),
     layer(x=:time, y=:temperature_2m, Geom.line, Theme(default_color="red")),
     Guide.xlabel("Time (hours)"),
     Guide.ylabel("Temperature (°F)"),
     Guide.title("Indoor and Outdoor Air Temperature Over Time"),
     Scale.y_continuous(minvalue=50, maxvalue=90),
     Scale.x_continuous(minvalue=0, maxvalue=24*6))