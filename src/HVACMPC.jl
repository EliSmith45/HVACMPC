module HVACMPC

using OrdinaryDiffEqSDIRK  # re-export KenCarp4 below

include("WeatherData.jl")
include("ModelArchitecture.jl")
include("EnergyModel.jl")
include("Controls.jl")
include("ModelTraining.jl")

using .WeatherData
using .ModelArchitecture
using .EnergyModel
using .Controls
using .ModelTraining

# Default stiff solver used by `simulate_forward`.
export KenCarp4

# Model architecture.
export ZoneDef, WallDef, BuildingGraphModel, Exogenous,
       make_forward_odefun, make_forward_problem, simulate_forward,
       generate_sample_control, update_exo_inputs

# HVAC equipment / energy modeling.
export HVACSystem, KW_PER_TON, part_load_ratio, efficiency, efficiency_at_load,
       clamp_load, electrical_power, estimate_energy

# Controls.
export AbstractController, PIDController, reset!, compute_control,
       ClosedLoopResult, simulate_closed_loop

# Training / parameter recovery.
export train!, air_node_residual_loss, air_node_residual_loss_and_gradients,
       air_node_residual_loss_and_gradient, air_node_residual_gradient,
       observation_interpolators

# Weather data.
export geocode_address, download_historical, download_forecast, get_weather,
       weather_from_openmeteo

end
