# Download hourly weather data from the OpenMeteo API.
#
#   * Historical data (temperature, humidity, solar intensity) for a date range
#   * N-day forecast (up to 16 days), NOAA models for US locations
#   * Geocodes a US street address -> latitude / longitude
#
# Required packages:
#   using Pkg; Pkg.add(["HTTP", "JSON3", "DataFrames"])

using HTTP
using JSON3
using Dates
using DataFrames

const HOURLY_VARS = ["temperature_2m", "relative_humidity_2m", "shortwave_radiation"]

# NOAA forecast models offered by OpenMeteo (US locations):
#   gfs_global        NOAA GFS global model (up to 16 days)          <- default
#   gfs_hrrr          NOAA HRRR high-resolution CONUS (up to 48 h)
#   ncep_nbm_conus    NOAA National Blend of Models (up to ~7 days)
const NOAA_MODEL = "gfs_global"

# ---------------------------------------------------------------------------
# Geocoding
# ---------------------------------------------------------------------------

"""
    geocode_address(address::String) -> NamedTuple

Geocode a US address to (latitude, longitude). Uses the US Census Bureau
geocoder first (handles full street addresses); falls back to the OpenMeteo
geocoding API (handles city / place names).
"""
function geocode_address(address::String)
    # 1) US Census Bureau geocoder (best for full US street addresses)
    census_url = "https://geocoding.geo.census.gov/geocoder/locations/onelineaddress"
    r = HTTP.get(census_url; query=Dict(
        "address"   => address,
        "benchmark" => "Public_AR_Current",
        "format"    => "json",
    ))
    if r.status == 200
        data = JSON3.read(String(r.body))
        matches = data.result.addressMatches
        if !isempty(matches)
            m = matches[1]
            return (;
                latitude        = Float64(m.coordinates.y),
                longitude       = Float64(m.coordinates.x),
                matched_address = String(m.matchedAddress),
                source          = "US Census",
            )
        end
    end

    # 2) OpenMeteo geocoder fallback (city / place names)
    om_url = "https://geocoding-api.open-meteo.com/v1/search"
    r = HTTP.get(om_url; query=Dict(
        "name"     => address,
        "count"    => "1",
        "language" => "en",
        "format"   => "json",
    ))
    if r.status == 200
        data = JSON3.read(String(r.body))
        if !isempty(data.results)
            res = data.results[1]
            return (;
                latitude        = Float64(res.latitude),
                longitude       = Float64(res.longitude),
                matched_address = "$(res.name), $(res.admin1), $(res.country)",
                source          = "OpenMeteo",
            )
        end
    end

    error("Could not geocode address: $address")
end

# ---------------------------------------------------------------------------
# Response parsing
# ---------------------------------------------------------------------------

function hourly_to_dataframe(data)
    h = data.hourly
    df = DataFrame(time=DateTime.(collect(h.time)))
    for name in propertynames(h)
        name === :time && continue
        df[!, name] = collect(getproperty(h, name))
    end
    return df
end

# ---------------------------------------------------------------------------
# Data downloads
# ---------------------------------------------------------------------------

"""
    download_historical(lat, lon, start_date, end_date; kwargs...) -> DataFrame

Download hourly historical weather from the OpenMeteo archive API.

Columns: `time`, `temperature_2m` (°F), `relative_humidity_2m` (%),
`shortwave_radiation` (W/m², global horizontal irradiance).
"""
function download_historical(latitude, longitude, start_date, end_date;
                             hourly=HOURLY_VARS,
                             temperature_unit="fahrenheit",
                             timezone="auto")
    url = "https://archive-api.open-meteo.com/v1/archive"
    query = Dict(
        "latitude"         => string(latitude),
        "longitude"        => string(longitude),
        "start_date"       => string(start_date),   # Date or "YYYY-MM-DD"
        "end_date"         => string(end_date),
        "hourly"           => join(hourly, ","),
        "temperature_unit" => temperature_unit,
        "timezone"         => timezone,
    )
    r = HTTP.get(url; query=query, status_exception=true)
    data = JSON3.read(String(r.body))
    data.hourly === nothing && error("Archive API returned no hourly data: $(String(r.body)[1:min(end, 300)])")
    return hourly_to_dataframe(data)
end

"""
    download_forecast(lat, lon, forecast_days; kwargs...) -> DataFrame

Download an hourly N-day forecast (1 ≤ N ≤ 16) from the OpenMeteo forecast API
using a NOAA model (`gfs_global` by default).
"""
function download_forecast(latitude, longitude, forecast_days::Integer;
                           hourly=HOURLY_VARS,
                           temperature_unit="fahrenheit",
                           timezone="auto",
                           models=NOAA_MODEL)
    forecast_days = clamp(Int(forecast_days), 1, 16)
    url = "https://api.open-meteo.com/v1/forecast"
    query = Dict(
        "latitude"         => string(latitude),
        "longitude"        => string(longitude),
        "forecast_days"    => string(forecast_days),
        "hourly"           => join(hourly, ","),
        "temperature_unit" => temperature_unit,
        "timezone"         => timezone,
        "models"           => models,
    )
    r = HTTP.get(url; query=query, status_exception=true)
    data = JSON3.read(String(r.body))
    data.hourly === nothing && error("Forecast API returned no hourly data: $(String(r.body)[1:min(end, 300)])")
    return hourly_to_dataframe(data)
end

"""
    get_weather(address, start_date, end_date, forecast_days; kwargs...) -> NamedTuple

One-stop helper: geocode `address`, then download historical data over
`[start_date, end_date]` and an N-day forecast.

Returns `(; location, historical, forecast)`.
"""
function get_weather(address::String, start_date, end_date, forecast_days::Integer;
                     kwargs...)
    location = geocode_address(address)
    hist = download_historical(location.latitude, location.longitude,
                               start_date, end_date; kwargs...)
    fcst = download_forecast(location.latitude, location.longitude,
                             forecast_days; kwargs...)
    return (; location, historical=hist, forecast=fcst)
end

# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

if abspath(PROGRAM_FILE) == @__FILE__
    address       = length(ARGS) >= 1 ? ARGS[1] : (print("US address: "); readline())
    start_date    = length(ARGS) >= 2 ? Date(ARGS[2]) : (print("Start date (YYYY-MM-DD): "); Date(readline()))
    end_date      = length(ARGS) >= 3 ? Date(ARGS[3]) : (print("End date (YYYY-MM-DD): "); Date(readline()))
    forecast_days = length(ARGS) >= 4 ? parse(Int, ARGS[4]) : (print("Forecast days (1-16): "); parse(Int, readline()))

    loc = geocode_address(address)
    println("Location: $(loc.matched_address)  (source: $(loc.source))")
    println("Coordinates: lat=$(loc.latitude), lon=$(loc.longitude)\n")

    hist = download_historical(loc.latitude, loc.longitude, start_date, end_date)
    println("Historical data ($start_date → $end_date): $(nrow(hist)) hourly rows")
    show(first(hist, 5), allcols=true); println()

    fcst = download_forecast(loc.latitude, loc.longitude, forecast_days)
    println("\nForecast ($forecast_days days): $(nrow(fcst)) hourly rows")
    show(first(fcst, 5), allcols=true); println()
end
