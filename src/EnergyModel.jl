module EnergyModel

# ---------------------------------------------------------------------------
# HVAC equipment model.
#
# Units: thermal load (Q_hvac) is expressed in tons of refrigeration; electrical
# energy is derived from the delivered thermal load and a coefficient of
# performance (COP).  `KW_PER_TON` converts tons of refrigeration to kW of
# thermal load (1 ton = 12,000 BTU/h = 3.516852842 kW).
# ---------------------------------------------------------------------------

const KW_PER_TON = 3.516852842

"""
    HVACSystem(q_max; coeffs = DEFAULT_COP_COEFFS, min_cop = 1.5)

A single HVAC unit with a maximum thermal capacity `q_max` (tons) and a
biquadratic coefficient-of-performance (COP) surface.

The COP is a biquadratic in the "lift" (ambient dry-bulb temperature `T_oa`,
°F) and the part-load ratio `PLR = |q| / q_max ∈ [0, 1]`:

    COP = c[1] + c[2]·T + c[3]·PLR + c[4]·T² + c[5]·T·PLR + c[6]·PLR²

`coeffs` is the 6-element coefficient vector `[c1, ..., c6]` in that order,
and `min_cop` floors the result to keep the surface physical (and to avoid
division by zero / negative COP).  The default surface is a plausible
cooling-oriented example; fit it to the equipment you are modeling.
"""
struct HVACSystem
    q_max::Float64
    coeffs::NTuple{6, Float64}
    min_cop::Float64

    function HVACSystem(q_max::Real; coeffs = DEFAULT_COP_COEFFS, min_cop::Real = 1.5)
        q_max > 0 || throw(ArgumentError("q_max must be positive, got $q_max"))
        min_cop > 0 || throw(ArgumentError("min_cop must be positive, got $min_cop"))
        c = NTuple{6, Float64}(coeffs)
        new(Float64(q_max), c, Float64(min_cop))
    end
end

# Plausible cooling-oriented example: COP ≈ 4 at (T=50 °F, PLR=0),
# ≈ 2.1 at (T=95 °F, PLR=1).
const DEFAULT_COP_COEFFS = (5.0, -0.02, -0.8, 0.0, 0.002, -0.4)

# ---------------------------------------------------------------------------
# Capacity and efficiency
# ---------------------------------------------------------------------------

"""
    part_load_ratio(hvac, q) -> PLR

Part-load ratio of a requested load `q` (tons), clamped to `[0, 1]`.
"""
function part_load_ratio(hvac::HVACSystem, q::Real)
    return clamp(abs(q) / hvac.q_max, 0.0, 1.0)
end

"""
    efficiency(hvac, T_oa, PLR) -> COP

Evaluate the biquadratic COP surface at ambient temperature `T_oa` (°F) and
part-load ratio `PLR`, floored at `hvac.min_cop`.
"""
function efficiency(hvac::HVACSystem, T_oa::Real, plr::Real)
    c = hvac.coeffs
    cop = c[1] + c[2] * T_oa + c[3] * plr +
          c[4] * T_oa^2 + c[5] * T_oa * plr + c[6] * plr^2
    return max(cop, hvac.min_cop)
end

"""
    efficiency_at_load(hvac, T_oa, q) -> COP

Convenience function taking a requested load `q` (tons); the part-load ratio is
derived from `q` via [`part_load_ratio`](@ref).
"""
efficiency_at_load(hvac::HVACSystem, T_oa::Real, q::Real) =
    efficiency(hvac, T_oa, part_load_ratio(hvac, q))

"""
    clamp_load(hvac, q) -> q_clamped

Clamp a requested load to the unit's maximum capacity: `-q_max ≤ q ≤ q_max`.
"""
function clamp_load(hvac::HVACSystem, q::Real)
    return clamp(q, -hvac.q_max, hvac.q_max)
end

# ---------------------------------------------------------------------------
# Electrical power and energy
# ---------------------------------------------------------------------------

"""
    electrical_power(hvac, T_oa, q) -> kW

Electrical input power (kW) required to deliver thermal load `q` (tons) at
ambient temperature `T_oa` (°F): `|q| · KW_PER_TON / COP`.
"""
function electrical_power(hvac::HVACSystem, T_oa::Real, q::Real)
    iszero(q) && return 0.0
    return abs(q) * KW_PER_TON / efficiency_at_load(hvac, T_oa, q)
end

function _trap_weight(times, k)
    n = length(times)
    n > 1 || throw(ArgumentError("at least two times are required for integration"))
    if k == 1
        return (times[2] - times[1]) / 2
    elseif k == n
        return (times[n] - times[n - 1]) / 2
    else
        return (times[k + 1] - times[k - 1]) / 2
    end
end

"""
    estimate_energy(hvac, T_oa, times, q_hvac) -> NamedTuple

Estimate electrical energy consumption (kWh) from a load trajectory.

`q_hvac` is an `n_zones × n_t` matrix of delivered loads (tons) sampled at
`times` (hours); `T_oa` is a callable giving ambient temperature (°F) at a
time in the same units.  `hvac` is a single [`HVACSystem`](@ref) applied to all
zones, or a vector/tuple with one unit per zone.  Power is integrated with the
trapezoidal rule over `times`.

Returns `(; per_zone, total, power, times)` where `per_zone` are per-zone kWh,
`total` is their sum, and `power` is the `n_zones × n_t` kW trajectory.
"""
function estimate_energy(hvac, T_oa, times::AbstractVector, q_hvac::AbstractMatrix)
    n_zones, n_t = size(q_hvac)
    length(times) == n_t || throw(DimensionMismatch(
        "q_hvac has $n_t columns but $(length(times)) times were given"))

    hvac_list = hvac isa HVACSystem ? fill(hvac, n_zones) : collect(hvac)
    length(hvac_list) == n_zones || throw(DimensionMismatch(
        "one HVAC system per zone is required ($n_zones), got $(length(hvac_list))"))

    power = zeros(n_zones, n_t)
    for i in 1:n_zones, k in 1:n_t
        power[i, k] = electrical_power(hvac_list[i], T_oa(times[k]), q_hvac[i, k])
    end

    per_zone = zeros(n_zones)
    for i in 1:n_zones, k in 1:n_t
        per_zone[i] += power[i, k] * _trap_weight(times, k)
    end

    return (; per_zone, total = sum(per_zone), power, times)
end

export HVACSystem, KW_PER_TON, part_load_ratio, efficiency, efficiency_at_load,
       clamp_load, electrical_power, estimate_energy

end
