using DelimitedFiles

# File paths for datasets
const sobol_a_file = joinpath(@__DIR__, "sobol_a.csv")
const sobol_minit_file = joinpath(@__DIR__, "sobol_minit.csv")
const niederreiter_a_file = joinpath(@__DIR__, "niederreiter_a.csv")
const niederreiter_minit_file = joinpath(@__DIR__, "niederreiter_minit.csv")
const niederreiter_s_file = joinpath(@__DIR__, "niederreiter_s.csv")
const niederreiter_a_21201_file = joinpath(@__DIR__, "niederreiter_a_21201.csv")
const niederreiter_minit_21201_file = joinpath(@__DIR__, "niederreiter_minit_21201.csv")
const niederreiter_s_21201_file = joinpath(@__DIR__, "niederreiter_s_21201.csv")

# we need to re-compile if these files change:
include_dependency(sobol_a_file)
include_dependency(sobol_minit_file)
include_dependency(niederreiter_a_file)
include_dependency(niederreiter_minit_file)
include_dependency(niederreiter_s_file)
include_dependency(niederreiter_a_21201_file)
include_dependency(niederreiter_minit_21201_file)
include_dependency(niederreiter_s_21201_file)

# Enumeration for direction number source
@enum DirectionSource joe_kuo=1 niederreiter_nut=2 niederreiter_nut_21201=3
const niederreiter_nut_s21201 = niederreiter_nut_21201

# Pre-loaded constants for datasets to avoid disk re-reading
const sobol_a = vec(readdlm(sobol_a_file, ',', UInt32))
const sobol_minit = readdlm(sobol_minit_file, ',', UInt32)

const niederreiter_a = vec(readdlm(niederreiter_a_file, ',', UInt32))
const niederreiter_minit = Matrix(transpose(readdlm(niederreiter_minit_file, ',', UInt32)))
const niederreiter_s = vec(readdlm(niederreiter_s_file, ',', UInt32))

const niederreiter_a_21201 = vec(readdlm(niederreiter_a_21201_file, ',', UInt32))
const niederreiter_minit_21201 = Matrix(transpose(readdlm(niederreiter_minit_21201_file, ',', UInt32)))
const niederreiter_s_21201 = vec(readdlm(niederreiter_s_21201_file, ',', UInt32))

"""
    load_direction_numbers(source::DirectionSource) -> (a, minit, s)

Load polynomial coefficients, initial direction numbers, and polynomial degrees for Sobol sequences.

# Arguments
- `source::DirectionSource`: `joe_kuo` (default), `niederreiter_nut`, or `niederreiter_nut_21201` (`niederreiter_nut_s21201`)

# Returns
- `a`: Vector of polynomial coefficients
- `minit`: Matrix of initial direction numbers (rows=m values, cols=dimensions)
- `s`: Vector of polynomial degrees (s = degree of primitive polynomial)
"""
function load_direction_numbers(source::DirectionSource)
    if source == joe_kuo
        return sobol_a, sobol_minit, nothing
    elseif source == niederreiter_nut
        return niederreiter_a, niederreiter_minit, niederreiter_s
    elseif source == niederreiter_nut_21201
        return niederreiter_a_21201, niederreiter_minit_21201, niederreiter_s_21201
    else
        error("Unknown DirectionSource: $source")
    end
end
