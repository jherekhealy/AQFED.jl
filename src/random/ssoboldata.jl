using DelimitedFiles

const sobol_a_file = joinpath(@__DIR__, "sobol_a.csv")
const sobol_minit_file = joinpath(@__DIR__, "sobol_minit.csv")

# we need to re-compile if these files change:
include_dependency(sobol_a_file)
include_dependency(sobol_minit_file)

#successive primitive binary-coefficient polynomials p(z)
#   = a_0 + a_1 z + a_2 z^2 + ... a_31 z^31, where a_i is the
#     i-th bit of sobol_a[j] for the j-th polynomial.
const sobol_a = vec(readdlm(sobol_a_file, ',', UInt32))

# starting direction #'s m[i] = sobol_minit[i][j] for i=0..d of the
# degree-d primitive polynomial sobol_a[j].
const sobol_minit = readdlm(sobol_minit_file, ',', UInt32)
