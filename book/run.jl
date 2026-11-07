# Renders the book figures, or runs the computations that produce their data.
#   julia --project=book book/run.jl                      # all figures
#   julia --project=book book/run.jl 'heston_qe'          # figures whose name matches the regex
#   julia --project=book book/run.jl --compute 'jaeckel'  # computations whose name matches, then nothing else
#   julia --project=book book/run.jl --list               # names of the figures and computations
# Figures are written to $EQD_BOOK_DIR (default ~/mypapers/eqd_book/figures, see \graphicspath in eqd_book.tex).
include(joinpath(@__DIR__, "BookTools.jl"))
using .BookTools
using CairoMakie, DataFrames, CSV, DelimitedFiles

for f in sort(readdir(@__DIR__))
    occursin(r"^ch\d\d_.*\.jl$", f) && include(joinpath(@__DIR__, f))
end

function main(args)
    if "--list" in args
        println("figures:\n  ", join(sort(collect(keys(FIGURES))), "\n  "))
        println("computations:\n  ", join(sort(collect(keys(COMPUTATIONS))), "\n  "))
        return
    end
    patterns = filter(a -> !startswith(a, "--"), args)
    pattern = isempty(patterns) ? r"" : Regex(first(patterns))
    if "--compute" in args
        runComputations(pattern)
    else
        for path in renderFigures(pattern)
            println(path)
        end
    end
end

abspath(PROGRAM_FILE) == (@__FILE__) && main(ARGS)
