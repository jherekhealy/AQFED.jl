# Tests of the book tooling: julia --project=book book/test_booktools.jl
using Test
include(joinpath(@__DIR__, "BookTools.jl"))
using .BookTools

@testset "parseIncludeWidths" begin
    tex = raw"""
    \subfloat[a]{\includegraphics[width=.49\textwidth]{figures/one.pdf}}
    \subfloat[b]{
        \includegraphics[width=.49\textwidth]{
            two.pdf
        }
    }
    \includegraphics[width=\textwidth]{three}
    \includegraphics[width=0.5\linewidth]{three.pdf}
    \includegraphics[width=10cm]{four.pdf}
    \includegraphics[height=5cm]{five.pdf}
    %\includegraphics[width=0.1\textwidth]{six.pdf}
    \includegraphics[width=0.3\textwidth]{seven.pdf} % \includegraphics[width=0.1\textwidth]{seven.pdf}
    """
    w = BookTools.parseIncludeWidths!(Dict{String,Float64}(), tex, 400.0, sizes=Dict("five" => (300.0, 150.0)))
    @test w["one"] ≈ 196
    @test w["two"] ≈ 196 #file name over several lines
    @test w["three"] ≈ 200 #smallest of two inclusions, with or without the extension
    @test w["four"] ≈ 10 * 72.27 / 2.54
    @test w["five"] ≈ 5 * 72.27 / 2.54 * 2 #height scaled by the reference aspect ratio
    @test !haskey(w, "six") #commented out
    @test w["seven"] ≈ 120 #the commented inclusion on the same line is ignored
end
