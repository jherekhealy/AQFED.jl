using AQFED, Test, Random
import AQFED.GlobalOptimization as GO
import RandomNumbers.MersenneTwisters: MT19937

@testset "DE" begin
    sphere(x) = sum(abs2, x .- 0.3)
    low, high = fill(-1.0, 3), fill(1.0, 3)
    termination = GO.TerminationCriteria(300, 100, 1e-12, 1e-6)
    @testset "isSameAsPrevious" begin
        @test !GO.isSameAsPrevious([1, 2, 3, 4], 1)
        @test GO.isSameAsPrevious([1, 1, 3, 4], 2)
        @test GO.isSameAsPrevious([1, 2, 2, 4], 3)
        @test GO.isSameAsPrevious([1, 2, 3, 1], 4)
        @test !GO.isSameAsPrevious([1, 2, 3, 4], 4)
    end
    @testset "$(nameof(typeof(s)))" for s in (GO.Best1Bin(0.9, 0.5), GO.Rand1Bin(0.9, 0.5))
        o = GO.makeDifferentialEvolutionOptimizer(GO.OptimizerParams(10), GO.Problem(3, sphere, low, high), MT19937(1234), s)
        GO.optimize(o, termination)
        @test GO.minimum(o) < 1e-10
    end
    @testset "quasi random initial population" begin
        o = GO.makeDifferentialEvolutionOptimizer(GO.OptimizerParams(10), GO.Problem(3, sphere, low, high), MT19937(1234), GO.Best1Bin(0.9, 0.5), useQuasiInit=true)
        @test all(low[1] .<= o.g0 .<= high[1])
        GO.optimize(o, termination)
        @test GO.minimum(o) < 1e-10
    end
    @testset "NaN in the initial population" begin
        first = Ref(true)
        f(x) = first[] ? (first[] = false; NaN) : sphere(x)
        o = GO.makeDifferentialEvolutionOptimizer(GO.OptimizerParams(10), GO.Problem(3, f, low, high), MT19937(1234), GO.Best1Bin(0.9, 0.5))
        @test isfinite(GO.minimum(o))
        GO.optimize(o, termination)
        @test GO.minimum(o) < 1e-10
    end
    @testset "stagnation" begin
        #small steps add up to a large relative improvement: not a stagnation
        criteria = GO.TerminationCriteria(10_000, 10, 0.0, 1e-3)
        stat = GO.TerminationStatistics(0, 0, false, false, floatmax(Float64))
        cost = 1.0
        for _ in 1:1000
            GO.recordGeneration!(stat, criteria, cost)
            cost *= 1 - 1e-2
        end
        @test !stat.IsStagnating
        for _ in 1:12 #the first one still counts as an improvement
            GO.recordGeneration!(stat, criteria, cost)
        end
        @test stat.IsStagnating
    end
end
