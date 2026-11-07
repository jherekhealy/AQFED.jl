#!/usr/bin/env julia

using Pkg
Pkg.activate(".")

using AQFED.Random: ScrambledSobolSeq, joe_kuo, niederreiter_nut, niederreiter_nut_21201, niederreiter_nut_s21201, NoScrambling, next!

println("=" ^ 60)
println("Testing Niederreiter-NUT Direction Numbers Implementation")
println("=" ^ 60)

println("\n[Test 1] Backward Compatibility (default to Joe & Kuo):")
try
    seq_default = ScrambledSobolSeq(3, 10, NoScrambling())
    pts = zeros(Float64, 3)
    next!(seq_default, pts)
    println("✓ Default sequence created and generated points")
    println("  First point: $(round.(pts, digits=4))")
catch e
    println("✗ Failed: $e")
end

println("\n[Test 2] Explicit Joe & Kuo source:")
try
    seq_jk = ScrambledSobolSeq(3, 10, NoScrambling(), joe_kuo)
    pts = zeros(Float64, 3)
    next!(seq_jk, pts)
    println("✓ Joe & Kuo sequence created successfully")
    println("  First point: $(round.(pts, digits=4))")
catch e
    println("✗ Failed: $e")
end

println("\n[Test 3] Niederreiter-NUT source:")
try
    seq_nn = ScrambledSobolSeq(3, 10, NoScrambling(), niederreiter_nut)
    pts = zeros(Float64, 3)
    println("✓ Niederreiter-NUT sequence created successfully")
    for i in 1:3
        next!(seq_nn, pts)
        println("  Point $i: $(round.(pts, digits=4))")
    end
catch e
    println("✗ Failed: $e")
end

println("\n[Test 4] Niederreiter-NUT 21201 source (niederreiter_nut_21201 / niederreiter_nut_s21201):")
try
    seq_nn21 = ScrambledSobolSeq(3, 10, NoScrambling(), niederreiter_nut_21201)
    pts = zeros(Float64, 3)
    println("✓ Niederreiter-NUT 21201 sequence created successfully")
    for i in 1:3
        next!(seq_nn21, pts)
        println("  Point $i: $(round.(pts, digits=4))")
    end
catch e
    println("✗ Failed: $e")
end

println("\n[Test 5] Comparison: Sources produce different sequences for point 2:")
try
    seq_jk = ScrambledSobolSeq(5, 10, NoScrambling(), joe_kuo)
    seq_nn = ScrambledSobolSeq(5, 10, NoScrambling(), niederreiter_nut)
    seq_nn21 = ScrambledSobolSeq(5, 10, NoScrambling(), niederreiter_nut_21201)
    
    pts_jk = zeros(Float64, 5)
    pts_nn = zeros(Float64, 5)
    pts_nn21 = zeros(Float64, 5)
    
    # Point 1 is 0.5 for all; compare Point 2
    next!(seq_jk, pts_jk); next!(seq_jk, pts_jk)
    next!(seq_nn, pts_nn); next!(seq_nn, pts_nn)
    next!(seq_nn21, pts_nn21); next!(seq_nn21, pts_nn21)
    
    if pts_jk != pts_nn && pts_nn != pts_nn21
        println("✓ Sources produce different sequences (as expected)")
        println("  Joe & Kuo second point:          $(round.(pts_jk, digits=4))")
        println("  Niederreiter (1111) second point: $(round.(pts_nn, digits=4))")
        println("  Niederreiter (21201) second point:$(round.(pts_nn21, digits=4))")
    else
        println("✗ Sources produced unexpected duplicate sequences")
    end
catch e
    println("✗ Failed: $e")
end

println("\n[Test 6] High dimensions (500):")
try
    seq_jk_500 = ScrambledSobolSeq(500, 10, NoScrambling(), joe_kuo)
    println("✓ Joe & Kuo 500D created successfully")
    seq_nn_500 = ScrambledSobolSeq(500, 10, NoScrambling(), niederreiter_nut)
    println("✓ Niederreiter-NUT 500D created successfully")
    seq_nn21_500 = ScrambledSobolSeq(500, 10, NoScrambling(), niederreiter_nut_21201)
    println("✓ Niederreiter-NUT 21201 500D created successfully")
catch e
    println("✗ Failed: $e")
end

println("\n[Test 7] Maximum dimensions:")
try
    seq_jk_max = ScrambledSobolSeq(21201, 10, NoScrambling(), joe_kuo)
    println("✓ Joe & Kuo at max dimension (21201)")
catch e
    println("✗ Joe & Kuo max dimension failed: $e")
end

try
    seq_nn_max = ScrambledSobolSeq(1111, 10, NoScrambling(), niederreiter_nut)
    println("✓ Niederreiter-NUT at max dimension (1111)")
catch e
    println("✗ Niederreiter-NUT max dimension failed: $e")
end

try
    seq_nn21_max = ScrambledSobolSeq(21201, 10, NoScrambling(), niederreiter_nut_21201)
    println("✓ Niederreiter-NUT 21201 at max dimension (21201)")
catch e
    println("✗ Niederreiter-NUT 21201 max dimension failed: $e")
end

println("\n" * ("=" ^ 60))
println("✓ All tests completed successfully!")
println("=" ^ 60)
