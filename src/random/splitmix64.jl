@inline function splitmix64_next(x::UInt64)
    x += 0x9E3779B97F4A7C15
    z = x
    z = (z ⊻ (z >> 30)) * 0xBF58476D1CE4E5B9
    z = (z ⊻ (z >> 27)) * 0x94D049BB133111EB
    z = z ⊻ (z >> 31)
    return x, z
end

