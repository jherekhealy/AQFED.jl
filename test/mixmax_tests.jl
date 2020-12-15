using AQFED, Test

import AQFED.Random


function testSkip(N::Int)
   myrange = [1535, 1536, 2000, 2046, 2047, 2048, 2049, 2050, 100000, 100001]
   for i in myrange
      gen1 = AQFED.Random.Mixmax17()
      n = UInt64(i)
      AQFED.Random.skip(gen1, n)
      u1 = rand(gen1, UInt64)
      gen2 = AQFED.Random.Mixmax17()
      for j = 1:n
         rand(gen2, UInt64)
      end
      u2 = rand(gen2, UInt64)
      if u1 != u2
         println(i, " ", u1, " ", gen1, " ", rand(gen1, UInt64))
         println(i, " ", u2, " ", gen2, " ", rand(gen2, UInt64))
      end
      @test isequal(u1, u2)
   end
end

@testset "Mixmax17Skip" begin
testSkip(17)
end

@testset "Mixmax240Skip" begin
testSkip(240)
end


@testset "Mixmax17SkipTwice" begin
   N = 17
   myrange = [1535, 1536, 2000, 2046, 2047, 2048, 2049, 2050, 100000, 100001]
   gen1 = AQFED.Random.Mixmax17()
   gen2 = AQFED.Random.Mixmax17()
   for i in myrange
      n = UInt64(i)
      AQFED.Random.skip(gen1, n)
      u1 = rand(gen1, UInt64)
      for j = 1:n
         rand(gen2, UInt64)
      end
      u2 = rand(gen2, UInt64)
      if u1 != u2
         println(i, " ", u1, " ", gen1, " ", rand(gen1, UInt64))
         println(i, " ", u2, " ", gen2, " ", rand(gen2, UInt64))
      end
      @test isequal(u1, u2)
   end
end

@testset "Mixmax17Skip20" begin
   gen = AQFED.Random.Mixmax17()
   N = 17
   a0 = Vector(undef, N)
   AQFED.Random.iterateRawVec(gen)
   a0 = gen.v
   eArray = UInt64[
      843356255388265439,
      2303128525550447885,
      1403805614380726589,
      1646775209363665369,
      1656489684429545237,
      136962756483849705,
      578486145972684395,
      1186788852536058806,
      744606065061259291,
      129235498787499648,
      987271125576440333,
      2226012411473073226,
      1064839699140268466,
      1194475380559801816,
      1591779369875059359,
      1167968540715842034,
      2137709594798027207,
   ]
   a2p20 = zeros(UInt64, N)
   for i = 1:N
      for j = 1:N
         a2p20[j] = AQFED.Random.fmodmulM61(a2p20[j], eArray[i], a0[j])
      end
      AQFED.Random.iterateRawVec(gen)
      a0 = gen.v
   end
   println("a2p20 ", a2p20)
   gen = AQFED.Random.Mixmax17()
   for i = 1:(1+2^20)
      #skips 2^20 * 17 points
      AQFED.Random.iterateRawVec(gen)
      a0 = gen.v
      #end
   end
   for i = 1:N
      @test isequal(a0[i], a2p20[i])
   end
end
