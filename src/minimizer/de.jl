using Random
import RandomNumbers: AbstractRNG

#metric instead of terminationcriteria.
#DEStrategy instead of Strategy? if same package for multiple optim, otherwise 
#Optim.optimize method. 

abstract type Strategy end


#Best1Bin The classic strategy with binary crossover.
struct Best1Bin <: Strategy
	F::Float64
	CR::Float64
end


function apply(s::Best1Bin, dim::Int, x::AbstractArray{T}, genBest::AbstractArray{T}, g0::AbstractMatrix{T}, rng) where {T}
	i = rand(rng, 1:dim)
	for counter ∈ 1:dim
		if (rand(rng, T) < s.CR) || (counter == dim)
			x[i] = genBest[i] + s.F * (g0[1, i] - g0[2, i])
		end
		i += 1
		i = ((i - 1) % dim) + 1
	end
end

function getRandomDimension(s::Best1Bin)
	return 2
end

#Best1Bin The classic strategy with binary crossover.
struct Rand1EitherOr <: Strategy
	F::Float64
	CR::Float64
end



function apply(s::Rand1EitherOr, dim::Int, x::AbstractArray{T}, genBest::AbstractArray{T}, g0::AbstractMatrix{T}, rng) where {T}
	if rand(rng, T) < s.CR
		@. x = g0[1, :] + s.F * (g0[2, :] - g0[3, :])
	else
		@. x = g0[1, :] + 0.5 * (f + 1) * (g0[2, :] - g0[3, :] - 2 * g0[1, :])
	end
end


function getRandomDimension(s::Rand1EitherOr)
	return 3
end

struct Best2Bin <: Strategy
	F::Float64
	CR::Float64
end
function apply(s::Best2Bin,  dim::Int, x::AbstractArray{T}, genBest::AbstractArray{T}, g0::AbstractMatrix{T}, rng) where {T}
	i = rand(rng, 1:dim)
	for counter ∈ 1:dim
		if (rand(rng, T) < s.CR) || (counter == dim)
			x[i] = genBest[i] + s.F * (g0[1, i] + g0[2, i] - g0[3, i] - g0[4, i])
		end
		i += 1
		i = ((i - 1) % dim) + 1
	end
end

function getRandomDimension(s::Best2Bin)
	return 4
end


struct Rand1Bin <: Strategy
	F::Float64
	CR::Float64
end
function apply(s::Rand1Bin,  dim::Int, x::AbstractArray{T}, genBest::AbstractArray{T}, g0::AbstractMatrix{T}, rng) where {T}
	i = rand(rng, 1:dim)
	for counter ∈ 1:dim
		if (rand(rng, T) < s.CR) || (counter == dim)
			@. x = g0[1, :] + s.F * (g0[2, :] - g0[3, :])
		end
		i += 1
		i = ((i - 1) % dim) + 1
	end
end


function getRandomDimension(s::Rand1Bin)
	return 3
end

#*Rand1Exp * Perhaps the most universally applicaple strategy, but ** not always the
# * fastest one. Still ** this is one of my favourite strategies. It works espe-
# * ** cially well when the "Best"-schemes experience mis- ** convergence. Try
# * e.g. F=0.7 and CR=0.5 as a first ** guess. ** ** Authors: Mikal Keenan **
# * Rainer Storn */
struct Rand1Exp
	F::Float64
	CR::Float64
end

function apply(s::Rand1Exp, dim::Int, x::AbstractArray{T}, genBest::AbstractArray{T}, g0::AbstractMatrix{T}, rng) where {T}
	i = rand(rng, 1:dim)
	for _ = 1:dim
		@. x = g0[1, :] + s.F * (g0[2, :] - g0[3, :])
		i += 1
		i = ((i - 1) % dim) + 1
		if rand(rng, T) >= s.CR
			break
		end
	end
end

function getRandomDimension(s::Rand1Exp)
	return 3
end

struct TerminationCriteria{T}
	Max::Int
	StagnationMax::Int
	AbsoluteTolerance::T
	StagnationTolerance::T
end

mutable struct TerminationStatistics{T}
	Count::Int
	StagnationCount::Int
	IsStagnating::Bool
	IsUnderTolerance::Bool
	MinCostPreviousGeneration::T
end

function recordGeneration!(t::TerminationStatistics{T}, c::TerminationCriteria{T}, mincost::T) where {T <: Number}
	t.Count += 1
	if c.AbsoluteTolerance > 0
		if abs(mincost) < c.AbsoluteTolerance
			t.IsUnderTolerance = true
		end
	end
	if !t.IsUnderTolerance && c.StagnationTolerance > 0
		if abs(t.MinCostPreviousGeneration - mincost) < c.StagnationTolerance
			t.StagnationCount += 1
			if t.StagnationCount > c.StagnationMax
				t.IsStagnating = true
			end
		else
			t.StagnationCount = 0
		end
		t.MinCostPreviousGeneration = mincost
	end
end

function isReached(t::TerminationStatistics{T}, c::TerminationCriteria{T})::Bool where {T}
	t.Count >= c.Max || t.IsUnderTolerance || t.IsStagnating
end

function reset!(t::TerminationStatistics{T}) where {T <: Number}
	t.Count = 0
	t.StagnationCount = 0
	t.IsStagnating = false
	t.IsUnderTolerance = false
	t.MinCostPreviousGeneration = floatmax(T)
end

struct Problem{T}
	Dimension::Int
	ObjectiveFunction::Function
	LowerBound::Vector{T}
	UpperBound::Vector{T}
end

struct OptimizerParams
	NP::Int     #population size
end

mutable struct Optimizer{T}
	params::OptimizerParams
	problem::Problem{T}
	rng::AbstractRNG
	strategy::Strategy

	#modified variables:
	generation::Int
	evaluation::Int
	mincost::T
	trial::Vector{T} # the trial vector
	best::Vector{T} #the best vector so far
	genbest::Vector{T} #the best vector of the current generation
	cost::Vector{T}
	p1::Matrix{T} # array of vectors
	p2::Matrix{T}
	rvec::Matrix{T} # array of randomly chosen vectors
	rnd::Vector{Int} # array of random indices
	g0::Matrix{T} # just some pointers (placeholders) for the generation at t and t+1
	g1::Matrix{T}
	minIndex::Int
end

# MakeDifferentialEvolutionOptimizer creates a new instance of differential evolution optimizer for a given problem
function makeDifferentialEvolutionOptimizer(params::OptimizerParams, problem::Problem{T}, rng::AbstractRNG, strategy::Strategy; useQuasiInit::Bool = false, x0::Vector{T}=T[]) where {T <: Number}
	generation = 0
	evaluation = 0

	dim = problem.Dimension # get size of the problem
	maxR = getRandomDimension(strategy)
	trial = zeros(T, dim) #the trial vector
	best = zeros(T, dim) # the best vector so far
	genbest = zeros(T, dim) #  the best vector of the current generation
	cost = zeros(T, params.NP)
	p1 = zeros(T, params.NP, dim) # array of vectors
	p2 = zeros(T, params.NP, dim)
	rvec = zeros(T, maxR, dim) # array of randomly chosen vectors
	rnd = zeros(Int, maxR) # array of random indices
	r = zeros(T, dim)
	iStart = if length(x0) == dim 
		2
	else 
		1
	end
	if iStart == 2
		p1[1,:] .= x0
		cost[1] = problem.ObjectiveFunction(p1[1, :])
			evaluation += 1
	end
	if useQuasiInit
		qrng = ScramblingSobol(params.dim, params.N, NoScrambling)
		for i ∈ iStart:params.NP
			next!(qrng, r)
			@. p1[i, :] = problem.LowerBound + (problem.UpperBound - problem.LowerBound) * r
			cost[i] = problem.ObjectiveFunction(p1[i, :])
			evaluation += 1
		end
	else
		for i ∈ iStart:params.NP
			rand!(rng, r)
			@. p1[i, :] = problem.LowerBound + (problem.UpperBound - problem.LowerBound) * r
			cost[i] = problem.ObjectiveFunction(p1[i, :])
			evaluation += 1
		end
	end
	
	mincost = cost[1]
	minIndex = 1
	for j ∈ 1:params.NP
		x = cost[j]
		if x < mincost
			mincost = x
			minIndex = j
		end
	end

	best .= p1[minIndex, :]
	genbest .= best

	g0 = p1 # generation t
	g1 = p2 # generation t+1
	return Optimizer(params, problem, rng, strategy,
		generation,
		evaluation,
		mincost,
		trial,
		best,
		genbest,
		cost,
		p1,
		p2,
		rvec,
		rnd,
		g0,
		g1,
		minIndex)
end

function optimize(o::Optimizer{T}, termination::TerminationCriteria{T}; MAX_SAME = 10000) where {T}
	stat = TerminationStatistics(0, 0, false, false, floatmax(T))
	while !isReached(stat, termination)
		for i ∈ 1:o.params.NP
			o.trial .= o.g0[i, :] # trial vector
			for j ∈ eachindex(o.rnd)
				sameCount = 0
				for o.rnd[j] ∈ rand(o.rng, 1:o.params.NP)
					isSame = isSameAsPrevious(o.rnd, j)
					sameCount += 1
					if sameCount == MAX_SAME
						# safeguard
						throw("Error in initial parameters, could not compute distinct random numbers")
					end
					if !isSame && o.rnd[j] != i
						break
					end
				end
			end

			for k ∈ eachindex(o.rnd) # select the random vectors
				o.rvec[k, :] = @view(o.g0[o.rnd[k], :])
			end
			#---Apply the DE strategy of choice------------------
			apply(o.strategy, o.problem.Dimension, o.trial, o.genbest, o.rvec, o.rng)
			#Apply boundary constraint (bounce back)
			bounceBack(@view(o.g0[i, :]), o.trial, o.problem.LowerBound, o.problem.UpperBound, o.rng)
			#---cost of trial vector------------------------
			testcost = o.problem.ObjectiveFunction(o.trial)
			o.evaluation += 1

			if !isnan(testcost) && testcost < o.cost[i] # Better solution than target vectors cost
				o.g1[i, :] .= o.trial                            # if yes put trial vector in new population
				o.cost[i] = testcost                               # and save the new cost value
				if !isnan(testcost) && testcost < o.mincost  # if testcost is best ever
					o.mincost = testcost  # new mincost
					o.best .= o.trial # best vector is trial vector
					o.minIndex = i     # save index of best vector
				end
			else # if trial vector is worse than target vector
				o.g1[i, :] .= o.g0[i, :]
			end
		end
		# ("current best ", o.best, o.mincost)
		o.genbest .= o.best # Save current generation's best

		o.g0, o.g1 = o.g1, o.g0 # Swap population pointers

		o.generation += 1
		recordGeneration!(stat, termination, o.mincost)
	end
	return o.mincost, stat
end

function minimum(o::Optimizer)
	o.mincost
end

function minimizer(o::Optimizer)
	return o.best
end

function currentGeneration(o::Optimizer)
	return o.generation
end


function numberOfEvaluations(o::Optimizer)
	return o.evaluation
end

function isSameAsPrevious(rnd::AbstractArray{Int}, j::Int)
	if j == 1
		return false
	elseif j == 2
		return rnd[1] == rnd[2]
	elseif j == 3
		return (rnd[3] == rnd[1]) || (rnd[3] == rnd[1])
	else
		for k ∈ 1:j-1
			if rnd[j] == rnd[k]
				return true
			end
		end
		return false
	end
end

function bounceBack(base::AbstractArray{T}, x::AbstractArray{T}, l::AbstractArray{T}, u::AbstractArray{T}, rng) where {T}
	for j ∈ eachindex(l)
		if x[j] < l[j]
			x[j] = base[j] + rand(rng, T) * (l[j] - base[j])
		end
		if x[j] > u[j]
			x[j] = base[j] + rand(rng, T) * (u[j] - base[j])
		end
	end
end
