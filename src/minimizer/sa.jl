#=
 !  This file is an example of the Corana et al. simulated annealing algorithm
 !  for multimodal and robust optimization as implemented and modified by
 !  Goffe, Ferrier and Rogers.  Counting the above line
 !  ABSTRACT as 1, the routine itself (SA), with its supplementary
 !  routines, is on lines 232-990. A multimodal example from Judge et al.
 !  (FCN) is on lines 150-231. The rest of this file (lines 1-149) is a
 !  driver routine with values appropriate for the Judge example.  Thus, this
 !  example is ready to run.

 !  To understand the algorithm, the documentation for SA on lines 236-
 !  484 should be read along with the parts of the paper that describe
 !  simulated annealing. Then the following lines will then aid the user
 !  in becomming proficient with this implementation of simulated annealing.

 !  Learning to use SA:
 !      Use the sample function from Judge with the following suggestions
 !  to get a feel for how SA works. When you've done this, you should be
 !  ready to use it on most any function with a fair amount of expertise.
 !    1. Run the program as is to make sure it runs okay. Take a look at
 !       the intermediate output and see how it optimizes as temperature
 !       (T) falls.  Notice how the optimal point is reached and how
 !       falling T reduces VM.
 !    2. Look through the documentation to SA so the following makes a
 !       bit of sense. In line with the paper, it shouldn't be that hard
 !       to figure out. The core of the algorithm is described on pp. 68-70
 !       and on pp. 94-95. Also see Corana et al. pp. 264-9.
 !    3. To see how it selects points and makes decisions about uphill and
 !       downhill moves, set IPRINT = 3 (very detailed intermediate output)
 !       and MAXEVL = 100 (only 100 function evaluations to limit output).
 !    4. To see the importance of different temperatures, try starting
 !       with a very low one (say T = 10E-5).  You'll see (i) it never
 !       escapes from the local optima (in annealing terminology, it
 !       quenches) & (ii) the step length (VM) will be quite small.  This
 !       is a key part of the algorithm: as temperature (T) falls, step
 !       length does too. In a minor point here, note how VM is quickly
 !       reset from its initial value. Thus, the input VM is not very
 !       important.  This is all the more reason to examine VM once the
 !       algorithm is underway.
 !    5. To see the effect of different parameters and their effect on
 !       the speed of the algorithm, try RT = .95 & RT = .1.  Notice the
 !       vastly different speed for optimization. Also try NT = 20.  Note
 !       that this sample function is quite easy to optimize, so it will
 !       tolerate big changes in these parameters.  RT and NT are the
 !       parameters one should adjust to modify the runtime of the
 !       algorithm and its robustness.
 !    6. Try constraining the algorithm with either LB or UB.
 ABSTRACT:
 !   Simulated annealing is a global optimization method that distinguishes
 !   between different local optima.  Starting from an initial point, the
 !   algorithm takes a step and the function is evaluated. When minimizing a
 !   function, any downhill step is accepted and the process repeats from this
 !   new point. An uphill step may be accepted. Thus, it can escape from local
 !   optima. This uphill decision is made by the Metropolis criteria. As the
 !   optimization process proceeds, the length of the steps decline and the
 !   algorithm closes in on the global optimum. Since the algorithm makes very
 !   few assumptions regarding the function to be optimized, it is quite
 !   robust with respect to non-quadratic surfaces. The degree of robustness
 !   can be adjusted by the user. In fact, simulated annealing can be used as
 !   a local optimizer for difficult functions.

 !   This implementation of simulated annealing was used in "Global Optimizatio
 !   of Statistical Functions with Simulated Annealing," Goffe, Ferrier and
 !   Rogers, Journal of Econometrics, vol. 60, no. 1/2, Jan./Feb. 1994, pp.
 !   65-100. Briefly, we found it competitive, if not superior, to multiple
 !   restarts of conventional optimization routines for difficult optimization
 !   problems.

 !   For more information on this routine, contact its author:
 !   Bill Goffe, bgoffe@whale.st.usm.edu

 ! This version in Fortran 90 has been prepared by Alan Miller.
 ! It is compatible with Lahey's ELF90 compiler.
 ! N.B. The 3 last arguments have been removed from subroutine sa.   these
 !      were work arrays and are now internal to the routine.
 ! e-mail:  amiller @ bigpond.net.au
 ! URL   :  http://users.bigpond.net.au/amiller
 Synopsis:
 !  This routine implements the continuous simulated annealing global
 !  optimization algorithm described in Corana et al.'s article "Minimizing
 !  Multimodal Functions of Continuous Variables with the "Simulated Annealing"
 !  Algorithm" in the September 1987 (vol. 13, no. 3, pp. 262-280) issue of
 !  the ACM Transactions on Mathematical Software.

 !  A very quick (perhaps too quick) overview of SA:
 !     SA tries to find the global optimum of an N dimensional function.
 !  It moves both up and downhill and as the optimization process
 !  proceeds, it focuses on the most promising area.
 !     To start, it randomly chooses a trial point within the step length
 !  VM (a vector of length N) of the user selected starting point. The
 !  function is evaluated at this trial point and its value is compared
 !  to its value at the initial point.
 !     In a maximization problem, all uphill moves are accepted and the
 !  algorithm continues from that trial point. Downhill moves may be
 !  accepted; the decision is made by the Metropolis criteria. It uses T
 !  (temperature) and the size of the downhill move in a probabilistic
 !  manner. The smaller T and the size of the downhill move are, the more
 !  likely that move will be accepted. If the trial is accepted, the
 !  algorithm moves on from that point. If it is rejected, another point
 !  is chosen instead for a trial evaluation.
 !     Each element of VM periodically adjusted so that half of all
 !  function evaluations in that direction are accepted.
 !     A fall in T is imposed upon the system with the RT variable by
 !  T(i+1) = RT*T(i) where i is the ith iteration. Thus, as T declines,
 !  downhill moves are less likely to be accepted and the percentage of
 !  rejections rise. Given the scheme for the selection for VM, VM falls.
 !  Thus, as T declines, VM falls and SA focuses upon the most promising
 !  area for optimization.

 !  The importance of the parameter T:
 !     The parameter T is crucial in using SA successfully. It influences
 !  VM, the step length over which the algorithm searches for optima. For
 !  a small intial T, the step length may be too small; thus not enough
 !  of the function might be evaluated to find the global optima. The user
 !  should carefully examine VM in the intermediate output (set IPRINT =
 !  1) to make sure that VM is appropriate. The relationship between the
 !  initial temperature and the resulting step length is function
 !  dependent.
 !     To determine the starting temperature that is consistent with
 !  optimizing a function, it is worthwhile to run a trial run first. Set
 !  RT = 1.5 and T = 1.0. With RT > 1.0, the temperature increases and VM
 !  rises as well. Then select the T that produces a large enough VM.

 !  For modifications to the algorithm and many details on its use,
 !  (particularly for econometric applications) see Goffe, Ferrier
 !  and Rogers, "Global Optimization of Statistical Functions with
 !  Simulated Annealing," Journal of Econometrics, vol. 60, no. 1/2,
 !  Jan./Feb. 1994, pp. 65-100.
 !  For more information, contact
 !              Bill Goffe
 !              Department of Economics and International Business
 !              University of Southern Mississippi
 !              Hattiesburg, MS  39506-5072
 !              (601) 266-4484 (office)
 !              (601) 266-4920 (fax)
 !              bgoffe@whale.st.usm.edu (Internet)

 !  As far as possible, the parameters here have the same name as in
 !  the description of the algorithm on pp. 266-8 of Corana et al.

 !  In this description, SP is single precision, DP is double precision,
 !  INT is integer, L is logical and (N) denotes an array of length n.
 !  Thus, DP(N) denotes a double precision array of length n.

 !  Input Parameters:
 !    Note: The suggested values generally come from Corana et al. To
 !          drastically reduce runtime, see Goffe et al., pp. 90-1 for
 !          suggestions on choosing the appropriate RT and NT.
 !    N - Number of variables in the function to be optimized. (INT)
 !    X - The starting values for the variables of the function to be
 !        optimized. (DP(N))
 !    MAX - Denotes whether the function should be maximized or minimized.
 !          A true value denotes maximization while a false value denotes
 !          minimization.  Intermediate output (see IPRINT) takes this into
 !          account. (L)
 !    RT - The temperature reduction factor.  The value suggested by
 !         Corana et al. is .85. See Goffe et al. for more advice. (DP)
 !    EPS - Error tolerance for termination. If the final function
 !          values from the last neps temperatures differ from the
 !          corresponding value at the current temperature by less than
 !          EPS and the final function value at the current temperature
 !          differs from the current optimal function value by less than
 !          EPS, execution terminates and IER = 0 is returned. (EP)
 !    NS - Number of cycles.  After NS*N function evaluations, each element of
 !         VM is adjusted so that approximately half of all function evaluations
 !         are accepted.  The suggested value is 20. (INT)
 !    NT - Number of iterations before temperature reduction. After
 !         NT*NS*N function evaluations, temperature (T) is changed
 !         by the factor RT.  Value suggested by Corana et al. is
 !         MAX(100, 5*N).  See Goffe et al. for further advice. (INT)
 !    NEPS - Number of final function values used to decide upon termi-
 !           nation.  See EPS.  Suggested value is 4. (INT)
 !    MAXEVL - The maximum number of function evaluations.  If it is
 !             exceeded, IER = 1. (INT)
 !    LB - The lower bound for the allowable solution variables. (DP(N))
 !    UB - The upper bound for the allowable solution variables. (DP(N))
 !         If the algorithm chooses X(I) .LT. LB(I) or X(I) .GT. UB(I),
 !         I = 1, N, a point is from inside is randomly selected. This
 !         This focuses the algorithm on the region inside UB and LB.
 !         Unless the user wishes to concentrate the search to a particular
 !         region, UB and LB should be set to very large positive
 !         and negative values, respectively.  Note that the starting
 !         vector X should be inside this region.  Also note that LB and
 !         UB are fixed in position, while VM is centered on the last
 !         accepted trial set of variables that optimizes the function.
 !    C - Vector that controls the step length adjustment.  The suggested
 !        value for all elements is 2.0. (DP(N))
 !    IPRINT - controls printing inside SA. (INT)
 !             Values: 0 - Nothing printed.
 !                     1 - Function value for the starting value and
 !                         summary results before each temperature
 !                         reduction. This includes the optimal
 !                         function value found so far, the total
 !                         number of moves (broken up into uphill,
 !                         downhill, accepted and rejected), the
 !                         number of out of bounds trials, the
 !                         number of new optima found at this
 !                         temperature, the current optimal X and
 !                         the step length VM. Note that there are
 !                         N*NS*NT function evalutations before each
 !                         temperature reduction. Finally, notice is
 !                         is also given upon achieveing the termination
 !                         criteria.
 !                     2 - Each new step length (VM), the current optimal
 !                         X (XOPT) and the current trial X (X). This
 !                         gives the user some idea about how far X
 !                         strays from XOPT as well as how VM is adapting
 !                         to the function.
 !                     3 - Each function evaluation, its acceptance or
 !                         rejection and new optima. For many problems,
 !                         this option will likely require a small tree
 !                         if hard copy is used. This option is best
 !                         used to learn about the algorithm. A small
 !                         value for MAXEVL is thus recommended when
 !                         using IPRINT = 3.
 !             Suggested value: 1
 !             Note: For a given value of IPRINT, the lower valued
 !                   options (other than 0) are utilized.
 !    ISEED1 - The first seed for the random number generator RANMAR.
 !             0 <= ISEED1 <= 31328. (INT)
 !    ISEED2 - The second seed for the random number generator RANMAR.
 !             0 <= ISEED2 <= 30081. Different values for ISEED1
 !             and ISEED2 will lead to an entirely different sequence
 !             of trial points and decisions on downhill moves (when
 !             maximizing).  See Goffe et al. on how this can be used
 !             to test the results of SA. (INT)

 !  Input/Output Parameters:
 !    T - On input, the initial temperature. See Goffe et al. for advice.
 !        On output, the final temperature. (DP)
 !    VM - The step length vector. On input it should encompass the region of
 !         interest given the starting value X.  For point X(I), the next
 !         trial point is selected is from X(I) - VM(I)  to  X(I) + VM(I).
 !         Since VM is adjusted so that about half of all points are accepted,
 !         the input value is not very important (i.e. is the value is off,
 !         SA adjusts VM to the correct value). (DP(N))

 !  Output Parameters:
 !    XOPT - The variables that optimize the function. (DP(N))
 !    FOPT - The optimal value of the function. (DP)
 !    NACC - The number of accepted function evaluations. (INT)
 !    NFCNEV - The total number of function evaluations. In a minor
 !             point, note that the first evaluation is not used in the
 !             core of the algorithm; it simply initializes the
 !             algorithm. (INT).
 !    NOBDS - The total number of trial function evaluations that
 !            would have been out of bounds of LB and UB. Note that
 !            a trial point is randomly selected between LB and UB. (INT)
 !    IER - The error return number. (INT)
 !          Values: 0 - Normal return; termination criteria achieved.
 !                  1 - Number of function evaluations (NFCNEV) is
 !                      greater than the maximum number (MAXEVL).
 !                  2 - The starting value (X) is not inside the
 !                      bounds (LB and UB).
 !                  3 - The initial temperature is not positive.
 !                  99 - Should not be seen; only used internally.

 !  Work arrays that must be dimensioned in the calling routine:
 !       RWK1 (DP(NEPS))  (FSTAR in SA)
 !       RWK2 (DP(N))     (XP    "  " )
 !       IWK  (INT(N))    (NACP  "  " )
 !  N.B. In the Fortran 90 version, these are automatic arrays.
=#
using Random
import RandomNumbers: AbstractRNG

struct SAStatusCode
	value::Int
	message::String
end

mutable struct SimulatedAnnealing{T} 
	problem::Problem{T}
	rng::AbstractRNG
	xopt::Vector{T}
	fopt::T
	nacc::Int
	nfcnev::Int
	nobds::Int
	err::SAStatusCode
end


#VMDefault(LowerBound::AbstractArray, UpperBound::AbstractArray) = UpperBound - LowerBound

function minimum(o::SimulatedAnnealing)
	o.fopt
end

function minimizer(o::SimulatedAnnealing)
	return o.xopt
end
function numberOfEvaluations(o::SimulatedAnnealing)
	return o.nfcnev
end

function SimulatedAnnealing(problem::Problem{T}, rng::AbstractRNG) where {T}
	return SimulatedAnnealing(problem, rng, zeros(T, problem.Dimension), typemax(T), 0, 0, 0, SAStatusCode(4, "Not Started"))
end

function optimize(
	o::SimulatedAnnealing{T},
	x::AbstractArray{T};
	rt = 0.25,
	eps = 1e-6,
	ns::Int = 3,
	nt::Int = 10,
	neps::Int = 4,
	maxevl::Int = 3000,
	t = 1.0,
	c = 2ones(T, o.problem.Dimension),
	vm = o.problem.UpperBound-o.problem.LowerBound,
	max = false,
) where {T}
	o.nacc = 0
	o.nobds = 0
	o.nfcnev = 0
	xopt = o.xopt
	xopt .= x
    n = o.problem.Dimension
	nacp = zeros(Int, n)
	fstar = ones(T, neps) * 1e20
	xp = zeros(T, n)
	o.fopt = typemax(T)

	# If the initial temperature is not positive, notify the user and
	#  return to the calling routine.
	if t <= 0.0
		o.err = SAStatusCode(3, "THE INITIAL TEMPERATURE IS NOT POSITIVE. reset the variable t")
		return o.fopt, o.err
	end

	#  If the initial value is out of bounds, notify the user and return
	#  to the calling routine.
	ub = o.problem.UpperBound
	lb = o.problem.LowerBound
	for i ∈ eachindex(ub)
		if (x[i] > ub[i]) || (x[i] < lb[i])
			o.err = SAStatusCode(2, "Initial value out of bounds")
			return o.fopt, o.err
		end
	end

	#  Evaluate the function with input X and return value as F.
	f = o.problem.ObjectiveFunction(x)
	if !max
		f = -f
	end

	o.nfcnev = o.nfcnev + 1
	o.fopt = f

	fstar[1] = f

	# Start the main loop. Note that it terminates if (i) the algorithm
	#  succesfully optimizes the function or (ii) there are too many
	#  function evaluations (more than MAXEVL).
	while true
		nup = 0
		nrej = 0
		nnew = 0
		ndown = 0
		lnobds = 0

		for m ∈ 1:nt
			for j ∈ 1:ns
				for h ∈ 1:n

					#  Generate XP, the trial value of X. Note use of VM to choose XP.
					for i ∈ 1:n
						if i == h
							xp[i] = x[i] + (rand(o.rng) * 2 - 1) * vm[i]
						else
							xp[i] = x[i]
						end
						#  If XP is out of bounds, select a point in bounds for the trial.
						if (xp[i] < lb[i]) || (xp[i] > ub[i])
							xp[i] = lb[i] + (ub[i] - lb[i]) * rand(o.rng)
							lnobds += 1
							o.nobds += 1
						end
					end

					# Evaluate the function with the trial point XP and return as FP.
					fp = o.problem.ObjectiveFunction(xp)
					if !max
						fp = -fp
					end
					o.nfcnev += 1

					#  If too many function evaluations occur, terminate the algorithm.
					if o.nfcnev >= maxevl
						if !max
							o.fopt = -o.fopt
						end
						o.err = SAStatusCode(1, "Too many function evaluations")
						return o.fopt, o.err
					end

					#  Accept the new point if the function value increases.
					if fp >= f
						#                 IF(iprint >= 3) THEN
						#    WRITE(*,'("  POINT ACCEPTED")')
						#  END IF
						x .= xp
						f = fp
						o.nacc += 1
						nacp[h] += 1
						nup += 1

						#  If greater than any other point, record as new optimum.
						if fp > o.fopt
							# IF(iprint >= 3) THEN
							#   WRITE(*,'("  NEW OPTIMUM")')
							# END IF
							xopt .= xp
							o.fopt = fp
							nnew += 1
						end

					else
						#  If the point is lower, use the Metropolis criteria to decide on
						#  acceptance or rejection.

						p = exp((fp - f) / t)
						pp = rand(o.rng)
						if pp < p
							#IF(iprint >= 3) CALL prt6(max)
							x .= xp
							f = fp
							o.nacc += 1
							nacp[h] += 1
							ndown += 1
						else
							nrej += 1
							#IF(iprint >= 3) CALL prt7(max)
						end
					end

				end
			end

			#  Adjust VM so that approximately half of all evaluations are accepted.
			for i ∈ 1:n
				ratio = nacp[i] / ns
				if ratio > 0.6
					vm[i] = vm[i] * (1.0 + c[i] * (ratio - 0.6) / 0.4)
				elseif ratio < 0.4
					vm[i] = vm[i] / (1.0 + c[i] * ((0.4 - ratio) / 0.4))
				end
				if vm[i] > (ub[i] - lb[i])
					vm[i] = ub[i] - lb[i]
				end
			end

			# IF(iprint >= 2) THEN
			# CALL prt8(n, vm, xopt, x)
			# END IF
			for i ∈ eachindex(nacp)
				nacp[i] = 0
			end
		end
		# IF(iprint >= 1) THEN
		#  CALL prt9(max,n,t,xopt,vm,o.fopt,nup,ndown,nrej,lnobds,nnew)
		# END IF
		#
		#  Check termination criteria.
		quit = false
		fstar[1] = f
		if (o.fopt - fstar[1]) <= eps
			quit = true
		end
		for i ∈ 1:neps
			if abs(f - fstar[i]) > eps
				quit = false
			end
		end

		#  Terminate SA if appropriate.
		if quit
			x .= xopt
			if !max
				o.fopt = -o.fopt
			end
			o.err = SAStatusCode(0, "Success")
			return o.fopt, o.err
		end

		#  If termination criteria is not met, prepare for another loop.
		t = rt * t
		for i ∈ neps:-1:2
			fstar[i] = fstar[i-1]
		end
		f = o.fopt
		x .= xopt

		#  Loop again.
	end
end
