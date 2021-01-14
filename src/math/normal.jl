export normcdf, norminv, normcdfCody

const DBL_EPSILON = eps()
const DBL_MIN = floatmin() #different from math.SmallestNon0Float64

using SpecialFunctions

const sqrt2 = sqrt(2)
const OneOverSqrtTwo = 1 / sqrt(2)

# normcdf(z::Real) =  normcdfCody(z) #
normcdf(z::Float64) = erfc(-z * OneOverSqrtTwo)/2
norminv(z::Float64) = -sqrt2*erfcinv(2*z)

normcdf(z::T) where {T} = erfc(-z/sqrt(T(2)))/2
norminv(z::T) where {T} = -sqrt(T(2))*erfcinv(2*z)

#const 1.0 = 1.0
#const 0.5 = 0.5
#const 0 = 0.0
const _normal_sixten = 1.6
const thrsh = 0.66291
const root32 = 5.656854248
const OneOverSqrt2Pi = 1 / sqrt(2 * pi)


 #Reference: Cody, W.D. (1993). ALGORITHM 715: SPECFUN - A Portable FORTRAN Package of Special Function Routines and Test Drivers" ACM Transactions on Mathematical Software. 19, 22-32.
 # This function evaluates the normal distribution function: The main computation evaluates near-minimax approximations derived from those in "Rational Chebyshev approximations for the error function" by W. J. Cody, Math. Comp., 1969,
 # 631-637. This transportable program uses rational functions that theoretically approximate the normal distribution function to at least 18 significant decimal digits. The accuracy achieved depends on the arithmetic system, the compiler,
 # the intrinsic functions, and proper selection of the machine-dependent constants.

#
#normcdf(z::Float64) = normcdfCody(z)

const c0 = 0.39894151208813466764
const c1 = 8.8831497943883759412
const c2 = 93.506656132177855979
const c3 = 597.27027639480026226
const c4 = 2494.5375852903726711
const c5 = 6848.1904505362823326
const c6 = 11602.651437647350124
const c7 = 9842.7148383839780218
const c8 = 1.0765576773720192317e-8

const d0 = 22.266688044328115691
const d1 = 235.38790178262499861
const d2 = 1519.377599407554805
const d3 = 6485.558298266760755
const d4 = 18615.571640885098091
const d5 = 34900.952721145977266
const d6 = 38912.003286093271411
const d7 = 19685.429676859990727

const p0 = 0.21589853405795699
const p1 = 0.1274011611602473639
const p2 = 0.022235277870649807
const p3 = 0.001421619193227893466
const p4 = 2.9112874951168792e-5
const p5 = 0.02307344176494017303

const q0 = 1.28426009614491121
const q1 = 0.468238212480865118
const q2 = 0.0659881378689285515
const q3 = 0.00378239633202758244
const q4 = 7.29751555083966205e-5

const a0 = 2.2352520354606839287
const a1 = 161.02823106855587881
const a2 = 1067.6894854603709582
const a3 = 18154.981253343561249
const a4 = 0.065682337918207449113

const b0 = 47.20258190468824187
const b1 = 976.09855173777669322
const b2 = 10260.932208618978205
const b3 = 45507.789335026729956


function normcdfCody(x::Float64)::Float64

    xden = 0.0
    temp = 0.0
    xnum = 0.0
    result = 0.0
    ccum = 0.0
    min = 0.0
    eps = 0.0
    xsq = 0.0
    y = 0.0

    eps = DBL_EPSILON * 0.5
    min = DBL_MIN
    y = abs(x)
    if y <= thrsh
        # Evaluate pnorm for |z| <= 0.66291 */
        xsq = 0
        if y > eps
            xsq = x^2
        end
        xnum = a4 * xsq
        xden = xsq
        xnum = (xnum + a0) * xsq
        xden = (xden + b0) * xsq
        xnum = (xnum + a1) * xsq
        xden = (xden + b1) * xsq
        xnum = (xnum + a2) * xsq
        xden = (xden + b2) * xsq
        result = x * (xnum + a3) / (xden + b3)
        temp = result
        result = 0.5 + temp
        ccum = 0.5 - temp
    elseif y <= root32
        #/* Evaluate pnorm for 0.66291 <= |z| <= sqrt(32) */
        xnum = c8 * y
        xden = y
        xnum = (xnum + c0) * y
        xden = (xden + d0) * y
        xnum = (xnum + c1) * y
        xden = (xden + d1) * y
        xnum = (xnum + c2) * y
        xden = (xden + d2) * y
        xnum = (xnum + c3) * y
        xden = (xden + d3) * y
        xnum = (xnum + c4) * y
        xden = (xden + d4) * y
        xnum = (xnum + c5) * y
        xden = (xden + d5) * y
        xnum = (xnum + c6) * y
        xden = (xden + d6) * y
        result = (xnum + c7) / (xden + d7)
        result *= pdfDenormalized(y)
        #            xsq = fint(y * _normal_sixten) / _normal_sixten;
        #            del = (y - xsq) * (y + xsq);
        #            result = math.Exp(-xsq * xsq * 0.5) * math.Exp(-del * 0.5) * result;
        ccum = 1.0 - result
        if x > 0
            temp = result
            result = ccum
            ccum = temp
        end
    else
        # Evaluate pnorm for |z| > sqrt(32) */
        result = 0
        xsq = 1.0 / (x * x)
        xnum = p5 * xsq
        xden = xsq
        xnum = (xnum + p0) * xsq
        xden = (xden + q0) * xsq
        xnum = (xnum + p1) * xsq
        xden = (xden + q1) * xsq
        xnum = (xnum + p2) * xsq
        xden = (xden + q2) * xsq
        xnum = (xnum + p3) * xsq
        xden = (xden + q3) * xsq
        result = xsq * (xnum + p4) / (xden + q4)
        result = (OneOverSqrt2Pi - result) / y
        result *= pdfDenormalized(y)
        #            xsq = fint(x * _normal_sixten) / _normal_sixten;
        #            del = (x - xsq) * (x + xsq);
        #           result = math.Exp(-xsq * xsq * 0.5) * math.Exp(-del * 0.5) * result;
        ccum = 1.0 - result
        if x > 0
            temp = result
            result = ccum
            ccum = temp
        end
    end
    if result < min
        result = 0.0
    end
    if ccum < min
        ccum = 0.0
    end
    return result
end

function pdfDenormalized(x::Float64)::Float64
	return exp(-0.5 * x * x)
end
