export normpdf, normcdfCody1990, erfCody, erfcCody, erfcxCody

const a = [3.1611237438705656, 113.864154151050156, 377.485237685302021, 3209.37758913846947, 0.185777706184603153]
const b = [23.6012909523441209, 244.024637934444173, 1282.61652607737228, 2844.23683343917062]
const c__ = [
    0.564188496988670089,
    8.88314979438837594,
    66.1191906371416295,
    298.635138197400131,
    881.95222124176909,
    1712.04761263407058,
    2051.07837782607147,
    1230.33935479799725,
    2.15311535474403846e-8,
]
const d__ = [
    15.7449261107098347,
    117.693950891312499,
    537.181101862009858,
    1621.38957456669019,
    3290.79923573345963,
    4362.61909014324716,
    3439.36767414372164,
    1230.33935480374942,
]
const p = [
    0.305326634961232344,
    0.360344899949804439,
    0.125781726111229246,
    0.0160837851487422766,
    6.58749161529837803e-4,
    0.0163153871373020978,
]
const q =
    [2.56852019228982242, 1.87295284992346047, 0.527905102951428412, 0.0605183413124413191, 0.00233520497626869185]


const sqrpi = 0.56418958354775628695
const thresh = 0.46875
const sixten = 16.0
# Explanation of machine-dependent constants */
#   XMIN   = the smallest positive floating-point number. */
#   XINF   = the largest positive finite floating-point number. */
#   XNEG   = the largest negative argument acceptable to ERFCX */
#            the negative of the solution to the equation */
#            2*exp(x*x) = XINF. */
#   XSMALL = argument below which erf(x) may be represented by */
#            2*x/sqrt(pi)  and above which  x*x  will not underflow. */
#            A conservative value is the largest machine number X */
#           such that   1.0 + X = 1.0   to machine precision. */
#   XBIG   = largest argument acceptable to ERFC  solution to */
#            the equation:  W(x) * (1-0.5/x**2) = XMIN,  where */
#            W(x) = Math.exp(-x*x)/[x*sqrt(pi)]. */
#   XHUGE  = argument above which  1.0 - 1/(2*x*x) = 1.0  to */
#            machine precision.  A conservative value is */
#            1/[2*sqrt(XSMALL)] */
#   XMAX   = largest acceptable argument to ERFCX the minimum */
#           of XINF and 1/[sqrt(pi)*XMIN]. */
# The numbers below were preselected for IEEE .
const xinf = 1.79e308
const xneg = -26.628
const xsmall = 1.11e-16
const xbig = 26.543
const xhuge = 6.71e7
const xmax = 2.53e307

const norm_cdf_asymptotic_expansion_first_threshold = -10.0
const norm_cdf_asymptotic_expansion_second_threshold = -1 / sqrt(eps())


function normpdf(x::T)::T where {T}
    sqrtTwoPi = sqrt(2*T(pi))
    return exp(-x^2 / 2) /sqrtTwoPi
end
function normpdf(x::Float64)::Float64
    return exp(-x^2 / 2) * OneOverSqrt2Pi
end
normpdf(z, μ, σ) = normpdf((z-μ)/σ)/σ

function normcdfCody1990(z::Float64)::Float64
    if z <= norm_cdf_asymptotic_expansion_first_threshold
        # Asymptotic expansion for very negative z following (26.2.12) on page 408
        # in M. Abramowitz and A. Stegun, Pocketbook of Mathematical Functions, ISBN 3-87144818-4.
        sum = 1.0
        if z >= norm_cdf_asymptotic_expansion_second_threshold
            zsqr = z * z
            i = 1
            g = 1.0
            x = floatmax()
            y = floatmax()
            a = floatmax()
            lasta = 0.0
            while (true)
                lasta = a
                x = Float64(4 * i - 3) / zsqr
                y = x * (Float64(4 * i - 1) / zsqr)
                a = g * (x - y)
                sum -= a
                g *= y
                i += 1
                a = abs(a)
                if lasta > a && a >= abs(sum * eps())
                    break
                end
            end
        end
        return -normpdf(z) * sum / z
    end
    return 0.5 * erfc(-z * OneOverSqrtTwo)
end

function d_int(x::Float64)::Float64
    if x > 0
        return floor(x)
    else
        return floor(-x)
    end
end

function calerf(x::Float64, jint::Int)::Float64

    y = 0.0
    del = 0.0
    ysq = 0.0
    xden = 0.0
    xnum = 0.0
    result = 0.0


    y = abs(x)
    if y <= thresh

        ysq = 0.0
        if y > xsmall
            ysq = y * y
        end
        xnum = a[5] * ysq
        xden = ysq
        for i__ = 1:3
            xnum = (xnum + a[i__]) * ysq
            xden = (xden + b[i__]) * ysq
        end
        result = x * (xnum + a[4]) / (xden + b[4])
        if jint != 0
            result = 1.0 - result
        end
        if jint == 2
            result = exp(ysq) * result
        end

        return result

    elseif y <= 4.0
        xnum = c__[9] * y
        xden = y
        for i__ = 1:7
            xnum = (xnum + c__[i__]) * y
            xden = (xden + d__[i__]) * y

        end
        result = (xnum + c__[8]) / (xden + d__[8])
        if jint != 2
            d__1 = y * sixten
            ysq = d_int(d__1) / sixten
            del = (y - ysq) * (y + ysq)
            d__1 = exp(-ysq * ysq) * exp(-del)
            result = d__1 * result
        end
    else
        result = 0.0
        if y >= xbig
            if jint != 2 || y >= xmax
                return l300(jint, x, result)
            end
            if y >= xhuge
                result = sqrpi / y
                return l300(jint, x, result)
            end
        end
        ysq = 1.0 / (y * y)
        xnum = p[6] * ysq
        xden = ysq
        for i__ = 1:4
            xnum = (xnum + p[i__]) * ysq
            xden = (xden + q[i__]) * ysq
        end
        result = ysq * (xnum + p[5]) / (xden + q[5])
        result = (sqrpi - result) / y
        if jint != 2
            d__1 = y * sixten
            ysq = d_int(d__1) / sixten
            del = (y - ysq) * (y + ysq)
            d__1 = exp(-ysq * ysq) * exp(-del)
            result = d__1 * result
        end
    end

    return l300(jint, x, result)
end

function l300(jint::Int, x::Float64, result::Float64)::Float64
    if jint == 0
        result = (0.5 - result) + 0.5
        if x < 0.0
            result = -(result)
        end
    elseif jint == 1
        if x < 0.0
            result = 2.0 - result
        end
    else
        if x < 0.0
            if x < xneg
                result = xinf
            else
                d__1 = x * sixten
                ysq = d_int(d__1) / sixten
                del = (x - ysq) * (x + ysq)
                y = exp(ysq * ysq) * exp(del)
                result = y + y - result
            end
        end
    end
    return result
end

function erfCody(x::Float64)::Float64
    # -------------------------------------------------------------------- */
    #This subprogram computes approximate values for erf(x). */
    #   (see comments heading CALERF). */
    #   Author/date: W. J. Cody, January 8, 1985 */

    return calerf(x, 0)
end

function erfcCody(x::Float64)::Float64

    return calerf(x, 1)
end

function erfcxCody(x::Float64)::Float64
    # ------------------------------------------------------------------ */
    # This subprogram computes approximate values for Math.exp(x*x) * erfc(x). */
    #   (see comments heading CALERF). */
    #   Author/date: W. J. Cody, March 30, 1987 */
    # ------------------------------------------------------------------ */

    return calerf(x, 2)
end
