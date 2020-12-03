export norminvAS241

const SPLIT1 = 0.425
const SPLIT2 = 5.0
const CONST1 = 0.180625
const CONST2 = 1.6

# Coefficients for P close to 0.5
const A0 = 3.3871328727963666080E0
const A1 = 1.3314166789178437745E+2
const A2 = 1.9715909503065514427E+3
const A3 = 1.3731693765509461125E+4
const A4 = 4.5921953931549871457E+4
const A5 = 6.7265770927008700853E+4
const A6 = 3.3430575583588128105E+4
const A7 = 2.5090809287301226727E+3
const B1 = 4.2313330701600911252E+1
const B2 = 6.8718700749205790830E+2
const B3 = 5.3941960214247511077E+3
const B4 = 2.1213794301586595867E+4
const B5 = 3.9307895800092710610E+4
const B6 = 2.8729085735721942674E+4
const B7 = 5.2264952788528545610E+3

#  Coefficients for P not close to 0, 0.5 or 1.
const C0 = 1.42343711074968357734E0
const C1 = 4.63033784615654529590E0
const C2 = 5.76949722146069140550E0
const C3 = 3.64784832476320460504E0
const C4 = 1.27045825245236838258E0
const C5 = 2.41780725177450611770E-1
const C6 = 2.27238449892691845833E-2
const C7 = 7.74545014278341407640E-4
const D1 = 2.05319162663775882187E0
const D2 = 1.67638483018380384940E0
const D3 = 6.89767334985100004550E-1
const D4 = 1.48103976427480074590E-1
const D5 = 1.51986665636164571966E-2
const D6 = 5.47593808499534494600E-4
const D7 = 1.05075007164441684324E-9

#  Coefficients for P near 0 or 1.
const E0 = 6.65790464350110377720E0
const E1 = 5.46378491116411436990E0
const E2 = 1.78482653991729133580E0
const E3 = 2.96560571828504891230E-1
const E4 = 2.65321895265761230930E-2
const E5 = 1.24266094738807843860E-3
const E6 = 2.71155556874348757815E-5
const E7 = 2.01033439929228813265E-7
const F1 = 5.99832206555887937690E-1
const F2 = 1.36929880922735805310E-1
const F3 = 1.48753612908506148525E-2
const F4 = 7.86869131145613259100E-4
const F5 = 1.84631831751005468180E-5
const F6 = 1.42151175831644588870E-7
const F7 = 2.04426310338993978564E-15


#InverseCdf computes the inverse cumulative normal distribution value using AS241 algorithm
function norminvAS241(p::Float64)::Float64
    #    DOUBLE PRECISION FUNCTION PPND16 (P, IFAULT)
    # ALGORITHM AS241  APPL. STATIST. (1988) VOL. 37, NO. 3
    # Produces the normal deviate Z corresponding to a given lower
    #   tail area of P Z is accurate to about 1 part in 10**16.
    #
    #   The hash sums below are the sums of the mantissas of the
    #   coefficients.   They are included for use in checking
    #  transcription.

    q = 0.0
    r = 0.0
    ppnd16 = 0.0

    q = p - 0.5
    if abs(q) <= SPLIT1
        r = CONST1 - q * q
        ppnd16 =
            q * (((((((A7 * r + A6) * r + A5) * r + A4) * r + A3) * r + A2) * r + A1) * r + A0) /
            (((((((B7 * r + B6) * r + B5) * r + B4) * r + B3) * r + B2) * r + B1) * r + 1.0)
        return ppnd16
    else
        if q < 0
            r = p
        else
            r = 1.0 - p
        end

        if r <= 0
            if p == 0.0
                ppnd16 = typemin(Float64)
            elseif p == 1.0
                ppnd16 = typemax(Float64)
            else
                ppnd16 = NaN
            end
            return ppnd16
        end

        r = sqrt(-log(r))
        if r <= SPLIT2
            r = r - CONST2
            ppnd16 =
                (((((((C7 * r + C6) * r + C5) * r + C4) * r + C3) * r + C2) * r + C1) * r + C0) /
                (((((((D7 * r + D6) * r + D5) * r + D4) * r + D3) * r + D2) * r + D1) * r + 1.0)
        else
            r = r - SPLIT2
            ppnd16 =
                (((((((E7 * r + E6) * r + E5) * r + E4) * r + E3) * r + E2) * r + E1) * r + E0) /
                (((((((F7 * r + F6) * r + F5) * r + F4) * r + F3) * r + F2) * r + F1) * r + 1.0)
        end
        return q < 0 ? -ppnd16 : ppnd16
    end
end
