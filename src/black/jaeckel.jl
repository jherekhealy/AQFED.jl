using AQFED.Math

const TwoPiOverSqrtTwentSeven = 1.209199576156145233729385505094770488189377498728
const PiOverSix = 0.523598775598298873077107230546583814032861566563
const FourthRootEpsilon = sqrt(SqrtEpsilon)
const SixteenthRootEpsilon = sqrt(sqrt(FourthRootEpsilon))

const SqrtPiOverTwo = sqrt(pi / 2)
const SqrtThree = sqrt(3)
const SqrtTwoPi = sqrt(2 * pi)
const SqrtOneOverThree = 1 / sqrt(3)
#TwoPiOverSqrtTwentSeven = 2*pi/sqrt(27)
# Set this to 0 if you want positive results for (positive) denormalized inputs, else to DBL_MIN.
# Note that you cannot achieve full machine accuracy from denormalized inputs
const DENORMALIZATION_CUTOFF = 0.0
const VOLATILITY_VALUE_TO_SIGNAL_PRICE_IS_BELOW_INTRINSIC = typemin(Float64)
const VOLATILITY_VALUE_TO_SIGNAL_PRICE_IS_ABOVE_MAXIMUM = typemax(Float64)

const SqrtMaxDouble = sqrt(floatmax())
const SqrtMinDouble = sqrt(floatmin())

function is_below_horizon(x::Float64)::Bool
    return abs(x) < DENORMALIZATION_CUTOFF
end # This weeds out denormalized (a.k.a. 'subnormal') numbers.

# private static::Int implied_volatility_householder_method_order =  4
function householder_factor(newton::Float64, halley::Float64, hh3::Float64)::Float64
    return (1 + halley * newton / 2) / (1 + newton * (halley + hh3 * newton / 6))
end

function normalised_intrinsic(x::Float64, q::Int)::Float64
    if q * x <= 0
        return 0
    end
    x2 = x^2
    if x2 < 98 * FourthRootEpsilon  # The factor 98 is computed from last coefficient: √√92897280 = 98.1749
        return abs(max(
            q *
            x *
            (1 + x2 * ((1.0 / 24.0) + x2 * ((1.0 / 1920.0) + x2 * ((1.0 / 322560.0) + (1.0 / 92897280.0) * x2)))),
            0,
        ))
    end
    b_max = exp(x / 2)
    one_over_b_max = 1 / b_max
    value = abs(max(q * (b_max - one_over_b_max), 0))
    return value
end

normalised_intrinsic_call(x) = normalised_intrinsic(x, 1)

# Asymptotic expansion of
#
#                  b  =  Φ(h+t)·math.Exp(x/2) - Φ(h-t)·math.Exp(-x/2)
# with
#                  h  =  x/s   and   t  =  s/2
# which makes
#                  b  =  Φ(h+t)·math.Exp(h·t) - Φ(h-t)·math.Exp(-h·t)
#
#                        math.Exp(-(h²+t²)/2)
#                     =  ---------------  ·  [ Y(h+t) - Y(h-t) ]
#                            √(2π)
# with
#               Y(z) := Φ(z)/φ(z)
#
# for large negative (t-|h|) by the aid of Abramowitz & Stegun (26.2.12) where Φ(z) = φ(z)/|z|·[1-1/z^2+...].
# We define
#                         r
#             A(h,t) :=  --- · [ Y(h+t) - Y(h-t) ]
#                         t
#
# with r := (h+t)·(h-t) and give an expansion for A(h,t) in q:=(h/r)² expressed in terms of e:=(t/h)² .
function asymptotic_expansion_of_normalized_black_call(h::Float64, t::Float64)::Float64
    e = (t / h) * (t / h)
    r = ((h + t) * (h - t))
    q = (h / r) * (h / r)
    # 17th order asymptotic expansion of A(h,t) in q, sufficient for Φ(h) [and thus y(h)] to have relative accuracy of 1.64E-16 for h <= η  with  η:=-10.
    asymptotic_expansion_sum = (
        2.0 +
        q * (
            -6.0E0 - 2.0 * e +
            3.0 *
            q *
            (
                1.0E1 +
                e * (2.0E1 + 2.0 * e) +
                5.0 *
                q *
                (
                    -1.4E1 +
                    e * (-7.0E1 + e * (-4.2E1 - 2.0 * e)) +
                    7.0 *
                    q *
                    (
                        1.8E1 +
                        e * (1.68E2 + e * (2.52E2 + e * (7.2E1 + 2.0 * e))) +
                        9.0 *
                        q *
                        (
                            -2.2E1 +
                            e * (-3.3E2 + e * (-9.24E2 + e * (-6.6E2 + e * (-1.1E2 - 2.0 * e)))) +
                            1.1E1 *
                            q *
                            (
                                2.6E1 +
                                e * (5.72E2 + e * (2.574E3 + e * (3.432E3 + e * (1.43E3 + e * (1.56E2 + 2.0 * e))))) +
                                1.3E1 *
                                q *
                                (
                                    -3.0E1 +
                                    e * (
                                        -9.1E2 +
                                        e * (
                                            -6.006E3 +
                                            e * (-1.287E4 + e * (-1.001E4 + e * (-2.73E3 + e * (-2.1E2 - 2.0 * e))))
                                        )
                                    ) +
                                    1.5E1 *
                                    q *
                                    (
                                        3.4E1 +
                                        e * (
                                            1.36E3 +
                                            e * (
                                                1.2376E4 +
                                                e * (
                                                    3.8896E4 +
                                                    e * (
                                                        4.862E4 +
                                                        e * (2.4752E4 + e * (4.76E3 + e * (2.72E2 + 2.0 * e)))
                                                    )
                                                )
                                            )
                                        ) +
                                        1.7E1 *
                                        q *
                                        (
                                            -3.8E1 +
                                            e * (
                                                -1.938E3 +
                                                e * (
                                                    -2.3256E4 +
                                                    e * (
                                                        -1.00776E5 +
                                                        e * (
                                                            -1.84756E5 +
                                                            e * (
                                                                -1.51164E5 +
                                                                e * (
                                                                    -5.4264E4 +
                                                                    e * (-7.752E3 + e * (-3.42E2 - 2.0 * e))
                                                                )
                                                            )
                                                        )
                                                    )
                                                )
                                            ) +
                                            1.9E1 *
                                            q *
                                            (
                                                4.2E1 +
                                                e * (
                                                    2.66E3 +
                                                    e * (
                                                        4.0698E4 +
                                                        e * (
                                                            2.3256E5 +
                                                            e * (
                                                                5.8786E5 +
                                                                e * (
                                                                    7.05432E5 +
                                                                    e * (
                                                                        4.0698E5 +
                                                                        e * (
                                                                            1.08528E5 +
                                                                            e * (1.197E4 + e * (4.2E2 + 2.0 * e))
                                                                        )
                                                                    )
                                                                )
                                                            )
                                                        )
                                                    )
                                                ) +
                                                2.1E1 *
                                                q *
                                                (
                                                    -4.6E1 +
                                                    e * (
                                                        -3.542E3 +
                                                        e * (
                                                            -6.7298E4 +
                                                            e * (
                                                                -4.90314E5 +
                                                                e * (
                                                                    -1.63438E6 +
                                                                    e * (
                                                                        -2.704156E6 +
                                                                        e * (
                                                                            -2.288132E6 +
                                                                            e * (
                                                                                -9.80628E5 +
                                                                                e * (
                                                                                    -2.01894E5 +
                                                                                    e * (
                                                                                        -1.771E4 +
                                                                                        e * (-5.06E2 - 2.0 * e)
                                                                                    )
                                                                                )
                                                                            )
                                                                        )
                                                                    )
                                                                )
                                                            )
                                                        )
                                                    ) +
                                                    2.3E1 *
                                                    q *
                                                    (
                                                        5.0E1 +
                                                        e * (
                                                            4.6E3 +
                                                            e * (
                                                                1.0626E5 +
                                                                e * (
                                                                    9.614E5 +
                                                                    e * (
                                                                        4.08595E6 +
                                                                        e * (
                                                                            8.9148E6 +
                                                                            e * (
                                                                                1.04006E7 +
                                                                                e * (
                                                                                    6.53752E6 +
                                                                                    e * (
                                                                                        2.16315E6 +
                                                                                        e * (
                                                                                            3.542E5 +
                                                                                            e * (
                                                                                                2.53E4 +
                                                                                                e * (6.0E2 + 2.0 * e)
                                                                                            )
                                                                                        )
                                                                                    )
                                                                                )
                                                                            )
                                                                        )
                                                                    )
                                                                )
                                                            )
                                                        ) +
                                                        2.5E1 *
                                                        q *
                                                        (
                                                            -5.4E1 +
                                                            e * (
                                                                -5.85E3 +
                                                                e * (
                                                                    -1.6146E5 +
                                                                    e * (
                                                                        -1.77606E6 +
                                                                        e * (
                                                                            -9.37365E6 +
                                                                            e * (
                                                                                -2.607579E7 +
                                                                                e * (
                                                                                    -4.01166E7 +
                                                                                    e * (
                                                                                        -3.476772E7 +
                                                                                        e * (
                                                                                            -1.687257E7 +
                                                                                            e * (
                                                                                                -4.44015E6 +
                                                                                                e * (
                                                                                                    -5.9202E5 +
                                                                                                    e * (
                                                                                                        -3.51E4 +
                                                                                                        e * (
                                                                                                            -7.02E2 -
                                                                                                            2.0 * e
                                                                                                        )
                                                                                                    )
                                                                                                )
                                                                                            )
                                                                                        )
                                                                                    )
                                                                                )
                                                                            )
                                                                        )
                                                                    )
                                                                )
                                                            ) +
                                                            2.7E1 *
                                                            q *
                                                            (
                                                                5.8E1 +
                                                                e * (
                                                                    7.308E3 +
                                                                    e * (
                                                                        2.3751E5 +
                                                                        e * (
                                                                            3.12156E6 +
                                                                            e * (
                                                                                2.003001E7 +
                                                                                e * (
                                                                                    6.919458E7 +
                                                                                    e * (
                                                                                        1.3572783E8 +
                                                                                        e * (
                                                                                            1.5511752E8 +
                                                                                            e * (
                                                                                                1.0379187E8 +
                                                                                                e * (
                                                                                                    4.006002E7 +
                                                                                                    e * (
                                                                                                        8.58429E6 +
                                                                                                        e * (
                                                                                                            9.5004E5 +
                                                                                                            e * (
                                                                                                                4.7502E4 +
                                                                                                                e *
                                                                                                                (
                                                                                                                    8.12E2 +
                                                                                                                    2.0 *
                                                                                                                    e
                                                                                                                )
                                                                                                            )
                                                                                                        )
                                                                                                    )
                                                                                                )
                                                                                            )
                                                                                        )
                                                                                    )
                                                                                )
                                                                            )
                                                                        )
                                                                    )
                                                                ) +
                                                                2.9E1 *
                                                                q *
                                                                (
                                                                    -6.2E1 +
                                                                    e * (
                                                                        -8.99E3 +
                                                                        e * (
                                                                            -3.39822E5 +
                                                                            e * (
                                                                                -5.25915E6 +
                                                                                e * (
                                                                                    -4.032015E7 +
                                                                                    e * (
                                                                                        -1.6934463E8 +
                                                                                        e * (
                                                                                            -4.1250615E8 +
                                                                                            e * (
                                                                                                -6.0108039E8 +
                                                                                                e * (
                                                                                                    -5.3036505E8 +
                                                                                                    e * (
                                                                                                        -2.8224105E8 +
                                                                                                        e * (
                                                                                                            -8.870433E7 +
                                                                                                            e * (
                                                                                                                -1.577745E7 +
                                                                                                                e *
                                                                                                                (
                                                                                                                    -1.472562E6 +
                                                                                                                    e *
                                                                                                                    (
                                                                                                                        -6.293E4 +
                                                                                                                        e *
                                                                                                                        (
                                                                                                                            -9.3E2 -
                                                                                                                            2.0 *
                                                                                                                            e
                                                                                                                        )
                                                                                                                    )
                                                                                                                )
                                                                                                            )
                                                                                                        )
                                                                                                    )
                                                                                                )
                                                                                            )
                                                                                        )
                                                                                    )
                                                                                )
                                                                            )
                                                                        )
                                                                    ) +
                                                                    3.1E1 *
                                                                    q *
                                                                    (
                                                                        6.6E1 +
                                                                        e * (
                                                                            1.0912E4 +
                                                                            e * (
                                                                                4.74672E5 +
                                                                                e * (
                                                                                    8.544096E6 +
                                                                                    e * (
                                                                                        7.71342E7 +
                                                                                        e * (
                                                                                            3.8707344E8 +
                                                                                            e * (
                                                                                                1.14633288E9 +
                                                                                                e * (
                                                                                                    2.07431664E9 +
                                                                                                    e * (
                                                                                                        2.33360622E9 +
                                                                                                        e * (
                                                                                                            1.6376184E9 +
                                                                                                            e * (
                                                                                                                7.0963464E8 +
                                                                                                                e *
                                                                                                                (
                                                                                                                    1.8512208E8 +
                                                                                                                    e *
                                                                                                                    (
                                                                                                                        2.7768312E7 +
                                                                                                                        e *
                                                                                                                        (
                                                                                                                            2.215136E6 +
                                                                                                                            e *
                                                                                                                            (
                                                                                                                                8.184E4 +
                                                                                                                                e *
                                                                                                                                (
                                                                                                                                    1.056E3 +
                                                                                                                                    2.0 *
                                                                                                                                    e
                                                                                                                                )
                                                                                                                            )
                                                                                                                        )
                                                                                                                    )
                                                                                                                )
                                                                                                            )
                                                                                                        )
                                                                                                    )
                                                                                                )
                                                                                            )
                                                                                        )
                                                                                    )
                                                                                )
                                                                            )
                                                                        ) +
                                                                        3.3E1 *
                                                                        (
                                                                            -7.0E1 +
                                                                            e * (
                                                                                -1.309E4 +
                                                                                e * (
                                                                                    -6.49264E5 +
                                                                                    e * (
                                                                                        -1.344904E7 +
                                                                                        e * (
                                                                                            -1.4121492E8 +
                                                                                            e * (
                                                                                                -8.344518E8 +
                                                                                                e * (
                                                                                                    -2.9526756E9 +
                                                                                                    e * (
                                                                                                        -6.49588632E9 +
                                                                                                        e * (
                                                                                                            -9.0751353E9 +
                                                                                                            e * (
                                                                                                                -8.1198579E9 +
                                                                                                                e *
                                                                                                                (
                                                                                                                    -4.6399188E9 +
                                                                                                                    e *
                                                                                                                    (
                                                                                                                        -1.6689036E9 +
                                                                                                                        e *
                                                                                                                        (
                                                                                                                            -3.67158792E8 +
                                                                                                                            e *
                                                                                                                            (
                                                                                                                                -4.707164E7 +
                                                                                                                                e *
                                                                                                                                (
                                                                                                                                    -3.24632E6 +
                                                                                                                                    e *
                                                                                                                                    (
                                                                                                                                        -1.0472E5 +
                                                                                                                                        e *
                                                                                                                                        (
                                                                                                                                            -1.19E3 -
                                                                                                                                            2.0 *
                                                                                                                                            e
                                                                                                                                        )
                                                                                                                                    )
                                                                                                                                )
                                                                                                                            )
                                                                                                                        )
                                                                                                                    )
                                                                                                                )
                                                                                                            )
                                                                                                        )
                                                                                                    )
                                                                                                )
                                                                                            )
                                                                                        )
                                                                                    )
                                                                                )
                                                                            )
                                                                        ) *
                                                                        q
                                                                    )
                                                                )
                                                            )
                                                        )
                                                    )
                                                )
                                            )
                                        )
                                    )
                                )
                            )
                        )
                    )
                )
            )
        )
    )
    b = Math.OneOverSqrt2Pi * exp((-(h^2 + t^2) / 2)) * (t / r) * asymptotic_expansion_sum
    return abs(max(b, 0))
end

const asymptotic_expansion_accuracy_threshold = -10.0

function normalised_black_call_using_erfcx(h::Float64, t::Float64)::Float64
    # Given h = x/s and t = s/2, the normalised Black function can be written as
    #
    #     b(x,s)  =  Φ(x/s+s/2)·math.Exp(x/2)  -   Φ(x/s-s/2)·math.Exp(-x/2)
    #             =  Φ(h+t)·math.Exp(h·t)      -   Φ(h-t)·math.Exp(-h·t) .                     (*)
    #
    # It is mentioned in section 4 (and discussion of figures 2 and 3) of George Marsaglia's article "Evaluating the
    # Normal Distribution" (available at http:#www.jstatsoft.org/v11/a05/paper) that the error of any cumulative normal
    # function Φ(z) is dominated by the hardware (or compiler implementation) accuracy of math.Exp(-z²/2) which is not
    # reliably more than 14 digits when z is large. The accuracy of Φ(z) typically starts coming down to 14 digits when
    # z is around -8. For the (normalised) Black function, as above in (*), this means that we are subtracting two terms
    # that are each products of terms with about 14 digits of accuracy. The net result, in each of the products, is even
    # less accuracy, and then we are taking the difference of these terms, resulting in even less accuracy. When we are
    # using the asymptotic expansion asymptotic_expansion_of_normalized_black_call() invoked in the second branch at the
    # beginning of this function, we are using only *one* exponential instead of 4, and this improves accuracy. It
    # actually improves it a bit more than you would expect from the above logic, namely, almost the full two missing
    # digits (in 64 bit IEEE floating point).  Unfortunately, going higher order in the asymptotic expansion will not
    # enable us to gain more accuracy (by extending the range in which we could use the expansion) since the asymptotic
    # expansion, being a divergent series, can never gain 16 digits of accuracy for z=-8 or just below. The best you can
    # get is about 15 digits (just), for about 35 terms in the series (26.2.12), which would result in an prohibitively
    # long expression in function asymptotic expansion asymptotic_expansion_of_normalized_black_call(). In this last branch,
    # here, we therefore take a different tack as follows.
    #     The "scaled complementary error function" is defined as erfcx(z) = math.Exp(z²)·erfc(z). Cody's implementation of this
    # function as published in "Rational Chebyshev approximations for the error function", W. J. Cody, Math. Comp., 1969, pp.
    # 631-638, uses rational functions that theoretically approximates erfcx(x) to at least 18 significant decimal digits,
    # *without* the use of the exponential function when x>4, which translates to about z<-5.66 in Φ(z). To make use of it,
    # we write
    #             Φ(z) = math.Exp(-z²/2)·erfcx(-z/√2)/2
    #
    # to transform the normalised black function to
    #
    #   b   =  ½ · math.Exp(-½(h²+t²)) · [ erfcx(-(h+t)/√2) -  erfcx(-(h-t)/√2) ]
    #
    # which now involves only one exponential, instead of three, when |h|+|t| > 5.66 , and the difference inside the
    # square bracket is between the evaluation of two rational functions, which, typically, according to Marsaglia,
    # retains the full 16 digits of accuracy (or just a little less than that).
    #
    b =
        exp(-(h^2 + t^2) / 2) / 2 *
        (jaeckel_erfcx(-Math.OneOverSqrtTwo * (h + t)) - jaeckel_erfcx(-Math.OneOverSqrtTwo * (h - t)))
    return abs(max(b, 0))
end

# Calculation of
#
#                  b  =  Φ(h+t)·math.Exp(h·t) - Φ(h-t)·math.Exp(-h·t)
#
#                        math.Exp(-(h²+t²)/2)
#                     =  --------------- ·  [ Y(h+t) - Y(h-t) ]
#                            √(2π)
# with
#               Y(z) := Φ(z)/φ(z)
#
# using an expansion of Y(h+t)-Y(h-t) for small t to twelvth order in t.
# Theoretically accurate to (better than) precision  ε = 2.23E-16  when  h<=0  and  t < τ  with  τ := 2·ε^(1/16) ≈ 0.21.
# The main bottleneck for precision is the coefficient a:=1+h·Y(h) when |h|>1 .
function small_t_expansion_of_normalized_black_call(h::Float64, t::Float64)::Float64
    # Y(h) := Φ(h)/φ(h) = √(π/2)·erfcx(-h/√2)
    # a := 1+h·Y(h)  --- Note that due to h<0, and h·Y(h) -> -1 (from above) as h -> -∞, we also have that a>0 and a -> 0 as h -> -∞
    # w := t² , h2 := h²
    a = 1 + h * (0.5 * SqrtTwoPi) * jaeckel_erfcx(-Math.OneOverSqrtTwo * h)
    w = t^2
    h2 = h^2
    expansion =
        2 *
        t *
        (
            a +
            w * (
                (-1 + 3 * a + a * h2) / 6 +
                w * (
                    (-7 + 15 * a + h2 * (-1 + 10 * a + a * h2)) / 120 +
                    w * (
                        (-57 + 105 * a + h2 * (-18 + 105 * a + h2 * (-1 + 21 * a + a * h2))) / 5040 +
                        w * (
                            (
                                -561 +
                                945 * a +
                                h2 * (-285 + 1260 * a + h2 * (-33 + 378 * a + h2 * (-1 + 36 * a + a * h2)))
                            ) / 362880 +
                            w * (
                                (
                                    -6555 +
                                    10395 * a +
                                    h2 * (
                                        -4680 +
                                        17325 * a +
                                        h2 * (-840 + 6930 * a + h2 * (-52 + 990 * a + h2 * (-1 + 55 * a + a * h2)))
                                    )
                                ) / 39916800 +
                                (
                                    (
                                        -89055 +
                                        135135 * a +
                                        h2 * (
                                            -82845 +
                                            270270 * a +
                                            h2 * (
                                                -20370 +
                                                135135 * a +
                                                h2 * (
                                                    -1926 +
                                                    25740 * a +
                                                    h2 * (-75 + 2145 * a + h2 * (-1 + 78 * a + a * h2))
                                                )
                                            )
                                        )
                                    ) * w
                                ) / 6227020800.0
                            )
                        )
                    )
                )
            )
        )
    b = Math.OneOverSqrt2Pi * exp((-(h * h + t * t) / 2)) * expansion
    return abs(max(b, 0))
end

const small_t_expansion_of_normalized_black_threshold = 2 * SixteenthRootEpsilon


#         b(x,s)  =  Φ(x/s+s/2)·math.Exp(x/2)  -   Φ(x/s-s/2)·math.Exp(-x/2)
#                 =  Φ(h+t)·math.Exp(x/2)      -   Φ(h-t)·math.Exp(-x/2)
# with
#                  h  =  x/s   and   t  =  s/2
function normalized_black_call_using_jaeckel_normcdf(x::Float64, s::Float64)::Float64
    h = x / s
    t = 0.5 * s
    b_max = exp(0.5 * x)
    b = jaeckel_normcdf(h + t) * b_max - jaeckel_normcdf(h - t) / b_max

    return abs(max(b, 0))
end

function normalised_black_call(x::Float64, s::Float64)::Float64
    if x > 0
        return normalised_intrinsic_call(x) + normalised_black_call(-x, s)
    end
    ax = abs(x)
    as = abs(s)
    if as <= ax * DENORMALIZATION_CUTOFF
        return normalised_intrinsic_call(x)
    end
    # Denote h := x/s and t := s/2.
    # We evaluate the condition |h|>|η|, i.e., h<η  &&  t < τ+|h|-|η|  avoiding any divisions by s , where η = asymptotic_expansion_accuracy_threshold  and τ = small_t_expansion_of_normalized_black_threshold .
    if (x < as * asymptotic_expansion_accuracy_threshold) && (
        0.5 * s * s + x <
        as * (small_t_expansion_of_normalized_black_threshold + asymptotic_expansion_accuracy_threshold)
    )
        # Region 1.
        return asymptotic_expansion_of_normalized_black_call(x / s, 0.5 * s)
    end
    if 0.5 * as < small_t_expansion_of_normalized_black_threshold
        # Region 2.
        return small_t_expansion_of_normalized_black_call(x / s, 0.5 * s)
    end
    # When b is more than, say, about 85% of b_max=math.Exp(x/2), then b is dominated by the first of the two terms in the Black formula, and we retain more accuracy by not attempting to combine the two terms in any way.
    # We evaluate the condition h+t>0.85  avoiding any divisions by s.
    if x + 0.5 * s^2 > as * 0.85
        # Region 3.
        return normalized_black_call_using_jaeckel_normcdf(x, s)
        # Region 4.
    end
    return normalised_black_call_using_erfcx(x / s, 0.5 * s)
end


function normalised_vega(x::Float64, s::Float64)::Float64
    ax = abs(x)
    if ax <= 0
        return Math.OneOverSqrt2Pi * exp(-0.125 * s^2)
    end

    if s <= 0 || s <= ax * SqrtMinDouble
        return 0
    end

    return Math.OneOverSqrt2Pi * exp(-0.5 * ((x / s)^2 + (0.5 * s)^2))
end

function normalised_black(x::Float64, s::Float64, q::Int)::Float64
    if q < 0
        return normalised_black_call(-x, s) #Reciprocal-strike call-put equivalence
    else
        return normalised_black_call(x, s)
    end
end

function black(F::Float64, K::Float64, sigma::Float64, T::Float64, q::Int)::Float64
    intrinsic = abs(max(q * (F - K), 0))
    # Map in-the-money to out-of-the-money
    if q * (F - K) > 0
        return intrinsic + black(F, K, sigma, T, -q)
    end
    return max(intrinsic, (sqrt(F) * sqrt(K)) * normalised_black(log(F / K), sigma * sqrt(T), q))
end

#    #ifdef COMPUTE_LOWER_MAP_DERIVATIVES_INDIVIDUALLY
#    double f_lower_map(final double x,final double s){
#        if (is_below_horizon(x))
#            return 0
#        if (is_below_horizon(s))
#            return 0
#        final double z=SQRT_ONE_OVER_THREE*math.Abs(x)/s, Phi=jaeckel_normcdf(-z)
#        return TWO_PI_OVER_SQRT_TWENTY_SEVEN*math.Abs(x)*(Phi*Phi*Phi)
#    }
#    double d_f_lower_map_d_beta(final double x,final double s){
#        if (is_below_horizon(s))
#            return 1
#        final double z=SQRT_ONE_OVER_THREE*math.Abs(x)/s, y = z*z, Phi=jaeckel_normcdf(-z)
#        return TWO_PI*y*(Phi*Phi) * math.Exp(y+0.125*s*s)
#    }
#    double d2_f_lower_map_d_beta2(final double x,final double s){
#        final double ax=math.Abs(x), z=SQRT_ONE_OVER_THREE*ax/s, y = z*z, s2=s*s, Phi=jaeckel_normcdf(-z), phi=norm_pdf(z)
#        return PI_OVER_SIX * y/(s2*s) * Phi * ( 8*SQRT_THREE*s*ax + (3*s2*(s2-8)-8*x*x)*Phi/phi ) * math.Exp(2*y+0.25*s2)
#    }
#    void compute_f_lower_map_and_first_two_derivatives(final double x,final double s,double &f,double &fp,double &fpp){
#        f   = f_lower_map(x,s)
#        fp  = d_f_lower_map_d_beta(x,s)
#        fpp = d2_f_lower_map_d_beta2(x,s)
#    }
#    #else
function compute_f_lower_map_and_first_two_derivatives(x::Float64, s::Float64)
    ax = abs(x)
    z = SqrtOneOverThree * ax / s
    y = z^2
    s2 = s^2
    Phi = jaeckel_normcdf(-z)
    phi = normpdf(z)

    fpp =
        pi / 6 * y / (s2 * s) *
        Phi *
        (8 * SqrtThree * s * ax + (3 * s2 * (s2 - 8) - 8 * x * x) * Phi / phi) *
        exp(2 * y + 0.25 * s2)
    if is_below_horizon(s)
        fp = 1.0
        f = 0.0
        return f, fp, fpp
    else
        Phi2 = Phi * Phi
        fp = 2 * pi * y * Phi2 * exp(y + 0.125 * s * s)
        f = 0.0
        if !is_below_horizon(x)
            f = TwoPiOverSqrtTwentSeven * ax * (Phi2 * Phi)
        end
        return f, fp, fpp
    end
end

##endif
function inverse_f_lower_map(x::Float64, f::Float64)::Float64
    if is_below_horizon(f)
        return 0
    end
    return abs(x / (SqrtThree * jaeckel_norminv((f / (TwoPiOverSqrtTwentSeven * abs(x)))^(1.0 / 3.0))))
end

#    #ifdef COMPUTE_UPPER_MAP_DERIVATIVES_INDIVIDUALLY
#    double f_upper_map(final double s){
#        return jaeckel_normcdf(-0.5*s)
#    }
#    double d_f_upper_map_d_beta(final double x,final double s){
#        return is_below_horizon(x) ? -0.5 : -0.5*math.Exp(0.5*square(x/s))
#    }
#    double d2_f_upper_map_d_beta2(final double x,final double s){
#        if (is_below_horizon(x))
#            return 0
#        final double w = square(x/s)
#        return SQRT_PI_OVER_TWO*math.Exp(w+0.125*s*s)*w/s
#    }
#    void compute_f_upper_map_and_first_two_derivatives(final double x,final double s,double &f,double &fp,double &fpp){
#        f   = f_upper_map(s)
#        fp  = d_f_upper_map_d_beta(x,s)
#        fpp = d2_f_upper_map_d_beta2(x,s)
#    }
#    #else
function compute_f_upper_map_and_first_two_derivatives(x::Float64, s::Float64)
    f = jaeckel_normcdf(-0.5 * s)
    if is_below_horizon(x)
        fp = -0.5
        fpp = 0.0
        return f, fp, fpp
    else
        w = (x / s)^2
        fp = -0.5 * exp(0.5 * w)
        fpp = SqrtPiOverTwo * exp(w + 0.125 * s * s) * w / s
        return f, fp, fpp
    end
end

#    #endif

function inverse_f_upper_map(f::Float64)::Float64
    return -2.0 * jaeckel_norminv(f)
end

# See http:#en.wikipedia.org/wiki/Householder%27s_method for a detailed explanation of the third order Householder iteration.
#
# Given the objective function g(s) whose root x such that 0 = g(s) we seek, iterate
#
#         s_n+1  =  s_n  -  (g/g') · [ 1 - (g''/g')·(g/g') ] / [ 1 - (g/g')·( (g''/g') - (g'''/g')·(g/g')/6 ) ]
#
# Denoting  newton:=-(g/g'), halley:=(g''/g'), and hh3:=(g'''/g'), this reads
#
#         s_n+1  =  s_n  +  newton · [ 1 + halley·newton/2 ] / [ 1 + newton·( halley + hh3·newton/6 ) ]
#
#
# NOTE that this function returns 0 when beta<intrinsic without any safety checks.
#
function unchecked_normalised_implied_volatility_from_a_transformed_rational_guess_with_limited_iterations(
    beta::Float64,
    x::Float64,
    q::Int,
    N::Int,
)::Float64
    # Subtract::Intrinsic.
    if q * x > 0
        beta = abs(max(beta - normalised_intrinsic(x, q), 0.0))
        q = -q
    end
    # Map puts to calls
    if q < 0
        x = -x
        q = -q
    end
    if beta <= 0  # For negative or zero prices we return 0.
        return 0
    end
    if beta < DENORMALIZATION_CUTOFF  # For positive but denormalized (a.k.a. 'subnormal') prices, we return 0 since it would be impossible to converge to full machine accuracy anyway.
        return 0
    end
    b_max = exp(0.5 * x)
    if beta >= b_max
        return VOLATILITY_VALUE_TO_SIGNAL_PRICE_IS_ABOVE_MAXIMUM
    end
    iterations = 0
    direction_reversal_count = 0
    f = -floatmax()
    s = -floatmax()
    ds = s
    ds_previous = 0.0
    s_left = floatmin()
    s_right = floatmax()
    # The temptation is great to use the optimised form b_c = math.Exp(x/2)/2-math.Exp(-x/2)·Phi(math.Sqrt(-2·x)) but that would require implementing all of the above types of round-off and over/underflow handling for this expression, too.
    s_c = sqrt(abs(2 * x))
    b_c = normalised_black_call(x, s_c)
    v_c = normalised_vega(x, s_c)
    # Four branches.
    if beta < b_c
        #println(s_c," ",b_c," ", v_c," black ",x)
        s_l = s_c - b_c / v_c
        b_l = normalised_black_call(x, s_l)
        if beta < b_l
            f_lower_map_l, d_f_lower_map_l_d_beta, d2_f_lower_map_l_d_beta2 =
                compute_f_lower_map_and_first_two_derivatives(x, s_l)
            r_ll = convex_rational_cubic_control_parameter_to_fit_second_derivative_at_right_side(
                0.0,
                b_l,
                0.0,
                f_lower_map_l,
                1.0,
                d_f_lower_map_l_d_beta,
                d2_f_lower_map_l_d_beta2,
                true,
            )
            #println(x," ",s_l," ",r_ll," rat ",beta," ",b_l," ",f_lower_map_l," ",d_f_lower_map_l_d_beta, " ",d2_f_lower_map_l_d_beta2)
            f = rational_cubic_interpolation(beta, 0.0, b_l, 0.0, f_lower_map_l, 1.0, d_f_lower_map_l_d_beta, r_ll)
            if !(f > 0)  # This can happen due to roundoff truncation for extreme values such as |x|>500.
                # We switch to quadratic::Interpolation using f(0)≡0, f(b_l), and f'(0)≡1 to specify the quadratic.
                t = beta / b_l
                f = (f_lower_map_l * t + b_l * (1 - t)) * t
            end
            s = inverse_f_lower_map(x, f)
            #    println("inverse ",x,f)
            s_right = s_l
            #
            # In this branch, which comprises the lowest segment, the objective function is
            #     g(s) = 1/ln(b(x,s)) - 1/ln(beta)
            #          ≡ 1/ln(b(s)) - 1/ln(beta)
            # This makes
            #              g'               =   -b'/(b·ln(b)²)
            #              newton = -g/g'   =   (ln(beta)-ln(b))·ln(b)/ln(beta)·b/b'
            #              halley = g''/g'  =   b''/b'  -  b'/b·(1+2/ln(b))
            #              hh3    = g'''/g' =   b'''/b' +  2(b'/b)²·(1+3/ln(b)·(1+1/ln(b)))  -  3(b''/b)·(1+2/ln(b))
            #
            # The Householder(3) iteration is
            #     s_n+1  =  s_n  +  newton · [ 1 + halley·newton/2 ] / [ 1 + newton·( halley + hh3·newton/6 ) ]
            #
            while (iterations < N) && (abs(ds) > eps() * s)
                #    println("1 ds ",ds," it ",iterations," s ",s)

                if ds * ds_previous < 0
                    direction_reversal_count += 1
                end
                if iterations > 0 && (3 == direction_reversal_count || !(s > s_left && s < s_right))
                    # If looping inefficently, or the forecast step takes us outside the bracket, or onto its edges, switch to binary nesting.
                    # NOTE that this can only really happen for very extreme values of |x|, such as |x| = |ln(F/K)| > 500.
                    s = 0.5 * (s_left + s_right)
                    if s_right - s_left <= eps() * s
                        break
                    end
                    direction_reversal_count = 0
                    ds = 0
                end
                ds_previous = ds
                b = normalised_black_call(x, s)
                bp = normalised_vega(x, s)
                if b > beta && s < s_right
                    s_right = s
                elseif b < beta && s > s_left
                    s_left = s # Tighten the bracket if applicable.
                end
                if b <= 0 || bp <= 0  # Numerical underflow. Switch to binary nesting for this iteration.
                    ds = 0.5 * (s_left + s_right) - s
                else
                    ln_b = log(b)
                    ln_beta = log(beta)
                    bpob = bp / b
                    h = x / s
                    b_halley = h * h / s - s / 4
                    newton = (ln_beta - ln_b) * ln_b / ln_beta / bpob
                    halley = b_halley - bpob * (1 + 2 / ln_b)
                    b_hh3 = b_halley * b_halley - 3 * (h / s)^2 - 0.25
                    hh3 = b_hh3 + 2 * (bpob)^2 * (1 + 3 / ln_b * (1 + 1 / ln_b)) - 3 * b_halley * bpob * (1 + 2 / ln_b)
                    ds = newton * householder_factor(newton, halley, hh3)
                end
                ds = max(-0.5 * s, ds)
                s += ds
                iterations += 1
            end
            return s
        else
            v_l = normalised_vega(x, s_l)
            r_lm = convex_rational_cubic_control_parameter_to_fit_second_derivative_at_right_side(
                b_l,
                b_c,
                s_l,
                s_c,
                1 / v_l,
                1 / v_c,
                0.0,
                false,
            )
            s = rational_cubic_interpolation(beta, b_l, b_c, s_l, s_c, 1 / v_l, 1 / v_c, r_lm)
            s_left = s_l
            s_right = s_c
        end
    else

        s_h = s_c
        if v_c > floatmin()
            s_h = s_c + (b_max - b_c) / v_c
        end
        b_h = normalised_black_call(x, s_h)
        if beta <= b_h
            v_h = normalised_vega(x, s_h)
            r_hm = convex_rational_cubic_control_parameter_to_fit_second_derivative_at_left_side(
                b_c,
                b_h,
                s_c,
                s_h,
                1 / v_c,
                1 / v_h,
                0.0,
                false,
            )
            s = rational_cubic_interpolation(beta, b_c, b_h, s_c, s_h, 1 / v_c, 1 / v_h, r_hm)
            s_left = s_c
            s_right = s_h
        else
            f_upper_map_h, d_f_upper_map_h_d_beta, d2_f_upper_map_h_d_beta2 =
                compute_f_upper_map_and_first_two_derivatives(x, s_h)
            if d2_f_upper_map_h_d_beta2 > -SqrtMaxDouble && d2_f_upper_map_h_d_beta2 < SqrtMaxDouble
                r_hh = convex_rational_cubic_control_parameter_to_fit_second_derivative_at_left_side(
                    b_h,
                    b_max,
                    f_upper_map_h,
                    0.0,
                    d_f_upper_map_h_d_beta,
                    -0.5,
                    d2_f_upper_map_h_d_beta2,
                    true,
                )
                f = rational_cubic_interpolation(
                    beta,
                    b_h,
                    b_max,
                    f_upper_map_h,
                    0.0,
                    d_f_upper_map_h_d_beta,
                    -0.5,
                    r_hh,
                )
            end
            if f <= 0
                h = b_max - b_h
                t = (beta - b_h) / h
                f = (f_upper_map_h * (1 - t) + 0.5 * h * t) * (1 - t) # We switch to quadratic::Interpolation using f(b_h), f(b_max)≡0, and f'(b_max)≡-1/2 to specify the quadratic.
            end
            s = inverse_f_upper_map(f)
            s_left = s_h
            if beta > 0.5 * b_max  # Else we better drop through and let the objective function be g(s) = b(x,s)-beta.
                #
                # In this branch, which comprises the upper segment, the objective function is
                #     g(s) = ln(b_max-beta)-ln(b_max-b(x,s))
                #          ≡ ln((b_max-beta)/(b_max-b(s)))
                # This makes
                #              g'               =   b'/(b_max-b)
                #              newton = -g/g'   =   ln((b_max-b)/(b_max-beta))·(b_max-b)/b'
                #              halley = g''/g'  =   b''/b'  +  b'/(b_max-b)
                #              hh3    = g'''/g' =   b'''/b' +  g'·(2g'+3b''/b')
                # and the iteration is
                #     s_n+1  =  s_n  +  newton · [ 1 + halley·newton/2 ] / [ 1 + newton·( halley + hh3·newton/6 ) ].
                #
                while iterations < N && abs(ds) > eps() * s
                    #        println("2 ds ",ds," it ",iterations," s ",s)

                    if ds * ds_previous < 0
                        direction_reversal_count += 1
                    end
                    if iterations > 0 && (3 == direction_reversal_count || !(s > s_left && s < s_right))
                        # If looping inefficently, or the forecast step takes us outside the bracket, or onto its edges, switch to binary nesting.
                        # NOTE that this can only really happen for very extreme values of |x|, such as |x| = |ln(F/K)| > 500.
                        s = 0.5 * (s_left + s_right)
                        if s_right - s_left <= eps() * s
                            break
                        end
                        direction_reversal_count = 0
                        ds = 0
                    end
                    ds_previous = ds
                    b = normalised_black_call(x, s)
                    bp = normalised_vega(x, s)
                    if b > beta && s < s_right
                        s_right = s
                    elseif b < beta && s > s_left
                        s_left = s
                    end # Tighten the bracket if applicable.
                    if b >= b_max || bp <= floatmin()  # Numerical underflow. Switch to binary nesting for this iteration.
                        ds = 0.5 * (s_left + s_right) - s
                    else
                        b_max_minus_b = b_max - b
                        g = log((b_max - beta) / b_max_minus_b)
                        gp = bp / b_max_minus_b
                        b_halley = (x / s)^2 / s - s / 4
                        b_hh3 = b_halley * b_halley - 3 * (x / (s * s))^2 - 0.25
                        newton = -g / gp
                        halley = b_halley + gp
                        hh3 = b_hh3 + gp * (2 * gp + 3 * b_halley)
                        ds = newton * householder_factor(newton, halley, hh3)
                    end
                    ds = max(-0.5 * s, ds)
                    s += ds
                    iterations += 1
                end
                return s
            end
        end
    end

    # In this branch, which comprises the two middle segments, the objective function is g(s) = b(x,s)-beta, or g(s) = b(s) - beta, for short.
    # This makes
    #              newton = -g/g'   =  -(b-beta)/b'
    #              halley = g''/g'  =    b''/b'    =  x²/s³-s/4
    #              hh3    = g'''/g' =    b'''/b'   =  halley² - 3·(x/s²)² - 1/4
    # and the iteration is
    #     s_n+1  =  s_n  +  newton · [ 1 + halley·newton/2 ] / [ 1 + newton·( halley + hh3·newton/6 ) ].
    #

    while iterations < N && abs(ds) > eps() * s
        #    println("3 ds ",ds," it ",iterations," s ",s)

        if ds * ds_previous < 0
            direction_reversal_count += 1
        end

        if iterations > 0 && (3 == direction_reversal_count || !(s > s_left && s < s_right))
            # If looping inefficently, or the forecast step takes us outside the bracket, or onto its edges, switch to binary nesting.
            # NOTE that this can only really happen for very extreme values of |x|, such as |x| = |ln(F/K)| > 500.
            s = 0.5 * (s_left + s_right)
            if s_right - s_left <= eps() * s
                break
            end
            direction_reversal_count = 0
            ds = 0
        end
        ds_previous = ds
        b = normalised_black_call(x, s)
        bp = normalised_vega(x, s)
        if b > beta && s < s_right
            s_right = s
        elseif b < beta && s > s_left
            s_left = s
        end # Tighten the bracket if applicable.
        newton = (beta - b) / bp
        halley = ((x / s)^2) / s - s / 4
        hh3 = halley * halley - 3 * (x / (s * s))^2 - 0.25
        ds = max(-0.5 * s, newton * householder_factor(newton, halley, hh3))
        s += ds
        iterations += 1
    end
    return s
end

function implied_volatility_from_a_transformed_rational_guess_with_limited_iterations(
    price::Float64,
    F::Float64,
    K::Float64,
    T::Float64,
    q::Int,
    N::Int,
)::Float64
    intrinsic = abs(max(q * (F - K), 0.0))
    if price <= intrinsic
        return VOLATILITY_VALUE_TO_SIGNAL_PRICE_IS_BELOW_INTRINSIC
    end
    max_price = F
    if q < 0
        max_price = K
    end
    if price >= max_price
        return VOLATILITY_VALUE_TO_SIGNAL_PRICE_IS_ABOVE_MAXIMUM
    end
    x = log(F / K)
    # Map in-the-money to out-of-the-money
    if q * x > 0
        price = abs(max(price - intrinsic, 0.0))
        q = -q
    end
    if T < SqrtEpsilon
        #"Cannot solve for vol as time to expiry is %f", T)
        return 0
    end
    value = unchecked_normalised_implied_volatility_from_a_transformed_rational_guess_with_limited_iterations(
        price / (sqrt(F) * sqrt(K)),
        x,
        q,
        N,
    )
    return value / sqrt(T)
end

function implied_volatility_from_a_transformed_rational_guess(
    price::Float64,
    F::Float64,
    K::Float64,
    T::Float64,
    q::Int,
)::Float64
    return implied_volatility_from_a_transformed_rational_guess_with_limited_iterations(price, F, K, T, q, 2)
end

function normalised_implied_volatility_from_a_transformed_rational_guess_with_limited_iterations(
    beta::Float64,
    x::Float64,
    q::Int,
    N::Int,
)::Float64
    # Map in-the-money to out-of-the-money
    if q * x > 0
        beta -= normalised_intrinsic(x, q)
        q = -q
    end
    if beta < 0
        return VOLATILITY_VALUE_TO_SIGNAL_PRICE_IS_BELOW_INTRINSIC
    end
    return unchecked_normalised_implied_volatility_from_a_transformed_rational_guess_with_limited_iterations(
        beta,
        x,
        q,
        N,
    )
end

function normalised_implied_volatility_from_a_transformed_rational_guess(beta::Float64, x::Float64, q::Int)::Float64
    return normalised_implied_volatility_from_a_transformed_rational_guess_with_limited_iterations(beta, x, q, 2)
end

function impliedVolatility(isCall::Bool,
    optionPremiumOnAUnitNotional::Float64,
    forward::Float64,
    strike::Float64,
    timeToExpiry::Float64,
    discountFactorToPayDate::Float64,
)::Float64
    sign = -1
    if isCall
        sign = 1
    end
    return implied_volatility_from_a_transformed_rational_guess(
        optionPremiumOnAUnitNotional / discountFactorToPayDate,
        forward,
        strike,
        timeToExpiry,
        sign,
    )
end

using SpecialFunctions
function jaeckel_erfcx(d::Float64)::Float64
    return erfcx(d)
    # return erfcxCody(d) #original implementation
end

function jaeckel_normcdf(d::Float64)::Float64
    return normcdf(d)
    # return normcdfCody(d) #original imple
end

function jaeckel_norminv(f::Float64)::Float64
    return norminv(f)
    # return norminvAS241(f) #fails a test 1e-13 error
end
