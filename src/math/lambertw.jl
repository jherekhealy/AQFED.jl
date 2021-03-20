#Asymptotic series for the branch 0 of the Lambert W function from Darko Veberic "Having Fun with Lambert W ( x ) Function" (2009)
function lambertWAsymptotic(x::Number)
    a = log(x)
    b = log(a)
    ia = 1 / a
    return a - b +
           b / a * (
        1 +
        ia * (
            0.5 * (-2 + b) +
            ia * (
                (6 + b * (-9 + b * 2)) / 6 +
                ia * (
                    (-12 + b * (36 + b * (-22 + b * 3))) / 12 +
                    ia / 60 * (60 + b * (-300 + b * (350 + b * (-125 + b * 12))))
                )
            )
        )
    )
end

# branch 0, valid for [0.3,7]
@inline lambertWRationalApproximation3(x) = x * (
        1 +
        x * (
            2.4450530707265568 +
            x *
            (1.3436642259582265 + x * (0.14844005539759195 + x * 0.0008047501729129999))
        )
    ) / (
        1 +
        x * (
            3.4447089864860025 +
            x * (3.2924898573719523 + x * (0.9164600188031222 + x * 0.05306864044833221))
        )
    )

# branch 0, valid for [-0.31,0.3]
@inline lambertWRationalApproximation1(x) =
    x * (
        1 +
        x * (
            5.931375839364438 +
            x * (11.392205505329132 + x * (7.338883399111118 + x * 0.6534490169919599))
        )
    ) / (
        1 +
        x * (
            6.931373689597704 +
            x * (16.82349461388016 + x * (16.43072324143226 + x * 5.115235195211697))
        )
    )


#Branch 0 of the Lambert W function
#algorithm follows Darko Veberic "Having Fun with Lambert W ( x ) Function" (2009)
#Fritsch's iteration on top of asymptotic series - machine accuracy for x>=8.70 (practical range for tanhsinh quadrature)
#Fritsch on top Rational approximation for x < 8.70 to reach machine accuracy in this region.
function lambertW(x::Float64)::Float64
    local wn
    if x < 0.14546954290661823
        wn = lambertWRationalApproximation1(x)
    elseif x < 8.706658967856612
        wn = lambertWRationalApproximation3(x)
    else
        wn = lambertWAsymptotic(x)
    end
    zn = log(x / wn) - wn
    qn = 2 * (1 + wn) * (1 + wn + 2 * zn / 3)
    en = (zn / (1 + wn)) * ((qn - zn) / (qn - 2 * zn))
    return wn * (1 + en)
end
