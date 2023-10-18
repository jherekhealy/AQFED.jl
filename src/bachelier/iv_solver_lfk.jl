export bachelierImpliedVolatility
const alpha4 = 0.15
const aLFK4n = 7
const aLFK4d = 5


const aLFK4 = [
    0.06155371425063157,
    2.723711658728403,
    10.83806891491789,
    301.0827907126612,
    1082.864564205999,
    790.7079667603721,
    109.330638190985,
    0.1515726686825187,
    1.436062756519326,
    118.6674859663193,
    441.1914221318738,
    313.4771127147156,
    40.90187645954703,
]

const cLFK4n = 9
const cLFK4d = 7


const bLFK4 = [
    0.6409168551974357,
    788.5769356915809,
    445231.8217873989,
    149904950.4316367,
    32696572166.83277,
    4679633190389.852,
    420159669603232.9,
    2.053009222143781e+16,
    3.434507977627372e+17,
    2.012931197707014e+16,
    644.3895239520736,
    211503.4461395385,
    42017301.42101825,
    5311468782.258145,
    411727826816.0715,
    17013504968737.03,
    247411313213747.3,
]
const cLFK4 = [
    0.6421106629595358,
    654.5620600001645,
    291531.4455893533,
    69009535.38571493,
    9248876215.120627,
    479057753706.175,
    9209341680288.471,
    61502442378981.76,
    107544991866857.5,
    63146430757.94501,
    437.9924136164148,
    90735.89146171122,
    9217405.224889684,
    400973228.1961834,
    7020390994.356452,
    44654661587.93606,
    76248508709.85633,
]
const dLFK4 = [
    0.936024443848096,
    328.5399326371301,
    177612.3643595535,
    8192571.038267588,
    110475347.0617102,
    545792367.0681282,
    1033254933.287134,
    695066365.5403566,
    123629089.1036043,
    756.3653755877336,
    173.9755977685531,
    6591.71234898389,
    82796.56941455391,
    396398.9698566103,
    739196.7396982114,
    493626.035952601,
    87510.31231623856,
]

const betaStart4 = -log(alpha4)
const betaEnd = 708.3964185322641


@inline function eta(u::AbstractFloat)::AbstractFloat
    if u < 0.01
        return 1 -
               u * (
            0.5 +
            u *
            (1.0 / 12 + u * (1.0 / 24 + u * (19.0 / 720 + u * (3.0 / 160 + u * (863.0 / 60480 + u * 275.0 / 24192)))))
        )
    end
    return -u / log1p(-u)
end
@inline function zofy(z::AbstractFloat, betaStart::AbstractFloat, betaEnd::AbstractFloat)::AbstractFloat
    return -(log(z) + betaStart) / (betaEnd - betaStart)
end

#
# @inline function ratval(cofs::Vector{Float64}, nn::Int, dd::Int, x::Float64)::Float64
# 	sumn = cofs[nn]
# 	sumd = 0.0
# 	for j = nn - 1:-1:1
# 		sumn = sumn*x + cofs[j]
# 	end
# 	for j = nn + dd - 1:-1:nn+1
# 		sumd = (sumd + cofs[j]) * x
# 	end
# 	return sumn / (1.0 + sumd)
# end

function bachelierImpliedVolatility(
    price::Number,
    isCall::Bool,
    strike::Number,
    tte::Number,
    forward::Number,
    discountDf::Number,
)::Number
    sign = 1
    if !isCall
        sign = -1
    end
    x = (forward - strike) * sign
    if abs(x) < eps(forward)
        return price / discountDf * sqrt(2 * pi / tte)
    end
    z = if x > 0
        (price / discountDf - x) / x
    else
        -price / (discountDf * x)
    end

    if z < alpha4 #highly OTM options up to 1e-100
        
        u = zofy(z, betaStart4, betaEnd)
        hz = if u < 0.0091
            num =
                bLFK4[1] +
                u * (
                    bLFK4[2] +
                    u * (
                        bLFK4[3] +
                        u * (
                            bLFK4[4] +
                            u * (
                                bLFK4[5] +
                                u * (bLFK4[6] + u * (bLFK4[7] + u * (bLFK4[8] + u * (bLFK4[9] + u * bLFK4[10]))))
                            )
                        )
                    )
                )
            den =
                1.0 +
                u * (
                    bLFK4[11] +
                    u * (
                        bLFK4[12] +
                        u * (bLFK4[13] + u * (bLFK4[14] + u * (bLFK4[15] + u * (bLFK4[16] + u * (bLFK4[17])))))
                    )
                )
                # println("bachelier iv solver, putprice, ",strike, " ",u, " ",num," ",den)
    
                num / den
        elseif u < 0.088
            num =
                cLFK4[1] +
                u * (
                    cLFK4[2] +
                    u * (
                        cLFK4[3] +
                        u * (
                            cLFK4[4] +
                            u * (
                                cLFK4[5] +
                                u * (cLFK4[6] + u * (cLFK4[7] + u * (cLFK4[8] + u * (cLFK4[9] + u * cLFK4[10]))))
                            )
                        )
                    )
                )
            den =
                1.0 +
                u * (
                    cLFK4[11] +
                    u * (
                        cLFK4[12] +
                        u * (cLFK4[13] + u * (cLFK4[14] + u * (cLFK4[15] + u * (cLFK4[16] + u * (cLFK4[17])))))
                    )
                )
            num / den
        else
            num =
                dLFK4[1] +
                u * (
                    dLFK4[2] +
                    u * (
                        dLFK4[3] +
                        u * (
                            dLFK4[4] +
                            u * (
                                dLFK4[5] +
                                u * (dLFK4[6] + u * (dLFK4[7] + u * (dLFK4[8] + u * (dLFK4[9] + u * dLFK4[10]))))
                            )
                        )
                    )
                )
            den =
                1.0 +
                u * (
                    dLFK4[11] +
                    u * (
                        dLFK4[12] +
                        u * (dLFK4[13] + u * (dLFK4[14] + u * (dLFK4[15] + u * (dLFK4[16] + u * (dLFK4[17])))))
                    )
                )
            num / den
        end
        return abs(x) / sqrt(hz * tte)
    else
        putPrice = if x <= 0
            price/discountDf - x
        else
            price/discountDf
        end
        z = abs(x) / putPrice
        u = eta(z)
        num =
            aLFK4[1] +
            u * (
                aLFK4[2] +
                u * (aLFK4[3] + u * (aLFK4[4] + u * (aLFK4[5] + u * (aLFK4[6] + u * (aLFK4[7] + u * (aLFK4[8]))))))
            )
        den = 1.0 + u * (aLFK4[9] + u * (aLFK4[10] + u * (aLFK4[11] + u * (aLFK4[12] + u * (aLFK4[13])))))
        hz = num / den

        return putPrice * hz / sqrt(tte)
    end
end
