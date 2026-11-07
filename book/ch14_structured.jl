# Chapter 14: structured products. Autocallables, Altiplano.

@figure "altiplano_eigenvalues" begin
    c1 = [0.11106295619505602, 0.18972542661377423, 0.2464735118199204, 0.3489278865668937, 0.43301516887262687, 0.4488071836403987, 0.628673634428382, 0.677395457830409, 2.3174466153150517, 4.59847215871749]
    c2 = [0.03270750048573754, 0.07136055122094749, 0.08816331196074648, 0.16747855944850265, 0.17878925736728732, 0.2990924188266631, 1.0694235527066924, 1.9252298518436854, 2.428959853333082, 3.7387951428066564]
    fig = bookFigure(size=(800, 400))
    ax = bookAxis(fig[1, 1], ylabel="Eigenvalue")
    lines!(ax, 1:10, c1, label="C1")
    lines!(ax, 1:10, c2, label="C2")
    bookLegend!(ax, position=:lt)
    fig
end

function altiplanoFigure(file)
    df = readtable(file)
    fig = bookFigure()
    ax = bookAxis(fig[1, 1], xlabel="Mixing parameter (λ)", ylabel="Price in %")
    groupLines!(ax, df.Correlation, df.Price, df.Model)
    ylims!(ax, 2.25, 3.25)
    bookLegend!(ax)
    return fig
end
@figure "altiplano_heston" altiplanoFigure("altiplano_heston.txt")
@figure "altiplano_heston_b" altiplanoFigure("altiplano_heston_b.txt")

function autocallFigure(file, x, xlabel)
    df = readtable(file)
    fig = bookFigure()
    ax = bookAxis(fig[1, 1], xlabel=xlabel, ylabel="Price in %")
    groupLines!(ax, df[!, x], df.Price .* 100, df.Model)
    bookLegend!(ax)
    return fig
end
@figure "holee_autocall" autocallFigure("holee_autocall.txt", :Correlation, "Correlation")
@figure "heston_autocall" autocallFigure("heston_autocall.txt", :Strike, "Strike")
