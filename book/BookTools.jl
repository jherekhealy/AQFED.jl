# Shared tools for the figures of the book "Equity Derivatives" (~/mypapers/eqd_book).
#
# Conventions
#  - book/data holds the persisted inputs (text tables written by earlier studies, or by the compute_* functions).
#  - A figure is registered with `@figure "name" begin ... end`: the block returns a Makie Figure saved as name.pdf,
#    or a vector of "name" => Figure pairs when one study produces several PDFs. "name" is the PDF name used in
#    \includegraphics of the book.
#  - Expensive computations are registered with `@compute "name" begin ... end` and write CSV files to book/data.
#    Figures only read data files, so that restyling a figure never reruns a calibration or a Monte-Carlo simulation.
#  - Volatilities are shown in percent.
module BookTools

using CairoMakie, DataFrames, DelimitedFiles, CSV

export DATA_DIR, figureDir, datapath, readtable, readcsv, writetable, readmatrix, rows, relabel!,
    @figure, @compute, FIGURES, COMPUTATIONS, renderFigures, runComputations,
    bookFigure, bookAxis, referenceScatter!, groupLines!, groupBands!, bookLegend!, logTicks, PALETTE, emptiestCorner,
    surface3D!, facetLines, SURFMAP, hexc, C_GRID, C_QUOTE, C_RING, C_RIBBON, PLOTDATA_DIR, writePlotData

const DATA_DIR = joinpath(@__DIR__, "data")

#directory where the PDF files are written, EQD_BOOK_DIR overrides the default
figureDir() = get(ENV, "EQD_BOOK_DIR", expanduser("~/mypapers/eqd_book/figures"))

datapath(name::AbstractString) = joinpath(DATA_DIR, name)

#narrow the Any columns produced by readdlm
narrow(col) = identity.(col)

"""
    readtable(name; delim=' ', squeeze=false)

Reads a text table with a header line from book/data into a DataFrame. Columns are parsed as numbers when possible.
`squeeze=true` replaces runs of blanks by a single separator first (for files aligned with several spaces or tabs).
"""
function readtable(name::AbstractString; delim=' ', squeeze=delim == ' ')
    path = datapath(name)
    #trailing blanks (frequent in the files written by the R and Go studies) would create empty columns
    text = replace(read(path, String), r"[^\S\r\n]+(\r?\n|$)" => s"\1")
    m = if squeeze
        readdlm(IOBuffer(replace(text, r"[^\S\r\n]+" => ";")), ';', header=true)
    else
        readdlm(IOBuffer(text), delim, header=true)
    end
    df = DataFrame(m[1], string.(strip.(string.(vec(m[2])))); makeunique=true)
    for c in names(df)
        df[!, c] = narrow(df[!, c])
    end
    return df
end

#a numeric matrix without header
function readmatrix(name::AbstractString; delim=' ')
    m = readdlm(datapath(name), delim)
    return [x isa Number ? Float64(x) : NaN for x in m]
end

writetable(name::AbstractString, df) = CSV.write(datapath(name), df)
readcsv(name::AbstractString) = CSV.read(datapath(name), DataFrame)

"rows of df where column col takes one of the values"
rows(df, col, values...) = df[in(values).(df[!, col]), :]

"rename the values of a column, e.g. relabel!(df, :Method, \"TRBDF2-LV\" => \"Analytic\")"
function relabel!(df, col, pairs::Pair...)
    d = Dict(pairs...)
    df[!, col] = [get(d, v, v) for v in df[!, col]]
    return df
end

## figure helpers

hexc(s) = parse(Makie.RGBA{Float32}, s)
#Okabe-Ito colors (safe for color vision deficiencies), in the order of the iv3 / cvi_rbf notes: blue, vermillion, green,
#reddish purple, orange, sky blue, yellow
const PALETTE = hexc.(["#0072B2", "#D55E00", "#009E73", "#CC79A7", "#E69F00", "#56B4E9", "#F0E442"])
const C_GRID = hexc("#D9D9D9")
const C_AXIS = hexc("#333333")
const C_RIBBON, C_RING, C_QUOTE = hexc("#9E9E9E"), hexc("#5A5A5A"), hexc("#3A3A3A")
#3D surfaces: viridis without its dark purple end, faint sparse mesh
const SURFMAP = cgrad(collect(cgrad(:viridis)[0.28:0.01:1.0]))

#Style of the iv3 and cvi_rbf notes, in IBM Plex Sans as the book. Sizes are in Makie units; saveFigure lays the figures out
#such that one unit prints at 0.5pt in the book: tick labels and legends 7.5pt, axis labels and titles 8.5pt, lines 1pt,
#solid light grey grid, no top/right spines.
const PLEX_DIR = "/usr/share/fonts/truetype"
plexFont(style) = (path = joinpath(PLEX_DIR, "IBMPlexSans-$style.otf"); isfile(path) ? path : "IBM Plex Sans $style")

const BOOK_THEME = Theme(
    fontsize=15,
    fonts=(regular=plexFont("Regular"), bold=plexFont("Medium"), italic=plexFont("Italic"), bold_italic=plexFont("MediumItalic")),
    palette=(color=PALETTE,),
    Axis=(xgridvisible=true, ygridvisible=true, xgridcolor=C_GRID, ygridcolor=C_GRID, xgridwidth=1.2, ygridwidth=1.2,
        xminorgridvisible=false, yminorgridvisible=false, topspinevisible=false, rightspinevisible=false, spinewidth=1.6,
        leftspinecolor=C_AXIS, bottomspinecolor=C_AXIS, xtickcolor=C_AXIS, ytickcolor=C_AXIS,
        xticksize=6, yticksize=6, xtickwidth=1.6, ytickwidth=1.6, xlabelsize=17, ylabelsize=17, xticklabelsize=15, yticklabelsize=15,
        titlesize=17, titlefont=:regular, titlegap=8, xlabelpadding=4, ylabelpadding=6),
    Axis3=(titlesize=17, titlefont=:regular, xlabelsize=17, ylabelsize=17, zlabelsize=17, xticklabelsize=15, yticklabelsize=15, zticklabelsize=15,
        xgridcolor=C_GRID, ygridcolor=C_GRID, zgridcolor=C_GRID, xgridwidth=1.0, ygridwidth=1.0, zgridwidth=1.0,
        xspinewidth=1.6, yspinewidth=1.6, zspinewidth=1.6, xlabeloffset=50, ylabeloffset=50, zlabeloffset=60),
    Colorbar=(labelsize=17, ticklabelsize=15, spinewidth=1.2, ticksize=6, width=14),
    Lines=(linewidth=2.0,),
    Scatter=(markersize=8, strokewidth=0),
    Legend=(framevisible=false, backgroundcolor=(:white, 0.85), labelsize=15, titlesize=15, titlefont=:bold, rowgap=0,
        patchsize=(28, 12), patchlabelgap=6, padding=(6, 6, 4, 4), margin=(4, 4, 4, 4)),
)

__init__() = set_theme!(BOOK_THEME)

bookFigure(; size=(400, 300)) = Figure(size=size)

bookAxis(pos; xlabel="", ylabel="", kwargs...) = Axis(pos; xlabel=xlabel, ylabel=ylabel, kwargs...)

"""
    referenceScatter!(ax, x, y; label="Reference", dense=length(x) >= 40, marker=:circle)

Market (reference) quotes, never hidden behind the model curves (style of the sanos_practice and cvi_rbf notes):
dense quotes as a light grey ribbon through them, drawn under the curves (a curve leaving the quotes leaves the ribbon);
sparse quotes as hollow grey rings drawn above the curves. Crosses (`marker=:cross`) are kept for exact reference points.
"""
function referenceScatter!(ax, x, y; label="Reference", dense=length(x) >= 40, marker=:circle, markersize=nothing, color=nothing, kwargs...)
    if dense && marker == :circle
        idx = sortperm(x)
        p = lines!(ax, x[idx], y[idx]; color=(something(color, C_RIBBON), 0.5), linewidth=6, linecap=:round, joinstyle=:round, label=label)
        translate!(p, 0, 0, -1)
    elseif marker in (:cross, :xcross, :+, :x)
        p = scatter!(ax, x, y; label=label, color=something(color, C_QUOTE), marker=marker, markersize=something(markersize, 10), kwargs...)
        translate!(p, 0, 0, 10)
    else
        p = scatter!(ax, x, y; label=label, color=:transparent, strokecolor=something(color, C_RING), strokewidth=1.8,
            marker=marker, markersize=something(markersize, 10), kwargs...)
        translate!(p, 0, 0, 10)
    end
    return p
end

"""
    facetLines(df, x, y, facet; order, ncol, xlabel, ylabel, size, colors=true, xticks, ylims, xlims, linkx=true, linky=true)

One small panel per value of the column `facet` (as ggplot facet_wrap), titled by the value, with linked axes.
"""
function facetLines(df, x, y, facet; order=unique(df[!, facet]), ncol=length(order), xlabel="", ylabel="", size=(600, 300),
    colors=true, xticks=Makie.automatic, yticks=Makie.automatic, ylims=nothing, xlims=nothing, linkx=true, linky=true)
    fig = bookFigure(size=size)
    nrow = cld(length(order), ncol)
    axs = Axis[]
    for (i, g) in enumerate(order)
        r, c = fldmod1(i, ncol)
        d = df[df[!, facet].==g, :]
        d = d[sortperm(d[!, x]), :]
        ax = bookAxis(fig[r, c], title=string(g), xlabel=r == nrow ? xlabel : "", ylabel=c == 1 ? ylabel : "", xticks=xticks, yticks=yticks)
        lines!(ax, d[!, x], d[!, y], color=colors ? PALETTE[mod1(i, length(PALETTE))] : PALETTE[1])
        xlims === nothing || xlims!(ax, xlims...)
        ylims === nothing || ylims!(ax, ylims...)
        c > 1 && linky && hideydecorations!(ax, grid=false, ticks=false)
        r < nrow && linkx && hidexdecorations!(ax, grid=false, ticks=false)
        push!(axs, ax)
    end
    linkx && linkxaxes!(axs...)
    linky && linkyaxes!(axs...)
    return fig
end

"""
    surface3D!(pos, xs, ys, Z; title, xlabel, ylabel, zlabel, zrange, azimuth, elevation, nmesh, protrusions)

3D surface panel in the style of the cvi_rbf note: lighter viridis, faint sparse mesh, rasterized fill (vector axes and text).
Values are clamped to `zrange` when given. NaN values are left transparent.
"""
function surface3D!(pos, xs, ys, Z; title="", xlabel="", ylabel="", zlabel="", zrange=nothing, azimuth=1.75π, elevation=0.16π,
    nmesh=10, protrusions=(80, 40, 16, 40), kwargs...)
    ax = Axis3(pos; xlabel=xlabel, ylabel=ylabel, zlabel=zlabel, title=title, azimuth=azimuth, elevation=elevation, viewmode=:fitzoom,
        protrusions=protrusions, kwargs...)
    Zc = zrange === nothing ? Z : clamp.(Z, zrange...)
    finite = filter(isfinite, Zc)
    cr = zrange === nothing ? (isempty(finite) ? (0, 1) : extrema(finite)) : zrange
    surface!(ax, xs, ys, Zc, colormap=SURFMAP, colorrange=cr, shading=NoShading, nan_color=:transparent, rasterize=4)
    sx, sy = max(1, length(xs) ÷ nmesh), max(1, length(ys) ÷ nmesh)
    wireframe!(ax, xs[1:sx:end], ys[1:sy:end], Zc[1:sx:end, 1:sy:end], color=(:black, 0.22), linewidth=0.7)
    zrange === nothing || zlims!(ax, zrange...)
    return ax
end

"""
    groupLines!(ax, x, y, group; order=unique(group), labels=Dict(), colors=Dict(), linestyles=false, sortx=true, offset=0)

One line per value of `group` (as `group=` in Plots.jl). Each group is sorted by x unless sortx=false.
Colors follow the order of the groups, starting at PALETTE[1+offset].
"""
function groupLines!(ax, x, y, group; order=unique(group), labels=Dict(), colors=Dict(), linestyles=false, sortx=true, offset=0, kwargs...)
    styles = [:solid, :dash, :dot, :dashdot, :dashdotdot]
    for (i, g) in enumerate(order)
        idx = findall(==(g), group)
        isempty(idx) && continue
        sortx && (idx = idx[sortperm(x[idx])])
        lines!(ax, x[idx], y[idx]; label=string(get(labels, g, g)),
            color=get(colors, g, PALETTE[mod1(i + offset, length(PALETTE))]),
            linestyle=linestyles ? styles[mod1(i, length(styles))] : :solid, kwargs...)
    end
end

"confidence bands (y ± w) per group, to be drawn before groupLines! with the same order"
function groupBands!(ax, x, y, w, group; order=unique(group), colors=Dict(), offset=0, alpha=0.15)
    for (i, g) in enumerate(order)
        idx = findall(==(g), group)
        isempty(idx) && continue
        idx = idx[sortperm(x[idx])]
        c = get(colors, g, PALETTE[mod1(i + offset, length(PALETTE))])
        band!(ax, x[idx], y[idx] .- w[idx], y[idx] .+ w[idx]; color=(c, alpha))
    end
end

bookLegend!(ax; title=nothing, position=:rt, kwargs...) =
    title === nothing ? axislegend(ax; position=position, kwargs...) :
    axislegend(ax, title; position=position, kwargs...)

"""
    emptiestCorner(x, y; logy=false, width=0.45, height=0.3)

Legend position (:lt, :rt, :lb, :rb, :lc, :rc, :ct) whose box, of the given fraction of the axis width and height, holds the
fewest of the data points (x, y): the legend then hides as little of the curves as possible.
"""
function emptiestCorner(x, y; logy=false, width=0.45, height=0.3)
    yy = logy ? log10.(y) : y
    xmin, xmax = extrema(x)
    ymin, ymax = extrema(yy)
    u = (x .- xmin) ./ (xmax - xmin)
    v = (yy .- ymin) ./ (ymax - ymin)
    boxes = Dict(:lt => (0, width, 1 - height, 1), :rt => (1 - width, 1, 1 - height, 1), :ct => (0.5 - width / 2, 0.5 + width / 2, 1 - height, 1),
        :lb => (0, width, 0, height), :rb => (1 - width, 1, 0, height),
        :lc => (0, width, 0.5 - height / 2, 0.5 + height / 2), :rc => (1 - width, 1, 0.5 - height / 2, 0.5 + height / 2))
    order = [:rt, :lt, :rc, :lc, :ct, :rb, :lb] #preference when several boxes are empty
    count_in(b) = count(i -> b[1] <= u[i] <= b[2] && b[3] <= v[i] <= b[4], eachindex(u))
    return order[argmin([count_in(boxes[p]) for p in order])]
end

"ticks at the given values with custom labels, e.g. logTicks([1/32,1/16],[\"1/32\",\"1/16\"])"
logTicks(values, labels) = (values, labels)

## registries

const FIGURES = Dict{String,Function}()
const COMPUTATIONS = Dict{String,Function}()

macro figure(name, body)
    return :(FIGURES[$(esc(name))] = () -> $(esc(body)))
end

macro compute(name, body)
    return :(COMPUTATIONS[$(esc(name))] = () -> $(esc(body)))
end

#page size in points of the original (pre-Makie) PDF of each figure, so that the book layout does not change
const FIGURE_SIZES = let df = CSV.read(datapath("figure_sizes.csv"), DataFrame)
    Dict(String(r.name) => (Float64(r.width_pt), Float64(r.height_pt)) for r in eachrow(df))
end

#book layout: width of the text block of eqd_book (7x10in SIAM class) and printed size of one Makie unit, so that
#one Makie unit prints at 0.5pt whatever the figure size and the width it is included at.
const BOOK_DIR = get(ENV, "EQD_BOOK_TEX", expanduser("~/mypapers/eqd_book"))
const TEXTWIDTH_PT = 383.5
const PRINTED_PT_PER_UNIT = 0.5
const MIN_LAYOUT_UNITS = 330
#other documents including figures of this registry, with their text width in points: the note on lognormal mixtures
#(article, 11pt, 1in margins on letter paper)
const EXTRA_SOURCES = [(expanduser("~/mypapers/lnmixture/lnmixture.tex"), 469.75)]

"""
    includeWidths()

Printed width in points of each figure, from the \\includegraphics of the book sources (the smallest if included several times).
"""
function includeWidths()
    widths = Dict{String,Float64}()
    files = [(joinpath(BOOK_DIR, d, d * ".tex"), TEXTWIDTH_PT) for d in readdir(BOOK_DIR) if occursin(r"^(chapter|appendix)-\d+$|^preface$", d)]
    for (file, textwidth) in filter(f -> isfile(f[1]), vcat(files, EXTRA_SOURCES))
        parseIncludeWidths!(widths, read(file, String), textwidth)
    end
    return widths
end

#adds to `widths` the printed width of each \includegraphics of the LaTeX `text` (commented lines skipped, file name possibly
#spread over several lines), keeping the smallest width of a figure included several times
function parseIncludeWidths!(widths, text, textwidth; sizes=FIGURE_SIZES)
    text = replace(text, r"(?<!\\)%[^\n]*" => "")
    for m in eachmatch(r"\\includegraphics\[([^\]]*)\]\{([^}]+)\}", text)
        opts, name = m.captures[1], replace(basename(strip(m.captures[2])), r"\.pdf$" => "")
        w = nothing
        if (mw = match(r"width\s*=\s*([\d.]*)\s*\\(textwidth|linewidth)", opts)) !== nothing
            w = (isempty(mw.captures[1]) ? 1.0 : parse(Float64, mw.captures[1])) * textwidth
        elseif (mw = match(r"width\s*=\s*([\d.]+)\s*cm", opts)) !== nothing
            w = parse(Float64, mw.captures[1]) * 72.27 / 2.54
        elseif (mh = match(r"height\s*=\s*([\d.]+)\s*cm", opts)) !== nothing && haskey(sizes, name)
            ref = sizes[name]
            w = parse(Float64, mh.captures[1]) * 72.27 / 2.54 * ref[1] / ref[2]
        end
        w === nothing || (widths[name] = min(get(widths, name, Inf), w))
    end
    return widths
end
const INCLUDE_WIDTHS = Ref{Union{Nothing,Dict{String,Float64}}}(nothing)
function includeWidth(name)
    INCLUDE_WIDTHS[] === nothing && (INCLUDE_WIDTHS[] = includeWidths())
    return get(INCLUDE_WIDTHS[], name, nothing)
end

"""
    saveFigure(name, fig; dir=figureDir())

Saves `fig` as name.pdf. When the figure has a reference size in book/data/figure_sizes.csv (the page size in points of
the original Plots.jl figure), it is written with that page size, so that the page layout of the book does not change, and laid
out such that its fonts, lines and markers print at the same size in the book whatever the width it is included at.
"""
function saveFigure(name::AbstractString, fig; dir=figureDir())
    mkpath(dir)
    base = endswith(name, ".pdf") ? name[1:end-4] : name
    path = joinpath(dir, base * ".pdf")
    #figures without an original use their own size as reference
    ref = get(FIGURE_SIZES, base, Tuple(Float64.(fig.scene.viewport[].widths)))
    begin
        #lay out at the size whose units print at PRINTED_PT_PER_UNIT once included in the book, write at the reference page size
        printed = something(includeWidth(base), ref[1])
        #small panels (less than about a third of the text width) are laid out wider, so their text prints a bit smaller
        w = max(printed / PRINTED_PT_PER_UNIT, MIN_LAYOUT_UNITS)
        resize!(fig, round(Int, w), round(Int, w * ref[2] / ref[1]))
        save(path, fig; pt_per_unit=ref[1] / round(Int, w))
    end
    writePlotData(base, fig)
    return path
end

## record of the numbers drawn by each figure

const PLOTDATA_DIR = joinpath(@__DIR__, "plotdata")

numericArg(v) = v isa Real ? [Float64(v)] : v isa AbstractRange || v isa AbstractVector{<:Real} ? collect(Float64, v) :
    v isa AbstractMatrix{<:Real} ? Matrix{Float64}(v) : v isa AbstractVector{<:Makie.VecTypes} ? reduce(hcat, [collect(Float64, q) for q in v])' : nothing
plotArgs(p) = [numericArg(getproperty(p, Symbol("arg", k))[]) for k in 1:3 if hasproperty(p, Symbol("arg", k))]

"""
    writePlotData(name, fig)

Writes the numbers drawn by `fig` to book/plotdata/name.csv (one row per point, columns panel, plot, kind, label, x, y, z), so
that every published figure can be checked or redrawn without rerunning the computations. Curves, markers, bands (y = lower,
z = upper), reference lines and surfaces/heatmaps (on their grid) are recorded, text and decorations are not.
"""
function writePlotData(name, fig)
    rows = DataFrame(panel=Int[], plot=Int[], kind=String[], label=String[], x=Float64[], y=Float64[], z=Float64[])
    blocks = [b for b in fig.content if b isa Axis || b isa Axis3]
    for (ia, ax) in enumerate(blocks), (ip, p) in enumerate(ax.scene.plots)
        kind = string(Makie.plotfunc(p))
        kind in ("text", "text!", "poly", "mesh") && continue
        args = filter(!isnothing, try plotArgs(p) catch; [] end)
        isempty(args) && continue
        label = hasproperty(p, :label) && p.label[] !== nothing ? string(p.label[]) : ""
        push3!(x, y, z) = push!(rows, (ia, ip, kind, label, x, y, z))
        if length(args) == 3 && args[3] isa AbstractMatrix && args[1] isa AbstractVector && args[2] isa AbstractVector
            for j in eachindex(args[2]), i in eachindex(args[1])
                push3!(args[1][i], args[2][j], args[3][i, j])
            end
        elseif length(args) == 1 && args[1] isa AbstractMatrix
            M = args[1]
            for i in axes(M, 1)
                push3!(M[i, 1], size(M, 2) > 1 ? M[i, 2] : NaN, size(M, 2) > 2 ? M[i, 3] : NaN)
            end
        elseif all(a -> a isa AbstractVector, args)
            n = maximum(length, args)
            get1(a, i) = length(a) == n ? a[i] : length(a) == 1 ? a[1] : NaN
            for i in 1:n
                push3!(get1(args[1], i), length(args) > 1 ? get1(args[2], i) : NaN, length(args) > 2 ? get1(args[3], i) : NaN)
            end
        end
    end
    mkpath(PLOTDATA_DIR)
    CSV.write(joinpath(PLOTDATA_DIR, name * ".csv"), rows)
end

"""
    renderFigures(pattern=r""; dir=figureDir())

Renders the registered figures whose name matches `pattern` and writes the PDFs to `dir`. Returns the written paths.
"""
function renderFigures(pattern=r""; dir=figureDir())
    written = String[]
    for name in sort(collect(keys(FIGURES)))
        occursin(pattern, name) || continue
        result = FIGURES[name]()
        figs = result isa AbstractVector ? result : [name => result]
        for (n, f) in figs
            push!(written, saveFigure(n, f; dir=dir))
        end
    end
    return written
end

function runComputations(pattern=r"")
    for name in sort(collect(keys(COMPUTATIONS)))
        occursin(pattern, name) || continue
        @info "computing $name"
        COMPUTATIONS[name]()
    end
end

end
