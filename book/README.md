# Figures of the book *Equity Derivatives*

Code and data behind the figures of the book in `~/mypapers/eqd_book`. The PDF files are written to the `figures` folder of
the book (`$EQD_BOOK_DIR`, default `~/mypapers/eqd_book/figures`, found by LaTeX through `\graphicspath{{figures/}}`),
never to this repository. The former PDF files of the book replaced by generated ones are kept in `eqd_book/figures_old`.

## Usage

```sh
julia --project=book book/run.jl --list                 # registered figures and computations
julia --project=book book/run.jl 'aapl_1m'              # render the figures whose name matches the regex
julia --project=book -t 8 book/run.jl --compute 'smile' # rerun the computations whose name matches, then render
EQD_BOOK_DIR=/tmp/check julia --project=book book/run.jl  # render everything elsewhere, e.g. to compare
julia book/figure_index.jl                               # book/FIGURES.md: figure label -> PDF -> code
```

## Layout

| Path | Content |
|---|---|
| `BookTools.jl` | paths, `readtable` for the persisted text tables, Makie theme, `@figure` / `@compute` registries, plot helpers |
| `chNN_<topic>.jl` | figures (`@figure "pdf name"`) and computations (`@compute "name"`) of chapter NN |
| `smile_methods.jl` | smile interpolation methods of chapter 7 behind a common interface (price, density, implied vol) |
| `data/` | persisted inputs of the figures: tables of earlier studies, and the CSV files written by the computations |
| `legacy/` | the former notebooks (`eqd_book.jl`, studies formerly in `test/`), kept for reference |
| `legacy/rqmc_*_study.jl`, `legacy/slv_*.jl` | former stand-alone experiments without a figure of the book: Brownian bridge and PCA on the GBM (it does not work with the Black model, a single Gaussian is used), Asian sequences, Hok and Kucherenko reproduction, SLV drafts. Not updated for the current API |
| `FIGURES.md` | generated index of the book figures and of the code producing them |
| `../test/data/smiles`, `../test/smile_data.jl` | market smiles shared by the unit tests and the figures |

## Conventions

- A figure only reads files of `data/`: restyling it never reruns a calibration or a Monte-Carlo simulation.
- Expensive studies are `@compute` blocks writing CSV files; their parameters are in the code, next to the figure.
- The figure name is the PDF name of `\includegraphics`; `figure_index.jl` links it to the `\label` of the book.
- Plots use CairoMakie and the theme of `BookTools.jl`; volatilities are shown in percent.

## Figures that can not be regenerated, or differ from the published ones

The published PDF is kept at the root of the book folder (it takes precedence over `figures/`) for:

- `deelstra_vol100_err.pdf`: the table with the Deelstra FA1/FA2 errors was never persisted.
- Fengler spline of `jaeckel_case_i_fengler_rbf*.pdf`: the quadratic program has no solution with the current solvers on this example.
- `gocsei_10y.pdf`, `zhang_extreme.pdf`: published with models (Deelstra, VL2, VL3) whose results were never persisted.
- `jaeckel_case_ii_bspl_{vol,dens}.pdf`: the cubic B-spline collocation now produces a different, spiky, density.
- `digital_impliedvol.pdf`, `dupire_labordere_liquidator_localvol{,_log}.pdf`, `two_sabr_1_smile.pdf`: the published curves do not
  follow from the notebook code or data (mislabelled curves); the generated versions follow the data.

Option A of the lognormal mixture now ends in a different local minimum on case II: the book uses the generated
`jaeckel_case_ii_mixture_{vol,dens}.pdf`, and the published files are kept as `jaeckel_case_ii_mixture_{vol,dens}_old.pdf`.

The product integral of `burley_directions` was computed before the half of `2^-32` of the Burley and modulo points: new runs differ from
`data/burley_owen_raw_results_*.csv` and `data/rqmc_convergence_results_m*.csv` by at most about `1e-7`, the other sequences are reproduced to rounding.
