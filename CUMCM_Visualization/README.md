# CUMCM Visualization

## V2 engineering upgrade

V2 preserves every V1 call and adds a visual theme, safe statistics, layout,
export and quality-check modules. The recommended pipeline is raw data → model
evaluation → CUMCM figure → diagnostics → PDF/SVG/600-dpi PNG → Word/LaTeX.

```python
from CUMCM_Style import cumcm_model_compare, cumcm_convergence, cumcm_pareto
from CUMCM_Statistics import r2_score, rmse, mae

cumcm_model_compare(["Linear", "RF", "XGBoost"], [.79, .86, .90],
                    metric="$R^2$", direction="max")
```

For lower-is-better metrics, use `direction="min"` (RMSE, MAE, MAPE, AIC,
BIC). The original `higher_is_better=False` remains supported. New figure APIs:
`cumcm_metric_compare`, `cumcm_residual_diagnostics`, `cumcm_panel`,
`cumcm_convergence`, `cumcm_pareto`, `cumcm_feature_importance`, and
`cumcm_distribution`.

`CUMCM_Statistics.py` provides `r2_score`, `adjusted_r2`, `mse`, `rmse`,
`mae`, `mape`, `smape`, `rmsle`, `aic`, and `bic`. Paired NaN/inf values are
excluded and MAPE safely ignores zero denominators. `CUMCM_Export.save_cumcm`
is the standard PDF/SVG/PNG exporter; `CUMCM_Diagnostics.check_figure` and
`check_output` give PASS/WARN/ERROR reports without modifying a figure.

Use `CUMCM_Layout.cumcm_panel(nrows=2, ncols=2)` for panel canvases and draw
into returned axes. The labels `(a)`, `(b)` are panel identifiers, not figure
captions. Keep formal captions outside axes in Word or LaTeX.

Additional realistic demonstrations are `demo_statistics.py`,
`demo_model_compare.py`, `demo_paper_panel.py`, and `demo_quality_check.py`.
Run all with `python -m pytest tests -q`. See [CHANGELOG.md](CHANGELOG.md) for
the V1→V2 migration record.

`CUMCM_Style.py` is a compact, independent visual system for Chinese mathematical-modeling papers.  It is deliberately low-ink: white background, serif figures, light horizontal grids, muted colours, vector-first output, and no figure number inside an axes.  Keep figure numbers and captions in Word or LaTeX.

## Installation and quick start

```bash
cd CUMCM_Visualization
python -m pip install -r requirements.txt
python examples/demo_all.py
python -m unittest discover -s tests
```

```python
from CUMCM_Style import cumcm_bar

cumcm_bar(
    categories=["A", "B", "C"], values=[120, 150, 110],
    xlabel="赛题", ylabel="论文数量", highlight="max",
    filename="figure_01", output_dir="figures"
)
```

Each named figure saves `figure_01.pdf`, `figure_01.svg`, and `figure_01.png`; PNG is 600 dpi.  The first two are preferred for manuscript editing and typesetting. All functions return `(fig, ax)` for manuscript-specific additions.

## API

| Function | Use |
| --- | --- |
| `set_cumcm_style`, `create_figure`, `polish_axes`, `save_figure`, `check_fonts` | shared style, canvas, output and diagnostics |
| `cumcm_bar` | categorical comparison; `highlight="max"` / `"min"`, mean line and labels |
| `cumcm_line`, `cumcm_multi_line` | one or several trends/models |
| `cumcm_scatter` | points with linear fit, equation, $R^2$ and optional RMSE |
| `cumcm_box`, `cumcm_heatmap` | distributions and matrices/correlations |
| `cumcm_sensitivity` | signed, absolute-effect-sorted tornado plot |
| `cumcm_model_compare` | best-score highlight for higher/lower-is-better metrics |
| `cumcm_residual`, `cumcm_prediction`, `cumcm_prediction_band` | model diagnostics and forecast results |
| `cumcm_multi_panel` | 2×2, 2×3, etc.; adds `(a)`, `(b)` panel labels |

Use `ax=` on every core plot to place it into a `cumcm_multi_panel` canvas.  `filename=None` (the default) keeps a plot in memory only.  `save_figure(fig, name, folder)` writes the three standard formats.

## Visual specification

- **Colours:** primary blue `#5B7FA3`; emphasis red `#B6534B`; supporting green/orange/purple are reserved for genuinely distinct series. Baselines should be gray and the best result red.
- **Fonts:** English/numerals prefer Times New Roman; Chinese prefers SimSun then STSong/Microsoft YaHei. `check_fonts()` reports the active fallbacks. `axes.unicode_minus=False` handles negative signs; PDF Type 42 embedding and editable SVG text are configured by default.
- **Dimensions:** `FIGSIZE_SINGLE` is 89 mm wide; `FIGSIZE_DOUBLE` 178 mm; `FIGSIZE_WIDE` is 178 mm × 70 mm. Pass any `(width_inches, height_inches)` through `figsize=` when necessary.
- **Line hierarchy:** axes 0.8 pt, primary data 1.45 pt, fit/attention 1.7–1.8 pt, reference 0.75–0.85 pt, grid 0.45 pt. Avoid title-heavy charts, all-colour bars, heavy borders, gradients, and 3-D effects.

## Python → Origin → Word workflow

Use Python for reproducible calculations, statistical analysis, complex graphics and batch generation. Export PDF/SVG, then use Origin only for the rare final visual refinements (local annotations, exact font/line choices, composite layouts). Export PDF/SVG/EMF from Origin for Word or LaTeX. Do not recreate the analysis in Origin: retain Python as the single source of truth.

## Troubleshooting

- **Chinese glyphs missing:** run `from CUMCM_Style import check_fonts; print(check_fonts())`. Install SimSun/STSong or a CJK font, restart Python, then clear Matplotlib's font cache if required.
- **Times New Roman unavailable:** the code falls back to Times/DejaVu Serif rather than failing. Install Times New Roman and rerun `check_fonts()` for publication matching.
- **Crop or output issue:** output errors are reported per format so an SVG/PDF failure does not block the remaining files. Use `filename` without an extension.
- **Too many lines:** split into panels or retain only models needed to support the conclusion; the fixed palette is not an invitation to use every colour.
