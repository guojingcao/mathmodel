"""Generate the complete CUMCM_Style gallery without external data."""
from pathlib import Path
import sys
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from CUMCM_Style import (cumcm_bar, cumcm_box, cumcm_heatmap, cumcm_line,
    cumcm_model_compare, cumcm_multi_line, cumcm_multi_panel, cumcm_prediction,
    cumcm_prediction_band, cumcm_residual, cumcm_scatter, cumcm_sensitivity, save_figure)
from CUMCM_Style import (cumcm_convergence, cumcm_distribution, cumcm_feature_importance,
    cumcm_metric_compare, cumcm_pareto, cumcm_residual_diagnostics)

OUT = ROOT / "figures"
rng = np.random.default_rng(2026)
x = np.arange(1, 11)
actual = 26 + 1.7 * x + np.sin(x) * 2
predicted = actual + rng.normal(0, 1.0, len(x))

cumcm_bar(["Scheme A", "Scheme B", "Scheme C", "Scheme D"], [82.4, 91.6, 87.2, 85.9],
          ylabel="Composite score", highlight="max", show_values=True, mean_line=True,
          filename="01_bar", output_dir=OUT)
cumcm_line(x, actual, xlabel="Period", ylabel="Demand (units)", label="Observed",
           filename="02_line", output_dir=OUT)
cumcm_multi_line(x, {"Baseline": actual - 2, "Proposed": actual, "Scenario": actual + 2.5},
                 xlabel="Period", ylabel="Response", ncol=3, filename="03_multi_line", output_dir=OUT)
sx = np.linspace(3, 18, 32); sy = 1.8 * sx + 5 + rng.normal(0, 3, len(sx))
cumcm_scatter(sx, sy, xlabel="Input variable", ylabel="Output variable", show_rmse=True,
              filename="04_scatter_fit", output_dir=OUT)
cumcm_box([rng.normal(3.2, .45, 55), rng.normal(2.7, .4, 55), rng.normal(2.4, .38, 55)],
          ["Model A", "Model B", "Model C"], ylabel="Absolute error", filename="05_box", output_dir=OUT)
matrix = np.array([[1, .76, .32, .18], [.76, 1, .41, .27], [.32, .41, 1, .63], [.18, .27, .63, 1]])
cumcm_heatmap(matrix, xlabels=["X1", "X2", "X3", "X4"], ylabels=["X1", "X2", "X3", "X4"],
              filename="06_heatmap", output_dir=OUT)
cumcm_sensitivity(["Capacity", "Demand", "Cost", "Penalty", "Efficiency"], [-.23, .41, -.16, .29, .10],
                  xlabel="Standardized effect", filename="07_sensitivity", output_dir=OUT)
cumcm_model_compare(["ARIMA", "RF", "XGBoost", "Proposed"], [.812, .846, .862, .901], metric="$R^2$",
                    filename="08_model_compare", output_dir=OUT)
cumcm_residual(predicted, actual-predicted, filename="09_residual", output_dir=OUT)
cumcm_prediction(x, actual, predicted, xlabel="Period", ylabel="Demand (units)",
                 filename="10_prediction", output_dir=OUT)
cumcm_prediction_band(x, actual, predicted, predicted-2.4, predicted+2.4, xlabel="Period", ylabel="Demand (units)",
                      filename="11_prediction_band", output_dir=OUT)
fig, axes = cumcm_multi_panel(2, 2)
for i, ax in enumerate(axes.flat):
    cumcm_line(x, actual + rng.normal(0, 1, len(x)) + i * 3, xlabel="Period", ylabel="Response", ax=ax)
save_figure(fig, "12_multi_panel", OUT)
cumcm_distribution(rng.normal(100, 12, 150), xlabel="Annual indicator", filename="13_distribution", output_dir=OUT)
cumcm_metric_compare(["Linear", "RF", "XGBoost", "SVR"], {"R²": [.79,.86,.90,.84], "RMSE": [14.1,10.3,8.8,11.2], "MAE": [10.1,7.3,6.8,8.2]}, directions={"R²":"max","RMSE":"min","MAE":"min"}, filename="14_metric_compare", output_dir=OUT)
cumcm_residual_diagnostics(predicted, actual-predicted, filename="15_residual_diagnostics", output_dir=OUT)
cumcm_convergence(np.arange(1,81), 100 + 45*np.exp(-np.arange(80)/16) + rng.normal(0,.6,80), filename="16_convergence", output_dir=OUT)
px=rng.uniform(20,100,45); py=180-px+rng.normal(0,15,45)
cumcm_pareto(px,py,xlabel="Operating cost",ylabel="Delivery time",filename="17_pareto",output_dir=OUT)
cumcm_feature_importance(["Demand","Capacity","Cost","Distance","Penalty"],[.42,.31,-.22,.14,.08],filename="18_feature_importance",output_dir=OUT)
print(f"Gallery written to: {OUT}")
