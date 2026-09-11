"""Comparison in both maximisation and minimisation metric directions."""
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1]; sys.path.insert(0,str(ROOT))
from CUMCM_Style import cumcm_model_compare, cumcm_metric_compare
out=ROOT/"figures"
models=["Linear","Random Forest","XGBoost","SVR"]
cumcm_model_compare(models,[.79,.86,.90,.84],metric="$R^2$",direction="max",filename="v2_model_r2",output_dir=out)
cumcm_model_compare(models,[14.1,10.3,8.8,11.2],metric="RMSE",direction="min",filename="v2_model_rmse",output_dir=out)
cumcm_metric_compare(models,{"R²":[.79,.86,.90,.84],"RMSE":[14.1,10.3,8.8,11.2],"MAE":[10.1,7.3,6.8,8.2]},directions={"R²":"max","RMSE":"min","MAE":"min"},filename="v2_metric_matrix",output_dir=out)
print("Model comparison figures written.")
