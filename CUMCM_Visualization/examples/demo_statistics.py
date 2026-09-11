"""Model-evaluation metrics for a realistic demand forecasting result."""
from pathlib import Path
import sys, numpy as np
ROOT=Path(__file__).resolve().parents[1]; sys.path.insert(0,str(ROOT))
from CUMCM_Statistics import r2_score, rmse, mae, mape, smape
actual=np.array([118,121,126,130,134,139.]); predicted=np.array([117,123,125,131,133,141.])
print({"R2":round(r2_score(actual,predicted),3),"RMSE":round(rmse(actual,predicted),3),"MAE":round(mae(actual,predicted),3),"MAPE%":round(mape(actual,predicted),2),"sMAPE%":round(smape(actual,predicted),2)})
