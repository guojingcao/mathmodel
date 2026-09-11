"""A four-panel paper figure drawn into one consistent canvas."""
from pathlib import Path
import sys, numpy as np
ROOT=Path(__file__).resolve().parents[1]; sys.path.insert(0,str(ROOT))
from CUMCM_Layout import cumcm_panel
from CUMCM_Style import cumcm_distribution,cumcm_scatter,cumcm_residual,cumcm_prediction
from CUMCM_Export import save_cumcm
rng=np.random.default_rng(42); x=np.arange(1,13); actual=100+2*x+rng.normal(0,1,12); fitted=100+2*x
fig,axes=cumcm_panel(nrows=2,ncols=2)
cumcm_distribution(rng.normal(0,1,120),xlabel="Standardized residual",ax=axes[0,0])
cumcm_scatter(x,actual,xlabel="Month",ylabel="Demand",show_ci=True,ax=axes[0,1])
cumcm_residual(fitted,actual-fitted,ax=axes[1,0])
cumcm_prediction(x,actual,fitted,xlabel="Month",ylabel="Demand",ax=axes[1,1])
save_cumcm(fig,ROOT/"figures"/"v2_paper_panel"); print("Paper panel written.")
