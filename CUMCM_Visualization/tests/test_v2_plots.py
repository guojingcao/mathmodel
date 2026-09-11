import matplotlib.pyplot as plt
import numpy as np
import pytest
from CUMCM_Style import *
def close(call):
    fig,_=call(); assert fig is not None; plt.close(fig)
def test_direction_min_and_max():
    close(lambda:cumcm_model_compare(["A","B"],[2,1],direction="min")); close(lambda:cumcm_model_compare(["A","B"],[2,1],direction="max"))
def test_advanced_plots():
    close(lambda:cumcm_metric_compare(["A","B"],{"R2":[.8,.9],"RMSE":[2,1]}))
    close(lambda:cumcm_convergence(range(5),[5,4,3,3,2]))
    close(lambda:cumcm_pareto([1,2,3],[3,2,1]))
    close(lambda:cumcm_feature_importance(["a","b"],[.3,-.2]))
    close(lambda:cumcm_distribution([1,2,3]))
    close(lambda:cumcm_residual_diagnostics([1,2,3,4],[.1,-.1,.2,0]))
def test_scatter_ci_and_polynomial(): close(lambda:cumcm_scatter([1,2,3,4],[2,5,10,17],fit_type="polynomial"))
def test_pareto_invalid_direction():
    with pytest.raises(ValueError): cumcm_pareto([1],[1],x_direction="up")
