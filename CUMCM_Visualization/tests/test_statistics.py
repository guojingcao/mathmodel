import math
import numpy as np
import pytest
from CUMCM_Statistics import *

def test_perfect_metrics():
    assert r2_score([1,2,3],[1,2,3]) == 1
    assert rmse([1,2],[1,2]) == 0 and mae([1,2],[1,2]) == 0 and mse([1,2],[1,2]) == 0
def test_nan_is_ignored(): assert rmse([1,np.nan,3],[1,99,5]) == pytest.approx(math.sqrt(2))
def test_mape_zero_safe(): assert mape([0,10],[2,11]) == pytest.approx(10)
def test_mape_all_zero_nan(): assert math.isnan(mape([0],[1]))
def test_smape(): assert smape([10],[12]) == pytest.approx(18.1818,rel=1e-4)
def test_adjusted_r2(): assert adjusted_r2([1,2,3,4],[1,2,3,4],1) == 1
def test_rmsle_negative_raises():
    with pytest.raises(ValueError): rmsle([-1],[1])
@pytest.mark.parametrize("fn",[mse,rmse,mae,r2_score,mape,smape])
def test_lengths_raise(fn):
    with pytest.raises(ValueError): fn([1],[1,2])
def test_information_criteria(): assert np.isfinite(aic([1,2,3],2)) and np.isfinite(bic([1,2,3],2))
