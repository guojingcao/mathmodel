import matplotlib.pyplot as plt
from CUMCM_Export import save_cumcm
def test_export_all(tmp_path):
    fig,_=plt.subplots(); paths=save_cumcm(fig,tmp_path/"chart"); assert len(paths)==3 and all(p.exists() for p in paths); plt.close(fig)
def test_export_bad_format(tmp_path):
    fig,_=plt.subplots()
    try:
        import pytest
        with pytest.raises(ValueError): save_cumcm(fig,tmp_path/"chart",["jpg"])
    finally: plt.close(fig)
