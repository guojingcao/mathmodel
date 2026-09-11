import matplotlib.pyplot as plt
from CUMCM_Diagnostics import check_figure,check_output
from CUMCM_Export import save_cumcm
def test_figure_report():
    fig,_=plt.subplots(); assert any("Font" in x for x in check_figure(fig)); plt.close(fig)
def test_output_report(tmp_path):
    fig,_=plt.subplots(); stem=tmp_path/"f"; save_cumcm(fig,stem); assert all(x.startswith("[PASS]") for x in check_output(stem)); plt.close(fig)
