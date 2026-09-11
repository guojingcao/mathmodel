import matplotlib.pyplot as plt
import pytest
from CUMCM_Layout import cumcm_panel
def test_panel_shape_and_labels():
    fig,axes=cumcm_panel(nrows=2,ncols=2); assert axes.shape==(2,2) and axes[0,0].texts[0].get_text()=="(a)"; plt.close(fig)
def test_panel_invalid():
    with pytest.raises(ValueError): cumcm_panel(nrows=0)
