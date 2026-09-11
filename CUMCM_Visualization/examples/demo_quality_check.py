"""Run non-destructive quality checks on a representative generated figure."""
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1]; sys.path.insert(0,str(ROOT))
from CUMCM_Style import cumcm_line
from CUMCM_Export import save_cumcm
from CUMCM_Diagnostics import check_figure,check_output
fig,_=cumcm_line([1,2,3],[2,3,5],xlabel="Period",ylabel="Indicator")
stem=ROOT/"figures"/"v2_quality_check"; save_cumcm(fig,stem)
print("\n".join(check_figure(fig)+check_output(stem)))
