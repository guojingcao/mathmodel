import os, time, traceback

out = r'D:/My_MathModeling_Project/corpus/03_analysis/topic_paper_count_journal.opju'
out_win = out.replace('/', '\\')

import originpro as op

print('graphs in this attach:', op.graph_list() if callable(op.graph_list) else op.graph_list)
try:
    print('find Book1:', op.find_sheet('w', 'Book1'))
except Exception as e:
    print('find err', e)

# Approach 1: LabTalk doc -s
for name, cmd in [
    ('doc -s (save as)', 'doc -s "%s";' % out_win),
    ('doc -sa', 'doc -sa "%s";' % out_win),
]:
    try:
        res = op.lt_exec(cmd)
        time.sleep(2)
        exists = os.path.exists(out)
        print(name, 'lt result:', res, 'exists=', exists,
              'size=', os.path.getsize(out) if exists else '-')
    except Exception:
        traceback.print_exc()

# Approach 2: originpro op.save again, wait longer
try:
    ok = op.save(out_win)
    time.sleep(4)
    print('op.save returned', ok, 'size now=', os.path.getsize(out) if os.path.exists(out) else '-')
except Exception:
    traceback.print_exc()

# Approach 3: raw COM LT on OriginExt app
try:
    app = op.OriginExt.ApplicationSI if hasattr(op.OriginExt, 'ApplicationSI') else None
    if app is None and hasattr(op, 'oext'):
        app = op.oext.ApplicationSI
    if app is not None:
        lt = app.LT_execute
        lt('doc -s "%s";' % out_win)
        time.sleep(2)
        print('COM doc -s size=', os.path.getsize(out) if os.path.exists(out) else '-')
except Exception:
    traceback.print_exc()
