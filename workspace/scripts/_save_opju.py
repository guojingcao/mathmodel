import sys, os, traceback
import originpro as op

out = r'D:\My_MathModeling_Project\corpus\03_analysis\topic_paper_count_journal.opju'
print('originpro version:', getattr(op, '__version__', 'n/a'))
save_fns = [x for x in dir(op) if 'save' in x.lower()]
print('save-like attrs:', save_fns)
try:
    if hasattr(op, 'save'):
        try:
            op.save(out)
            print('SAVE-OK via op.save')
        except TypeError as te:
            print('op.save TypeError:', te)
            # fallback: try with no args then LabTalk save as
            import win32com.client as win32
            app = op._orig_application if hasattr(op, '_orig_application') else None
            if app is not None:
                lt = app.LT_execute
                print('LT fallback attempt...')
    else:
        print('no op.save available')
except Exception:
    traceback.print_exc()

if os.path.exists(out):
    print('OPJU-EXISTS', os.path.getsize(out))
else:
    print('OPJU-MISSING')
    # Last-resort LabTalk through OriginExt COM directly
    try:
        import win32com.client as win32
        app = win32.GetActiveObject('Origin.ApplicationSI')
        lt = app.LT_execute
        lt('doc -s "' + out + '";')
        print('LT doc -s executed')
    except Exception:
        traceback.print_exc()
    print('OPJU-EXISTS-AFTER-LT', os.path.exists(out), os.path.getsize(out) if os.path.exists(out) else '')
