import os, traceback

out = r'D:\My_MathModeling_Project\corpus\03_analysis\topic_paper_count_journal.opju'

def list_pages(app):
    names = []
    try:
        pages = app.Pages
        for i in range(pages.Count):
            try:
                p = pages.Item(i + 1)
                names.append(str(p.Name))
            except Exception:
                names.append('<item-%d-error>' % (i + 1))
    except Exception:
        pass
    return names

import win32com.client as win32

for progid in ('Origin.ApplicationSI', 'Origin.Application'):
    try:
        app = win32.GetActiveObject(progid)
    except Exception:
        print(progid, '-> not in ROT')
        continue
    try:
        names = list_pages(app)
        print(progid, '-> pages:', names)
        lt = app.LT_execute
        has_graph = any(str(n).lower().startswith('graph') for n in names)
        lt('doc -s "' + out + '";')
        ok = os.path.exists(out)
        print('  saved via', progid, 'exists=', ok,
              'size=', os.path.getsize(out) if ok else '-', 'hadGraph=', has_graph)
        if ok and os.path.getsize(out) > 20000:
            print('GOOD-SAVE')
            break
    except Exception:
        print(progid, 'error:')
        traceback.print_exc()
