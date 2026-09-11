import os
out = r'D:\My_MathModeling_Project\corpus\03_analysis\topic_paper_count_journal.opju'
if os.path.exists(out):
    with open(out, 'rb') as f:
        head = f.read(16)
    print('opju head bytes:', head)
print('---')
import originpro as op
public = [x for x in dir(op) if not x.startswith('_')]
print('originpro public:', public)
# try to enumerate what this attachment sees
for probe in ('pages', 'worksheets', 'graphs', 'project'):
    if hasattr(op, probe):
        try:
            v = getattr(op, probe)
            print(probe, '->', v if not callable(v) else '<callable>')
        except Exception as e:
            print(probe, 'err', e)
# attempt: find our sheet
try:
    sh = op.find_sheet('w', 'Book3')
    print('found Book3 from script attach:', sh)
except Exception as e:
    print('find Book3 err:', e)
