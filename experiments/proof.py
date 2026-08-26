"""Copy-editing checks over the manuscript source.

Reports doubled words, spacing slips, inconsistent hyphenation, mixed
British/American spelling, unit formatting and term capitalisation. Prints a
line number and the offending fragment so each hit can be judged by eye.
"""
import io, os, re, sys

HERE = os.path.dirname(os.path.abspath(__file__))
PAPER = os.path.join(HERE, 'main.tex')
TABLES = os.path.join(HERE, '..', 'results', 'tables')

src = io.open(PAPER, encoding='utf-8').read()
lines = src.split('\n')
start = next(i for i, l in enumerate(lines) if l.startswith(r'\begin{document}'))
body = lines[start:]


def hits(pattern, label, flags=0, skip_math=True):
    out = []
    rx = re.compile(pattern, flags)
    for n, l in enumerate(body, start + 1):
        if l.lstrip().startswith('%'):
            continue
        probe = re.sub(r'\$[^$]*\$', ' ', l) if skip_math else l
        for m in rx.finditer(probe):
            out.append((n, ' '.join(probe[max(0, m.start() - 34):m.end() + 34].split())))
    if out:
        print('\n%s  (%d)' % (label, len(out)))
        for n, ctx in out[:12]:
            print('   %5d  %s' % (n, ctx))
        if len(out) > 12:
            print('   ... %d more' % (len(out) - 12))
    return out


total = 0
total += len(hits(r'\b(\w+)\s+\1\b', 'doubled words', re.I))
total += len(hits(r'\s+[,.;:]', 'space before punctuation'))
total += len(hits(r'[,.;:][A-Za-z]', 'missing space after punctuation'))
total += len(hits(r'\bwall clock\b|\bwallclock\b', 'wall-clock spelled inconsistently', re.I))
total += len(hits(r'\bnon IID\b|\bnoniid\b|\bnon-iid\b(?!\W)', 'non-IID spelled inconsistently'))
total += len(hits(r'\btest-set\b|\btestset\b', 'test set spelled inconsistently'))
total += len(hits(r'\bdata-set\b|\bdataset s\b', 'dataset spelled inconsistently'))
total += len(hits(r'"', 'straight quotes (use `` and \'\')'))
total += len(hits(r'\d\s*(ms|MB|GB|s)\b(?!\w)', 'unit not bound with \\, to its number'))
total += len(hits(r'\bFedbuff\b|\bFedbuf\b|\bFedAVG\b|\bFedavg\b|\bfedasync\b(?![_}])',
                  'strategy name miscapitalised'))
total += len(hits(r'\bmacro F1\b|\bMacro-F1\b(?!:)|\bmacro-f1\b', 'macro-F1 spelled inconsistently'))
total += len(hits(r'\bi\.e\.[^,]|\be\.g\.[^,]', 'i.e./e.g. without following comma'))
total += len(hits(r'\betc\b(?!\.)', 'etc without full stop'))

print('\n--- British / American spelling ---')
PAIRS = [('ise', 'ize'), ('isation', 'ization'), ('our', 'or')]
STEMS = [('serialis', 'serializ'), ('normalis', 'normaliz'), ('standardis', 'standardiz'),
         ('characteris', 'characteriz'), ('regularis', 'regulariz'), ('emphasis', 'emphasiz'),
         ('behaviour', 'behavior'), ('favour', 'favor'), ('neighbour', 'neighbor'),
         ('modelling', 'modeling'), ('labelled', 'labeled'), ('equalis', 'equaliz')]
mixed = []
txt = '\n'.join(body)
for br, am in STEMS:
    nb = len(re.findall(br, txt, re.I))
    na = len(re.findall(am, txt, re.I))
    if nb and na:
        mixed.append((br, nb, am, na))
    elif nb or na:
        print('   %-14s British %d / American %d' % (br + '/' + am, nb, na))
if mixed:
    print('\n   MIXED within one stem:')
    for br, nb, am, na in mixed:
        print('      %s %d  vs  %s %d' % (br, nb, am, na))
    total += len(mixed)

print('\n%d flagged items' % total)
sys.exit(0)
