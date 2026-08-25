"""Check the paper's numeric claims against the runs that produced them.

Each entry names a quantity, says how to compute it from the archived runs,
and gives the sentence pattern that must carry it. Both halves matter: the
computation catches a number that changed when the data was re-measured, and
the pattern catches an edit that was written but never saved, which a
set-membership check cannot see because a stale value often coincides with
some other quantity in the paper.

    python check_claims.py
"""
import io, os, re, json, sys, statistics as st

PAPER = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'main.tex')
TABLES = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                      '..', 'results', 'tables')
RUNS = r'G:\ccpe-flids-artifact\results\runs'


def ttt(m):
    """Interpolated wall-clock crossing of the target, as the paper defines it."""
    tgt = m['config']['target_macro_f1']
    prev = None
    for r in m['history']:
        if r['macro_f1'] >= tgt:
            if prev is None:
                return r['wall_s']
            sp = r['macro_f1'] - prev['macro_f1']
            fr = (tgt - prev['macro_f1']) / sp if sp else 0.0
            return prev['wall_s'] + fr * (r['wall_s'] - prev['wall_s'])
        prev = r
    return None


def runs(batch, pattern):
    out = []
    d0 = os.path.join(RUNS, batch)
    if not os.path.isdir(d0):
        return out
    rx = re.compile(pattern)
    for d in sorted(os.listdir(d0)):
        if not rx.fullmatch(d):
            continue
        p = os.path.join(d0, d, 'metrics.json')
        if os.path.exists(p):
            out.append(json.load(open(p)))
    return out


def T(batch, pattern):
    v = [ttt(m) for m in runs(batch, pattern)]
    v = [x for x in v if x is not None]
    return st.mean(v) if v else None


def F1(batch, pattern):
    v = [max(r['macro_f1'] for r in m['history']) for m in runs(batch, pattern)]
    return st.mean(v) if v else None


# label, computed value, regex the paper must contain (one capture = the number)
CLAIMS = [
    ('T1 CICIDS LAN FedAsync', T('fin_cross2', r'cicids2017__fedasync__lan__s\d'),
     r"FedAsync needs \$([\d.]+)\$\\,s"),
    ('T1 CICIDS LAN sync', T('fin_cross2', r'cicids2017__sync__lan__s\d'),
     r"against the barrier's \$([\d.]+)\$\\,s\."),
    ('T1 CICIDS sync degradation',
     T('fin_cross2', r'cicids2017__sync__mixed__s\d') /
     T('fin_cross2', r'cicids2017__sync__lan__s\d'),
     r"FedAvg \$([\d.]+)\\times\$ on CICIDS2017"),
    ('T1 N-BaIoT sync degradation',
     T('fin_cross2', r'nbaiot__sync__mixed__s\d') /
     T('fin_cross2', r'nbaiot__sync__lan__s\d'),
     r"\$([\d.]+)\\times\$ on N-BaIoT and"),
    ('T4 sync 10 steps', T('fin_steps_f', r'cicids2017__sync__n10__s\d'),
     r"FedAvg falls from \$([\d.]+)\{\\pm\}"),
    ('T4 sync 120 steps', T('fin_steps_f', r'cicids2017__sync__n120__s\d'),
     r"\n\$([\d.]+)\{\\pm\}[\d.]+\$\\,s, a \$[\d.]+\\times\$ improvement"),
    ('T4 sync speedup', T('fin_steps_f', r'cicids2017__sync__n10__s\d') /
     T('fin_steps_f', r'cicids2017__sync__n120__s\d'),
     r"s, a \$([\d.]+)\\times\$ improvement"),
    ('T5 N-BaIoT inflation', T('fin_depth_f', r'nbaiot__sync__large__s\d') /
     T('fin_depth_f', r'nbaiot__sync__small__s\d'),
     r"FedAvg from \$[\d.]+\$ to \$[\d.]+\$\\,s on N-BaIoT and from \$[\d.]+\$ to \$[\d.]+\$\\,s on\nBot-IoT, factors of \$([\d.]+)\$"),
    ('T6 sync 12->96', T('finscale_f', r'n96__sync__s\d') /
     T('finscale_f', r'n12__sync__s\d'),
     r"a factor of\n\$([\d.]+)\$, while FedAsync"),
    ('T8 r100 K16 beta0', T('fin_beta', r'r100__b0__K16__s\d'),
     r"the sweep runs \$([\d.]+)\$"),
    ('T8 r100 K16 beta1', T('fin_beta', r'r100__b1__K16__s\d'),
     r"and\n\$([\d.]+)\$\\,s\."),
    ('Fig2 sync 1 ms', T('fin_delay_f', r'cicids2017__sync__d1__s\d'),
     r"steeply, \$([\d.]+)\$\\,s at \$1\$\\,ms"),
    ('Fig2 sync 250 ms', T('fin_delay_f', r'cicids2017__sync__d250__s\d'),
     r"\$([\d.]+)\$ at \$250\$, a factor"),
    ('Fig3 sync IID', T('fin_noniid_f', r'cicids2017__sync__iid__s\d'),
     r"FedAvg\n\$([\d.]+)\\rightarrow"),
    ('Fig3 sync alpha 0.1', T('fin_noniid_f', r'cicids2017__sync__a0\.1__s\d'),
     r"\\rightarrow([\d.]+)\$\\,s, and every asynchronous"),
    ('T2 sync ceiling a0.5', F1('finsweep_f', r'cicids2017__sync__a0\.5__s\d'),
     r"against the barrier's\n\$([\d.]+)\$;"),
    ('clip r25 K16', T('fin_clip', r'r25__K16__s\d'),
     r"better than the raw\nterm, \$([\d.]+)\$ against"),
]

tex = io.open(PAPER, encoding='utf-8').read()
for f in sorted(os.listdir(TABLES)):
    tex += io.open(os.path.join(TABLES, f), encoding='utf-8').read()

print('%-28s %10s %10s  %s' % ('claim', 'in data', 'in paper', ''))
bad = 0
for label, value, pattern in CLAIMS:
    if value is None:
        print('%-28s %10s %10s  NO DATA' % (label, '-', '-'))
        bad += 1
        continue
    m = re.search(pattern, tex)
    if not m:
        print('%-28s %10.1f %10s  PATTERN NOT FOUND' % (label, value, '-'))
        bad += 1
        continue
    stated = float(m.group(1))
    ok = abs(stated - value) <= 0.05 * max(abs(value), 1e-9)
    if not ok:
        bad += 1
    print('%-28s %10.3f %10s  %s'
          % (label, value, m.group(1), 'ok' if ok else 'MISMATCH'))

print('\n%d claims, %d problems' % (len(CLAIMS), bad))
sys.exit(1 if bad else 0)
