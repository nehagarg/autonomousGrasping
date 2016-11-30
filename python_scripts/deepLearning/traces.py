import os
from os import path
act_dict = {
        'amn': 0,
        'ame': 1,
        'ams': 2,
        'amw': 3,
        'as': 4
}
obs_dict = {
        'ogood': 1,
        'obad': 2
}

for i in xrange(20):
    act_dict['ac{}'.format(i)] = i+5

def _parse_appl(f):
    line = next(f)
    assert 'begin' in line
    act = None
    for line in f:
        if 'terminated' in line:
            continue
        if '----- time' in line or 'Reached terminal' in line:
            return
        k, v = line.split(':')
        k = k.strip()
        v = v.strip()
        if k=='A':
            act = act_dict[v[1:-1]] # remove bracket
        elif k=='O':
            obs = obs_dict[v[1:-1]] # remove bracket
            if act <= act_dict['as']:
                obs = 0
            yield (act, obs)

def _parse_despot(f):
    line = next(f)
    while line and not line.startswith('###'):
        line = next(f)
    act = None
    for line in f:
        if line.startswith('Simulation terminated'):
            return
        if line.startswith('- Action ='):
            act = int(line.split('=')[1].strip())
        elif line.startswith('- Observation ='):
            obs = int(line.split('=')[1].strip())
            yield (act, obs)

def _parse_canadian_bridge(f):
    line = next(f)
    while line and not line.startswith('scenario'):
        line = next(f)
    act = None
    for line in f:
        if line.startswith('End'):
            return
        if line.startswith('****Action:'):
            act = int(line.split(":")[1].strip())
        elif line.startswith('****Observation:'):
            obs = int(line.split(':')[1].strip())
            yield (act, obs)


def _parse_file(fn, parse):
    f = open(fn)
    seqs = []
    while True:
        s = list(parse(f))
        if not s:
            break
        seqs.append(s)
    return seqs

def parse(fn='trace.txt', type='appl'):
    _parse = {'appl': _parse_appl, 'despot': _parse_despot, 'canadian_bridge': _parse_canadian_bridge}[type]
    if path.isdir(fn):
        fns = [path.join(fn, n) for n in os.listdir(fn)]
    else:
        fns = [fn]
    seqs = sum((_parse_file(f, _parse) for f in fns), [])
    return seqs

#test
if __name__ == '__main__':
    #fn = 'workspace/canadian_bridge/hard/canadian_bridge_trace/trace'
    fn = 'trace/rocksample_11_11'
    trace = parse(fn, 'appl')
    print (len(trace))
    print (len(trace[1]))
    seqs = trace
    st = ['$']
    xseqs = seqs
    yseqs = [[t[0] for t in seq][1:]+st for seq in seqs]
    print (xseqs[:10])
    print (yseqs[:10])
