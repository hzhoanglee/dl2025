import math
from utils import rng


class ConvProc:
    def __init__(self, in_ch, out_ch, k):
        self.in_ch = in_ch
        self.out_ch = out_ch
        self.k = k
        self.w = []
        self.b = []
        
        for _ in range(out_ch):
            filt = []
            for _ in range(in_ch):
                ch = []
                for _ in range(k):
                    row = []
                    for _ in range(k):
                        row.append(rng.uniform(-0.5, 0.5))
                    ch.append(row)
                filt.append(ch)
            self.w.append(filt)
            self.b.append(rng.uniform(-0.1, 0.1))
        
        self.last_in = None
    
    def fwd(self, x):
        self.last_in = x
        bs = len(x)
        h, w = len(x[0]), len(x[0][0])
        oh, ow = h - self.k + 1, w - self.k + 1
        
        out = []
        for b in range(bs):
            bout = []
            for f in range(self.out_ch):
                fmap = []
                for i in range(oh):
                    row = []
                    for j in range(ow):
                        s = 0
                        for ci in range(self.in_ch):
                            for ki in range(self.k):
                                for kj in range(self.k):
                                    if ci == 0:
                                        s += x[b][i + ki][j + kj] * self.w[f][ci][ki][kj]
                        s += self.b[f]
                        row.append(s)
                    fmap.append(row)
                bout.append(fmap)
            out.append(bout)
        return out
    
    def bwd(self, g, lr):
        return g


class PoolProc:
    def __init__(self, p):
        self.p = p
        self.pos = None
    
    def fwd(self, x):
        bs = len(x)
        ch = len(x[0])
        h, w = len(x[0][0]), len(x[0][0][0])

        out = []
        self.pos = []
        
        for b in range(bs):
            bout = []
            bpos = []
            for c in range(ch):
                cout = []
                cpos = []
                for i in range(0, h, self.p):
                    row = []
                    rpos = []
                    for j in range(0, w, self.p):
                        mv = float('-inf')
                        mi, mj = i, j
                        for pi in range(i, min(i + self.p, h)):
                            for pj in range(j, min(j + self.p, w)):
                                if x[b][c][pi][pj] > mv:
                                    mv = x[b][c][pi][pj]
                                    mi, mj = pi, pj
                        row.append(mv)
                        rpos.append((mi, mj))
                    cout.append(row)
                    cpos.append(rpos)
                bout.append(cout)
                bpos.append(cpos)
            out.append(bout)
            self.pos.append(bpos)
        
        return out
    
    def bwd(self, g, lr):
        return g


class FlatProc:
    def __init__(self):
        self.shape = None
    
    def fwd(self, x):
        bs = len(x)
        if len(x[0]) > 0 and isinstance(x[0][0], list) and isinstance(x[0][0][0], list):
            self.shape = (len(x[0]), len(x[0][0]), len(x[0][0][0]))
            out = []
            for b in range(bs):
                flat = []
                for c in range(len(x[b])):
                    for i in range(len(x[b][c])):
                        for j in range(len(x[b][c][i])):
                            flat.append(x[b][c][i][j])
                out.append(flat)
            return out
        else:
            return x
    
    def bwd(self, g, lr):
        return g


class DenseProc:
    def __init__(self, in_sz, out_sz):
        self.in_sz = in_sz
        self.out_sz = out_sz
        self.w = []
        self.b = []
        
        for i in range(out_sz):
            row = []
            for j in range(in_sz):
                row.append(rng.uniform(-0.5, 0.5))
            self.w.append(row)
            self.b.append(rng.uniform(-0.1, 0.1))
        
        self.last_in = None
    
    def fwd(self, x):
        self.last_in = x
        bs = len(x)
        out = []
        
        for b in range(bs):
            bout = []
            for i in range(self.out_sz):
                s = self.b[i]
                for j in range(self.in_sz):
                    s += x[b][j] * self.w[i][j]
                bout.append(s)
            out.append(bout)
        
        return out
    
    def bwd(self, g, lr):
        if self.last_in is None or len(g) == 0:
            return g
            
        bs = len(g)
        
        for b in range(min(bs, len(self.last_in))):
            gsz = len(g[b]) if b < len(g) else 0
            for i in range(min(self.out_sz, gsz)):
                if i < len(self.b):
                    self.b[i] -= lr * g[b][i] / bs
                    for j in range(min(self.in_sz, len(self.last_in[b]))):
                        if i < len(self.w) and j < len(self.w[i]):
                            self.w[i][j] -= lr * g[b][i] * self.last_in[b][j] / bs
        
        return g


class ActProc:
    def __init__(self, act):
        self.act = act
        self.last_in = None
    
    def fwd(self, x):
        self.last_in = x
        out = []
        
        if self.act == 'relu':
            for batch in x:
                if isinstance(batch[0], list):
                    bout = []
                    for ch in batch:
                        cout = []
                        for row in ch:
                            rout = []
                            for val in row:
                                rout.append(max(0, val))
                            cout.append(rout)
                        bout.append(cout)
                    out.append(bout)
                else:
                    bout = []
                    for val in batch:
                        bout.append(max(0, val))
                    out.append(bout)
        
        elif self.act == 'softmax':
            for batch in x:
                exp_vals = []
                mv = max(batch)
                for val in batch:
                    exp_vals.append(math.exp(val - mv))
                se = sum(exp_vals)
                sout = []
                for ev in exp_vals:
                    sout.append(ev / se)
                out.append(sout)
        
        return out
    
    def bwd(self, g, lr):
        return g