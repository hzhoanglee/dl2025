from layers import *
from metrics import Metric


class FashionNet:
    def __init__(self, cfg):
        self.cfg = cfg
        self.procs = []
        self.met = Metric()
        self._build()
    
    def _build(self):
        conv_p = ConvProc(
            self.cfg['depth'], 
            self.cfg['maps'], 
            self.cfg['k_size']
        )
        self.procs.append(conv_p)
        
        relu_p = ActProc('relu')
        self.procs.append(relu_p)
        
        pool_p = PoolProc(3)
        self.procs.append(pool_p)
        
        flat_p = FlatProc()
        self.procs.append(flat_p)
        
        conv_out_sz = (self.cfg['size'] - self.cfg['k_size'] + 1) // 3
        flat_sz = self.cfg['maps'] * conv_out_sz * conv_out_sz
        
        dense_p = DenseProc(flat_sz, self.cfg['neurons'])
        self.procs.append(dense_p)
        
        relu2_p = ActProc('relu')
        self.procs.append(relu2_p)
        
        out_p = DenseProc(self.cfg['neurons'], self.cfg['classes'])
        self.procs.append(out_p)
        
        soft_p = ActProc('softmax')
        self.procs.append(soft_p)
    
    def fwd_pass(self, inp):
        flow = inp
        for p in self.procs:
            flow = p.fwd(flow)
        return flow
    
    def bwd_pass(self, g_flow, lr):
        for p in reversed(self.procs):
            g_flow = p.bwd(g_flow, lr)
        return g_flow
    
    def train_sess(self, tr_data):
        loss_hist = []
        lr = self.cfg['lr']
        
        for ep in range(self.cfg['epochs']):
            tot_loss = 0
            batch_cnt = 0
            
            for bx, by in tr_data:
                pred = self.fwd_pass(bx)
                
                l_val, l_grad = self.met.cross_entropy(pred, by)
                
                tot_loss += l_val
                batch_cnt += 1
                
                self.bwd_pass(l_grad, lr)
            
            ep_avg = tot_loss / batch_cnt
            loss_hist.append(ep_avg)
            
            print(f"Training iteration {ep + 1}/{self.cfg['epochs']}: Loss = {ep_avg:.4f}")
            
            if ep_avg < 0.5:
                print("Training convergence detected - stopping early")
                break
        
        return loss_hist
    
    def eval_sess(self, te_data):
        all_pred = []
        all_true = []
        
        for bx, by in te_data:
            batch_pred = self.fwd_pass(bx)
            
            for i in range(len(batch_pred)):
                pred_prob = batch_pred[i]
                true_oh = by[i]
                
                p_cls = pred_prob.index(max(pred_prob))
                t_cls = true_oh.index(max(true_oh))
                
                all_pred.append(p_cls)
                all_true.append(t_cls)
        
        prec, rec, f1, acc = self.met.calc_scores(all_true, all_pred)
        
        self.met.gen_conf_viz(all_true, all_pred)
        
        return acc * 100, prec * 100, rec * 100, f1 * 100 