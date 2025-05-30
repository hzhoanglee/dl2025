import math
import matplotlib.pyplot as plt


class Metric:
    def __init__(self):
        self.cls_lbl = ['T-shirt', 'Trouser', 'Pullover', 'Dress', 'Coat', 
                       'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Boot']
    
    def cross_entropy(self, pred, targ):
        bs = len(pred)
        tot_loss = 0.0
        grads = []
        
        for b in range(bs):
            pb = pred[b]
            tb = targ[b]
            
            loss = 0.0
            gb = []
            
            for i in range(len(pb)):
                if pb[i] <= 0:
                    pb[i] = 1e-7
                if pb[i] >= 1:
                    pb[i] = 1 - 1e-7
                
                loss -= tb[i] * math.log(pb[i])
                gb.append(pb[i] - tb[i])
            
            tot_loss += loss
            grads.append(gb)
        
        avg_loss = tot_loss / bs
        return avg_loss, grads
    
    def calc_scores(self, true_lbl, pred_lbl):
        n_cls = len(self.cls_lbl)
        conf = [[0 for _ in range(n_cls)] for _ in range(n_cls)]
        
        tot_samp = len(true_lbl)
        correct = 0
        
        for i in range(tot_samp):
            t_cls = true_lbl[i]
            p_cls = pred_lbl[i]
            conf[t_cls][p_cls] += 1
            if t_cls == p_cls:
                correct += 1
        
        acc = correct / tot_samp
        
        cls_prec = []
        cls_rec = []
        
        for cls in range(n_cls):
            tp = conf[cls][cls]
            fp = sum(conf[i][cls] for i in range(n_cls)) - tp
            fn = sum(conf[cls][i] for i in range(n_cls)) - tp
            
            prec = tp / (tp + fp) if (tp + fp) > 0 else 0
            rec = tp / (tp + fn) if (tp + fn) > 0 else 0
            
            cls_prec.append(prec)
            cls_rec.append(rec)
        
        avg_prec = sum(cls_prec) / n_cls
        avg_rec = sum(cls_rec) / n_cls
        
        f1_scores = []
        for i in range(n_cls):
            if cls_prec[i] + cls_rec[i] > 0:
                f1 = 2 * (cls_prec[i] * cls_rec[i]) / (cls_prec[i] + cls_rec[i])
            else:
                f1 = 0
            f1_scores.append(f1)
        
        avg_f1 = sum(f1_scores) / n_cls
        
        return avg_prec, avg_rec, avg_f1, acc
    
    def gen_conf_viz(self, true_lbl, pred_lbl):
        n_cls = len(self.cls_lbl)
        conf_mat = [[0 for _ in range(n_cls)] for _ in range(n_cls)]
        
        for i in range(len(true_lbl)):
            t_cls = true_lbl[i]
            p_cls = pred_lbl[i]
            conf_mat[t_cls][p_cls] += 1
        
        fig, ax = plt.subplots(figsize=(10, 8))
        
        im = ax.imshow(conf_mat, interpolation='nearest', cmap='Blues')
        
        ax.set_xticks(range(n_cls))
        ax.set_yticks(range(n_cls))
        ax.set_xticklabels(self.cls_lbl, rotation=45, ha='right')
        ax.set_yticklabels(self.cls_lbl)
        
        ax.set_xlabel('Predicted Label')
        ax.set_ylabel('True Label')
        ax.set_title('Fashion-MNIST Confusion Matrix')
        
        for i in range(n_cls):
            for j in range(n_cls):
                text = ax.text(j, i, conf_mat[i][j],
                             ha="center", va="center", color="black")
        
        plt.colorbar(im)
        plt.tight_layout()
        plt.savefig('confusion_matrix.png', dpi=150, bbox_inches='tight')
        plt.close()
        
        return conf_mat 