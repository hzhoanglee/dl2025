from utils import rng


def load_config_to_kv(file_path):
    config = {}
    with open(file_path, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue

            if '=' in line:
                key, value = line.split('=', 1)
                key = key.strip()
                value = value.strip()
                if value.isdigit():
                    config[key] = int(value)
                else:
                    config[key] = float(value)
    return config


def parse_csv(file_path):
    with open(file_path, 'r') as f:
        lines = f.readlines()

    proc = []
    for line in lines[1:]:
        vals = line.strip().split(',')
        lbl = int(vals[0])
        pix = [int(v) / 255.0 for v in vals[1:]]
        img = []
        for row in range(28):
            img_row = []
            for col in range(28):
                img_row.append(pix[row * 28 + col])
            img.append(img_row)
        proc.append((img, lbl))
    return proc


def one_hot(label, n_cls):
    hot = [0.0] * n_cls
    hot[label] = 1.0
    return hot


class DataProc:
    def __init__(self, cfg_file):
        self.specs = load_config_to_kv(cfg_file)
        print(self.specs)

    def shuffle(self, data):
        idx = rng.shuffle(len(data))
        return [data[i] for i in idx]

    def chunk(self, dataset, chunk_sz):
        chunks = []
        for i in range(0, len(dataset), chunk_sz):
            bx, by = [], []
            for j in range(i, min(i + chunk_sz, len(dataset))):
                img, lbl = dataset[j]
                bx.append(img)
                by.append(one_hot(lbl, self.specs['classes']))
            chunks.append((bx, by))
        return chunks

    def load_data(self, train_f, test_f):
        print("um ba la-ing train data")
        tr_raw = parse_csv(train_f)
        tr_raw = self.shuffle(tr_raw)
        tr_data = tr_raw[:self.specs['train_max']]
        
        print("um ba la-ing test data")
        te_raw = parse_csv(test_f)
        te_raw = self.shuffle(te_raw)
        te_data = te_raw[:self.specs['test_max']]
        
        tr_batches = self.chunk(tr_data, self.specs['batch'])
        te_batches = self.chunk(te_data, self.specs['batch'])
        
        return {
            'training': tr_batches,
            'testing': te_batches,
            'config': self.specs
        } 