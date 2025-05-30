import math


class RNG:
    def __init__(self, seed):
        self.s = seed
        self.c = seed

    def next(self):
        self.c = (self.c * 1103515245 + 12345) & 0x7fffffff
        return self.c

    def uniform(self, a, b):
        r = self.next() / 0x7fffffff
        return a + r * (b - a)

    def randint(self, a, b):
        r = self.next() / 0x7fffffff
        return int(a + r * (b - a + 1))

    def shuffle(self, n):
        idx = list(range(n))
        for i in range(n - 1, 0, -1):
            j = self.randint(0, i)
            idx[i], idx[j] = idx[j], idx[i]
        return idx


rng = RNG(seed=66991122)


def matrix_mult_basic(A, B):
    rows_A, cols_A = len(A), len(A[0])
    rows_B, cols_B = len(B), len(B[0])
    result = [[0 for _ in range(cols_B)] for _ in range(rows_A)]
    for i in range(rows_A):
        for j in range(cols_B):
            for k in range(cols_A):
                result[i][j] += A[i][k] * B[k][j]
    return result


def flatten_3d_tensor(tensor):
    flat = []
    for batch in tensor:
        flat_batch = []
        for channel in batch:
            for row in channel:
                for val in row:
                    flat_batch.append(val)
        flat.append(flat_batch)
    return flat


def apply_relu_activation(data):
    if isinstance(data[0][0], list):
        result = []
        for batch in data:
            batch_result = []
            for channel in batch:
                channel_result = []
                for row in channel:
                    row_result = [max(0, val) for val in row]
                    channel_result.append(row_result)
                batch_result.append(channel_result)
            result.append(batch_result)
        return result
    else:
        result = []
        for batch in data:
            batch_result = [max(0, val) for val in batch]
            result.append(batch_result)
        return result


def compute_softmax_stable(logits):
    result = []
    for batch in logits:
        max_val = max(batch)
        exp_vals = [math.exp(x - max_val) for x in batch]
        sum_exp = sum(exp_vals)
        probs = [exp_val / sum_exp for exp_val in exp_vals]
        result.append(probs)
    return result


def shuffle_data_indices(data_len):
    return rng.shuffle_indices(data_len)


def calculate_conv_output_dim(input_size, kernel_size, stride=1, padding=0):
    return (input_size + 2 * padding - kernel_size) // stride + 1


def init_weights_xavier(fan_in, fan_out):
    limit = math.sqrt(6.0 / (fan_in + fan_out))
    return rng.uniform(-limit, limit)


def apply_gradient_clipping(gradients, max_norm=1.0):
    total_norm = 0
    for grad_batch in gradients:
        for grad in grad_batch:
            total_norm += grad * grad
    total_norm = math.sqrt(total_norm)

    if total_norm > max_norm:
        clip_coeff = max_norm / total_norm
        for i in range(len(gradients)):
            for j in range(len(gradients[i])):
                gradients[i][j] *= clip_coeff

    return gradients
