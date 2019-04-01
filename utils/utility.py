#coding=utf-8
import torch

def repackage_hidden(h):
    """Wraps hidden states in new Tensors, to detach them from their history."""
    if isinstance(h, torch.Tensor):
        new_h = torch.Tensor(h.data)
        new_h.zero_()
        del h
        return new_h
    else:
        return tuple(repackage_hidden(v) for v in h)


def compute_measure(predict, label, thresh):
    '''Compute precision, recall, f1 and accuracy'''
    tn, fp, fn, tp = 0, 0, 0, 0
    for i in range(len(predict)):
        if predict[i] >= thresh and int(label[i]) == 1:
            tp += 1
        elif predict[i] >= thresh and int(label[i]) == 0:
            fp += 1
        elif predict[i] < thresh and int(label[i]) == 1:
            fn += 1
        else:
            tn += 1
    return tp, fp, fn, tn
