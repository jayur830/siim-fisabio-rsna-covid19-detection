import numpy as np


def tp():
    pass


def fp():
    pass


def fn():
    pass


def tn():
    pass


def precision():
    tp = tp()
    fp = fp()
    return tp / (tp + fp)


def recall():
    tp = tp()
    fn = fn()
    return tp / (tp + fn)


def ap():
    ap = {
        "precision": [],
        "recall": []
    }
    for confidence in list(reversed(np.arange(0, 1, 0.1).tolist())):
        pass
    pass
