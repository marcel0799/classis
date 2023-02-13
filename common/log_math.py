import numpy as np



def logsumexp(arr, axis=0):
    """"Berechnet die Summe der Elemente in arr in der log-Domain:
    log(sum(exp(arr))). Dabei wird das Risiko eines Zahlenueberlaufs reduziert.

    Params:
        arr: ndarray von Werten in der log-Domaene
        axis: Index der ndarray axis entlang derer die Summe berechnet wird.

    Returns:
        out: ndarray mit Summen-Werten in der log-Domaene.
    """
    arr = np.rollaxis(arr, axis)
    vmax = arr.max(axis=0)
    out = np.log(np.sum(np.exp(arr - vmax), axis=0))
    out += vmax
    return out
