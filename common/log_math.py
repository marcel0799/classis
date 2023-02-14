import numpy as np

def log_normal_distr(data, mean, cov, weights=None):
    p = np.zeros((data.shape[0], mean.shape[0]))

    for index in np.arange(mean.shape[0]):
        data_i = data - mean[index, :]

        sign_det, log_det = np.linalg.slogdet(cov[index])
        p[:, index] = -0.5 * (np.sum(np.dot(
                data_i, np.linalg.inv(cov[index])) * data_i, axis=1) +
                data.shape[1] * np.log(2 * np.pi) + sign_det * log_det)
        if weights is not None:
            p[:, index] += np.log(weights[0, index])

    return p


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