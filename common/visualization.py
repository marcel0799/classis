from matplotlib import cm
import numpy as np

def plot_norm_dist_ellipse(ax, means, cov_list, color):
    """
    Parameter:
    ax -- Axes-Objekt, in das geplottet werden soll.
    means -- Mittelwerte der Normalverteilungen als numpy-array;
             Die Mittelwerte sind zeilenweise gespeichert.
    cov_list -- Kovarianzmatrizen der Normalverteilungen als Liste.
    color -- Farben, in denen die Ellipsen geplottet werden sollen, als Liste oder tuple.
    """
    for index, cov in enumerate(cov_list):
        val, vec = np.linalg.eig(cov)
        phi = -np.arccos(vec[0, 0])
        #phi = np.arctan2(vec[0, 1],vec[0, 0])


        th = np.linspace(0, 2 * np.pi, 100)
        x = val[0] * np.cos(th)
        y = val[1] * np.sin(th)
        c = np.cos(phi)
        s = np.sin(phi)
        X = x * c - y * s + means[index][0]
        Y = x * s + y * c + means[index][1]
        ax.plot(X, Y, color=color[index], linewidth=3)


def plot_svm(ax, data, labels, svm, step_size=0.1):
    """Visualisieren einer SVM Hyperebene

    Die Daten in data werden als scatter geplottet und anhand labels eingefarbt.
    Der Hintergrund wird anhand des Labels eingefaerbt, das durch svm.predict
    fuer diesen Bereich geschaetzt wird.

    Parameter:
    ax -- Axes-Objekt, in das geplottet werden soll.
    data -- 2D-Datenpunkte
    labels -- Label der Datenpunkte
    svm -- Trainiertes SVC-Objekt von sklearn
    step_size -- Abstand des Grids, mit dem die Hyperebene visualisiert wird (default 0.1)
    """
    predict_func = svm.predict
    plot_hyperplane(ax, data, labels, predict_func, step_size)


def plot_hyperplane(ax, data, labels, predict_func, step_size=0.1):
    """Visualisieren einer Hyperebene zur Trennung zweier Klassen durch Abtasten des Merkmalsraums.

    Parameter:
    siehe plot_svm
    predict_func -- Ein Funktionsobjekt, das Merkmale (Nxdim) zu Klassenlabels transformiert (Nx1)
    """
    labels = labels.astype(int)
    label_max = float(labels.max()) + 1
    data_min = data.min(axis=0) - 1
    data_max = data.max(axis=0) + 1

    xx, yy = np.meshgrid(np.arange(data_min[0], data_max[0], step_size),
                         np.arange(data_min[1], data_max[1], step_size))
    zz = predict_func(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)
    zz = np.array(zz, dtype=int)
    cmap = cm.get_cmap('gist_rainbow')
    c = cmap(zz / label_max)
    ax.imshow(c, interpolation='nearest',
              extent=(data_min[0], data_max[0], data_min[1], data_max[1]),
              origin='lower', aspect='auto')
    c = cmap(labels / label_max)
    ax.scatter(data[:, 0], data[:, 1], c=c, edgecolor=(0, 0, 0))
    ax.set_xlim(data_min[0], data_max[0])
    ax.set_ylim(data_min[1], data_max[1])
