'''
    Develop a function to visualize decision boundary
        for any classification models in 2D

    author: Pan Wu (ustcwupan@gmail.com)
'''
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches


def plot_decision_boundary(model, xrg=[-10, 10], yrg=[-10, 10],
                           Nx=100, Ny=100,
                           figsize=[6, 6], alpha=0.7):
    '''
    Plot decision boundary for any two dimension classification models
        in sklearn.

    Input:
        model: sklearn classification model class - already fitted
                (with "predict" and "predict_proba" method)
        xrg: xrange
        yrg: yrange
        Nx: x axis grid size
        Ny: y axis grid size

        figsize, alpha are parameters in matplotlib

    Output:
        matplotlib figure object
    '''

    # check model is legit to use
    try:
        getattr(model, 'predict')
    except:
        print "model do not have method predict 'predict' "
        return None
    try:
        getattr(model, 'predict_proba')
    except:
        print "model do not have method predict 'predict_proba' "
        return None

    # define color representation for each pure class
    colors = np.array([
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0],
        [1.0, 1.0, 0.0],
        [0.0, 1.0, 1.0],
        [1.0, 0.0, 1.0],
        [0.0, 0.0, 0.0]
    ])

    # generate grid, mesh, and X for model prediction
    xgrid = np.arange(xrg[0], xrg[1], 1. * (xrg[1] - xrg[0]) / Nx)
    ygrid = np.arange(yrg[0], yrg[1], 1. * (yrg[1] - yrg[0]) / Ny)

    xx, yy = np.meshgrid(xgrid, ygrid)
    X = np.array(zip(np.ravel(xx), np.ravel(yy)))

    # initialize figure & axes object
    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(1, 1, 1)

    # get data from model predictions
    Ypp = model.predict_proba(X)
    Yp = model.predict(X)

    # check nclass
    nclass = Ypp.shape[1]
    if nclass > colors.shape[0]:
        print 'Hard to visualize more than 10 classes.'
        return None

    # get decision boundary line
    Yp = Yp.reshape(xx.shape)
    Yb = np.zeros(xx.shape)

    Yb[:-1, :] = np.maximum((Yp[:-1, :] != Yp[1:, :]), Yb[:-1, :])
    Yb[1:, :] = np.maximum((Yp[:-1, :] != Yp[1:, :]), Yb[1:, :])
    Yb[:, :-1] = np.maximum((Yp[:, :-1] != Yp[:, 1:]), Yb[:, :-1])
    Yb[:, 1:] = np.maximum((Yp[:, :-1] != Yp[:, 1:]), Yb[:, 1:])

    # plot decision boundary first
    ax.imshow(Yb, origin='lower', interpolation=None, cmap='Greys',
              extent=[xrg[0], xrg[1], yrg[0], yrg[1]],
              alpha=1.0)

    # plot probability surface
    zz = np.dot(Ypp, colors[:nclass, :])
    zz_r = zz.reshape(xx.shape[0], xx.shape[1], 3)
    ax.imshow(zz_r, origin='lower', interpolation=None,
              extent=[xrg[0], xrg[1], yrg[0], yrg[1]],
              alpha=alpha)

    # add legend on each class
    colors_bar = []
    for v1 in colors[:nclass, :]:
        v1 = list(v1)
        v1.append(alpha)
        colors_bar.append(v1)

    # create a patch (proxy artist) for every color
    patches = [mpatches.Patch(color=colors_bar[i],
                              label="Class {k}".format(k=i))
               for i in range(nclass)]
    # put those patched as legend-handles into the legend
    plt.legend(handles=patches, bbox_to_anchor=(1.05, 1),
               loc=2, borderaxespad=0., framealpha=0.5)

    # make the figure nicer
    ax.set_xlim(xrg)
    ax.set_ylim(yrg)
    ax.set_xticks(np.arange(xrg[0], xrg[1], (xrg[1] - xrg[0])/10.))
    ax.set_yticks(np.arange(yrg[0], yrg[1], (yrg[1] - yrg[0])/10.))
    ax.grid(True)

    return fig
