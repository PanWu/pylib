'''
    Develop a function to visualize decision boundary
        for any classification models in 2D

    author: Pan Wu (ustcwupan@gmail.com)
'''
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from sklearn.decomposition import PCA, KernelPCA
from sklearn.preprocessing import StandardScaler


def plot_decision_boundary(model, dim_red_method='pca',
                           X=None, Y=None,
                           xrg=None, yrg=None,
                           Nx=300, Ny=300,
                           figsize=[6, 6], alpha=0.7):
    '''
    Plot decision boundary for any two dimension classification models
        in sklearn.

    Input:
        model: sklearn classification model class - already fitted
                (with "predict" and "predict_proba" method)

        dim_red_method: sklearn dimension reduction model
                (with "fit_transform" and "inverse_transform" method)

        xrg (list/tuple): xrange
        yrg (list/tuple): yrange
        Nx (int): x axis grid size
        Ny (int): y axis grid size

        X (nparray): dataset to project over decision boundary (X)
        Y (nparray): dataset to project over decision boundary (Y)

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
        [0.0, 0.0, 0.0],
        [0.0, 0.0, 0.75],
        [0.0, 0.75, 0.0],
        [0.75, 0.0, 0.0]
    ])

    # convert X into 2D data
    if X is not None:
        if X.shape[1] == 2:
            X2D = X
            ss, dr_model = None, None
        elif X.shape[1] > 2:
            # leverage PCA to dimension reduction to 2D if not already
            ss = StandardScaler()
            if dim_red_method == 'pca':
                dr_model = PCA(n_components=2)
            elif dim_red_method == 'kernal_pca':
                dr_model = KernelPCA(n_components=2,
                                     fit_inverse_transform=True)
            else:
                print 'dim_red_method {0} is not supported'.format(
                    dim_red_method)

            X2D = dr_model.fit_transform(ss.fit_transform(X))
        else:
            print 'X dimension is strange: {0}'.format(X.shape)
            return None

        # extract two dimension info.
        x1 = X2D[:, 0].min() - 0.1 * (X2D[:, 0].max() - X2D[:, 0].min())
        x2 = X2D[:, 0].max() + 0.1 * (X2D[:, 0].max() - X2D[:, 0].min())
        y1 = X2D[:, 1].min() - 0.1 * (X2D[:, 1].max() - X2D[:, 1].min())
        y2 = X2D[:, 1].max() + 0.1 * (X2D[:, 1].max() - X2D[:, 1].min())

    # convert Y into point color
    if Y is not None:
        # presume Y is labeled from 0 to N-1
        cY = [colors[i] for i in Y]

    # inti xrg and yrg based on given value
    if xrg is None:
        if X is None:
            xrg = [-10, 10]
        else:
            xrg = [x1, x2]

    if yrg is None:
        if X is None:
            yrg = [-10, 10]
        else:
            yrg = [y1, y2]

    # generate grid, mesh, and X for model prediction
    xgrid = np.arange(xrg[0], xrg[1], 1. * (xrg[1] - xrg[0]) / Nx)
    ygrid = np.arange(yrg[0], yrg[1], 1. * (yrg[1] - yrg[0]) / Ny)

    xx, yy = np.meshgrid(xgrid, ygrid)
    X_full_grid = np.array(zip(np.ravel(xx), np.ravel(yy)))

    # initialize figure & axes object
    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(1, 1, 1)

    # get data from model predictions
    if dr_model is None:
        Ypp = model.predict_proba(X_full_grid)
        Yp = model.predict(X_full_grid)
    else:
        X_full_grid_inverse = ss.inverse_transform(
            dr_model.inverse_transform(X_full_grid))

        Ypp = model.predict_proba(X_full_grid_inverse)
        Yp = model.predict(X_full_grid_inverse)

    # check nclass
    nclass = Ypp.shape[1]
    if nclass > colors.shape[0]:
        print 'Hard to visualize more than {0} classes.' \
            .format(colors.shape[0])
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

    # add scatter plot for X & Y if given
    if X is not None:
        if Y is not None:
            # print 'scatter plot', X.shape, cY.shape
            ax.scatter(X2D[:, 0], X2D[:, 1], c=cY)
        else:
            ax.scatter(X2D[:, 0], X2D[:, 1])

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
    ax.set_title('Classification decision boundary')
    if dr_model is None:
        ax.set_xlabel('Raw axis X')
        ax.set_ylabel('Raw axis Y')
    else:
        ax.set_xlabel('Dimension reduced axis 1')
        ax.set_ylabel('Dimension reduced axis 2')
    ax.set_xlim(xrg)
    ax.set_ylim(yrg)
    ax.set_xticks(np.arange(xrg[0], xrg[1], (xrg[1] - xrg[0])/5.))
    ax.set_yticks(np.arange(yrg[0], yrg[1], (yrg[1] - yrg[0])/5.))
    ax.grid(True)

    return fig, xrg, yrg
