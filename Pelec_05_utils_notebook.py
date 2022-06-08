""" Utils """
import os
import math
import warnings
import math
import datetime
import missingno as msno
import numpy as np
import pandas as pd
import seaborn as sns
from datetime import datetime
from hyperopt import tpe
from matplotlib import pyplot as plt
import statsmodels.api as sm
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import (StandardScaler,
                                   OneHotEncoder,
                                   LabelEncoder
                                  )
from sklearn.datasets import make_regression
from sklearn.ensemble import RandomForestClassifier
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.model_selection import (train_test_split,
                                     cross_val_score,
                                     StratifiedKFold,
                                     RepeatedStratifiedKFold,
                                     GridSearchCV
                                    )
from sklearn.metrics import (mean_squared_error, # Regression
                             explained_variance_score, # Regression
                             r2_score, # Regression
                             accuracy_score, # Classification
                             precision_score, # Classification
                             recall_score, # Classification
                             plot_confusion_matrix, # Classification
                             classification_report, # Classification
                            )
from sklearn.linear_model import (LinearRegression,
                                  Ridge,
                                  RidgeCV,
                                  Lasso
                                 )
from sklearn.svm import (LinearSVR,
                         SVR
                        )
from sklearn.preprocessing import LabelEncoder
from sklearn.neighbors import KNeighborsRegressor
from sklearn.pipeline import (Pipeline,
                              make_pipeline
                             )
#from lazypredict.Supervised import LazyClassifier
from string import ascii_letters


def info(dataframe):
    """Prints dataframe parameters

    Args:
        dataframe (pd.Dataframe): data source
    Returns:
        Prints parameters: number of columns, number of rows, rate of missing values
    """
    print(str(len(dataframe.columns.values)) + " columns" )
    print(str(len(dataframe)) + " rows")
    print("Rate of missing values in df : " + str(dataframe.isnull().mean().mean()*100) + " %")


def plot_grid(rows, cols, dataframe, kind) :
    """ Plots multiple given kind of plots in a grid

    Args:
        rows (int): number of rows
        cols (int): number of columns
        dataframe (pd.Dataframe): data source
        kind (str): kind of plot
    Returns:
         Multiple given kind of plots plotted in a grid
    """
    # Sublot
    nrows = rows
    ncols = cols
    fig, axes = plt.subplots(nrows = nrows, ncols = ncols, figsize = (20,35))

    # Plot
    nrows_ = 0
    ncols_ = 0
    i = 0

    for nrows_ in range(nrows) :
        for ncols_ in range(ncols) :
            if i < len(dataframe.columns) :
                (
                dataframe[dataframe.columns[i]]
                    .plot(kind = kind,
                          title = dataframe.columns[i],
                          xticks = [],
                          ax = axes[nrows_][ncols_]),

                )
            i = i + 1
            ncols_ = ncols_ + 1
        nrows_ = nrows_ + 1
    plt.tick_params(axis = "x", which = "both", bottom = False, top = False)
    plt.show()

def dataframe_correlation_graph(dataframe):
    """ Plots seaborn graph of correlations

    Args:
        dataframe (pd.Dataframe): data source

    Returns:
        Graph of correlations

    """
    # Setting theme
    sns.set_theme(style="white")

    # Compute the correlation matrix
    corr = dataframe.corr().round(2)

    # Generate a mask for the upper triangle
    mask = np.triu(np.ones_like(corr, dtype=bool))

    # Set up the matplotlib figure
    f, ax = plt.subplots(figsize=(20,20))

    # Generate a custom diverging colormap
    cmap = sns.diverging_palette(230, 20, as_cmap=True)

    # Draw the heatmap with the mask and correct aspect ratio
    sns.heatmap(corr, mask=mask, cmap=cmap, vmin=-1, vmax=1, center=0,
                square=True, linewidths=.5, cbar_kws={"shrink": .2}, annot = True)

    plt.show()


def pairs_correlation_ranking(dataframe):
    """ Ranking of correlation pairs

    Args:
        dataframe (pd.Dataframe): data source

    Returns:
        nd.Series: Ranking list of correlation pairs

    """
    # Correlation matrix dataframe
    df_corr = dataframe.corr().round(4)

    # Filter out identical pairs
    df_corr_filter = df_corr[df_corr != 1.000]

    # Create list and sort values
    series = (df_corr_filter.abs()
         .unstack()
         .drop_duplicates()
         .sort_values(kind="quicksort", ascending = False)
        ).head(20)

    # Show
    print("Pairs correlation ranking: \n\n" + str(series))

    return series


def visualization_distribution_features(dataframe):
    """
    """
    #
    X = dataframe[dataframe.columns].values
    fig = plt.figure(figsize=(25, 25))

    for feat_idx in range(X.shape[1]):
        ax = fig.add_subplot(7,4, (feat_idx+1))
        h = ax.hist(X[:, feat_idx], bins=100, color='steelblue', density=True, edgecolor='none')
        ax.set_title(dataframe.columns[feat_idx], fontsize=14)

def visualization_distribution_x_data_function(X_train):
    """
    """
    X_train_graph = X_train.copy().values
    fig = plt.figure(figsize=(25, 25))

    for feat_idx in range(X_train_graph.shape[1]):
        ax = fig.add_subplot(7,4, (feat_idx+1))
        h = ax.hist(X_train_graph[:, feat_idx], bins=100, color='steelblue', density=True, edgecolor='none')
        ax.set_title(X_train.columns[feat_idx], fontsize=14)


def inter_quartile_method_function(dataframe):
    """
    """
    #
    q1 = dataframe.quantile(0.25)
    q3 = dataframe.quantile(0.75)
    iqr = q3 - q1
    #
    dataframe = dataframe[(dataframe <= dataframe.quantile(0.75) + 1.5*iqr)
                                                        & (dataframe >= dataframe.quantile(0.25) - 1.5*iqr)]

    #
    return dataframe


def display_circles(pcs, n_comp, pca, axis_ranks, labels=None, label_rotation=0, lims=None):
    """ Plots PCA circle of correlation

    Args:
        pcs (numpy.ndarray):
        n_comp (int):
        pca (sklearn.decomposition._pca.PCA):
        axis_ranks (list):

    Returns:
        Plot
    """
    for d1, d2 in axis_ranks: # On affiche les 3 premiers plans factoriels, donc les 6 premières composantes
        if d2 < n_comp:

            # initialisation de la figure
            fig, ax = plt.subplots(figsize=(7,6))

            # détermination des limites du graphique
            if lims is not None :
                xmin, xmax, ymin, ymax = lims
            elif pcs.shape[1] < 30 :
                xmin, xmax, ymin, ymax = -1, 1, -1, 1
            else :
                xmin, xmax, ymin, ymax = min(pcs[d1,:]), max(pcs[d1,:]), min(pcs[d2,:]), max(pcs[d2,:])

            # affichage des flèches
            # s'il y a plus de 30 flèches, on n'affiche pas le triangle à leur extrémité
            if pcs.shape[1] < 30 :
                plt.quiver(np.zeros(pcs.shape[1]), np.zeros(pcs.shape[1]),
                   pcs[d1,:], pcs[d2,:],
                   angles='xy', scale_units='xy', scale=1, color="grey")
                # (voir la doc : https://matplotlib.org/api/_as_gen/matplotlib.pyplot.quiver.html)
            else:
                lines = [[[0,0],[x,y]] for x,y in pcs[[d1,d2]].T]
                ax.add_collection(LineCollection(lines, axes=ax, alpha=.1, color='black'))

            # affichage des noms des variables
            if labels is not None:
                for i,(x, y) in enumerate(pcs[[d1,d2]].T):
                    if x >= xmin and x <= xmax and y >= ymin and y <= ymax :
                        plt.text(x,
                                 y,
                                 labels[i],
                                 fontsize='14',
                                 ha='center',
                                 va='center',
                                 rotation=label_rotation,
                                 color="blue",
                                 alpha=0.5
                                )

            # affichage du cercle
            circle = plt.Circle((0,0), 1, facecolor='none', edgecolor='b')
            plt.gca().add_artist(circle)

            # définition des limites du graphique
            plt.xlim(xmin, xmax)
            plt.ylim(ymin, ymax)

            # affichage des lignes horizontales et verticales
            plt.plot([-1, 1], [0, 0], color='grey', ls='--')
            plt.plot([0, 0], [-1, 1], color='grey', ls='--')

            # nom des axes, avec le pourcentage d'inertie expliqué
            plt.xlabel('F{} ({}%)'.format(d1+1, round(100*pca.explained_variance_ratio_[d1],1)))
            plt.ylabel('F{} ({}%)'.format(d2+1, round(100*pca.explained_variance_ratio_[d2],1)))

            plt.title("Cercle des corrélations (F{} et F{})".format(d1+1, d2+1))
            plt.show(block=False)


def display_factorial_planes(X_projected, n_comp, pca, axis_ranks, labels=None, alpha=1, illustrative_var=None):
    """ Plots PCA factorial planes

    Args:
        X_projected (numpy.ndarray):
        n_comp (int): number of components to compute
        pca (sklearn.decomposition._pca.PCA):
        axis_ranks (list):

    Returns:
        Plot

    """
    for d1,d2 in axis_ranks:
        if d2 < n_comp:

            # initialisation de la figure
            plt.rc('axes', unicode_minus=False)
            fig = plt.figure(figsize=(7,6))


            # affichage des points
            if illustrative_var is None:
                plt.scatter(X_projected[:, d1], X_projected[:, d2], alpha=alpha)
            else:
                illustrative_var = np.array(illustrative_var)
                for value in np.unique(illustrative_var):
                    selected = np.where(illustrative_var == value)
                    plt.scatter(X_projected[selected, d1],
                                X_projected[selected, d2],
                                alpha=alpha,
                                label=value
                               )
                plt.legend()

            # affichage des labels des points
            if labels is not None:
                for i,(x,y) in enumerate(X_projected[:,[d1,d2]]):
                    plt.text(x, y, labels[i], fontsize='14', ha='center',va='center')

            # détermination des limites du graphique
            boundary = np.max(np.abs(X_projected[:, [d1,d2]])) * 1.1
            plt.xlim([-boundary,boundary])
            plt.ylim([-boundary,boundary])

            # affichage des lignes horizontales et verticales
            plt.plot([-100, 100], [0, 0], color='grey', ls='--')
            plt.plot([0, 0], [-100, 100], color='grey', ls='--')

            # nom des axes, avec le pourcentage d'inertie expliqué
            plt.xlabel('F{} ({}%)'.format(d1+1, round(100*pca.explained_variance_ratio_[d1],1)))
            plt.ylabel('F{} ({}%)'.format(d2+1, round(100*pca.explained_variance_ratio_[d2],1)))

            plt.title("Projection des individus (sur F{} et F{})".format(d1+1, d2+1))
            plt.show(block=False)


def display_scree_plot(pca):
    """ Plots PCA scree plot

    Args:
        pca (sklearn.decomposition._pca.PCA):

    Returns:
        Plot

    """
    scree = pca.explained_variance_ratio_*100
    plt.figure(figsize=(7,6))
    plt.bar(np.arange(len(scree))+1, scree)
    plt.plot(np.arange(len(scree))+1, scree.cumsum(),c="red",marker='o')
    plt.xlabel("Rang de l'axe d'inertie")
    plt.ylabel("Pourcentage d'inertie")
    plt.title("Eboulis des valeurs propres")
    plt.show(block=False)

    
    
    
    