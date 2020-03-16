from Model_Training import model, equalise
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns; sns.set()
import numpy as np
from sklearn.metrics import mean_squared_error



def RMSERange(y,predicted,size,modelname):
    """
    Produces a graph of RMSE over a range of MP's
    Parameters
    ----------
    y : array
        Actual values
    predicted : array
        Predicted values
    size : integer
        Number of bins to split data into
    modelname : string
        Name of model
    Returns
    -------
    Line graph
    """
    maxi = max(y)+1
    mini = min(y)
    bins = np.linspace(mini,maxi,size)
    ind =  np.digitize(y, bins)
    binnedy = []
    binnedx = []
    middlebins = []
    for i in range(size-1):
        mid = (bins[i+1]+bins[i])/2
        middlebins.append(mid)
        binnedy.append([])
        binnedx.append([])
    for n in range(y.size):
        binnedy[ind[n-1]-1].append([y[n-1]])
        binnedx[ind[n-1]-1].append([predicted[n-1]])
    rmses = []
    for act,pred in zip(binnedy,binnedx):
        RMSE = mean_squared_error(act,pred,squared=False)
        rmses.append(RMSE)
    plt.plot(middlebins,rmses)
    plt.xlabel("Temperature /°C")
    plt.ylabel("RMSE /°C")
    plt.title("RMSE of "+modelname+" against Temperature")
    plt.show()


def scatter(y,predicted):
    """
    Produces a scatter plot of predictions against actual
    Parameters
    ----------
    y : array
        Actual values
    predicted : array
        Predicted values
    Returns
    -------
    Scatter plot graph
    """
    fig, ax = plt.subplots()
    ax.scatter(y, predicted, edgecolors=(0, 0, 0),s=13,color='red')
    ax.plot([y.min(), y.max()], [y.min(), y.max()], 'k--', lw=4)
    ax.set_xlabel('Measured Melting Point /°C')
    ax.set_ylabel('Predicted Melting Point /°C')
    plt.title("DecisionTree Measured vs Predictions")
    plt.show()