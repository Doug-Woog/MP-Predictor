from rdkit import Chem
from mordred import descriptors
import mordred as m
from multiprocessing import freeze_support
import pandas as pd
import os
from sklearn.decomposition import PCA
import numpy as np
from joblib import dump, load
from sklearn import preprocessing
import tempfile


def generateDescriptors(dataset,filename,descs=None,big=None): 
    """
    Generates descriptor .csv file for given dataset
    Note: Run inside of if __name__ == "__main__":
    Parameters
    ----------
    dataset : string
        Filepath of .csv dataset to generate descriptors for
    filename : string
        Filepath to save new descriptor dataset to
    decs : dict
        Dictionary of mordred chemical descriptors
        Defaults to using all Mordred descriptors if not specified
    big : integer
        The size of each batch size
        Only calculates in batches if big is specified
    """
    
    data = pd.read_csv(dataset)
    mols = [Chem.MolFromSmiles(smi) for smi in data['SMILES']]

    calc = m.Calculator()
    if descs:
        for mod in descs:
            calc.register(mod)
    else:
        calc.register(descriptors)
    if big:
        
        with tempfile.NamedTemporaryFile() as temp:
            df = pd.DataFrame()
            df.to_csv(temp.name + '.csv')
            for i in range(0, len(mols), big):
                molcalc =calc.pandas(mols[i:i + big])
                molcalc.index = data['SMILES'][i:i + big]
                frame = pd.read_csv(temp.name + '.csv')
                frame = frame.append(molcalc,ignore_index=False)
                frame.to_csv(temp.name + '.csv')
                frame = None
                molcalc = None
            df = pd.read_csv(temp.name + '.csv')
            df = df.dropna(axis=1)
            df = df._get_numeric_data()
            df['MP'] = data['Melting Point {measured, converted}']
            df.index = data['SMILES']
    else:
        df = calc.pandas(mols)
        df = df._get_numeric_data()
        df['MP'] = data['Melting Point {measured, converted}']
        df.index = data['SMILES']
    
    if descs:
        nameString = filename+" descriptors = "+str(list(d.__name__.strip("mordred.") for d in descs))+" .csv"
    else:
        nameString = filename+" descriptors = All .csv"
    df.to_csv(nameString)


def PCA_fit(dataset,percent,save_path):
    """
    Fits a Principal Component Analysis transformation to the inputted dataset
    Parameters
    ----------
    dataset : string
        Filepath of .csv containing dataset
    percent : float between 0 and 1
        The percentage of the variance in the original dataset to be retained
    save_path : string
        Filepath to save the PCA transformation to
    """
    dataframe = pd.read_csv(dataset)
    X = dataframe.drop(['SMILES','MP'],axis=1).to_numpy()
    scaler = preprocessing.StandardScaler().fit(X)
    X = scaler.transform(X)
    pca = PCA(n_components=percent)
    pca.fit(X)
    print(pca.explained_variance_ratio_)
    print(len(pca.explained_variance_ratio_))
    dump([pca,scaler],save_path+'.joblib')

