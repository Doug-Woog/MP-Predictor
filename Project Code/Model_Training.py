from sklearn.linear_model import LinearRegression,SGDRegressor
from sklearn import svm,preprocessing
from sklearn.metrics import mean_squared_error,SCORERS,r2_score
from sklearn.model_selection import GridSearchCV, cross_val_score, KFold,train_test_split
from sklearn.neural_network import MLPRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.utils import shuffle
from sklearn.pipeline import make_pipeline
from sklearn.tree import DecisionTreeRegressor
from sklearn.kernel_approximation import Nystroem
import numpy as np
import statistics
import pandas as pd
import math
from joblib import dump, load
import matplotlib.pyplot as plt
import seaborn as sns; sns.set()

def equalise(df1,df2): #Input pandas dataframes
    """
    Takes in two datasets, removes any features that are not common between them and returns the modified datasets
    Parameters
    ----------
    df1 : pandas dataframe
        A dataset
    df2 : pandas dataframe
        Another dataset

    Returns
    -------
    df1 and df2 with any unshared features removed
    """
    result1 = df1.drop(df1.columns.difference(df2.columns),axis=1)
    result2 = df2.drop(df2.columns.difference(df1.columns),axis=1)
    return result1,result2 #Dataframes with only descriptor columns contained within both orginals dataframes



class model: #Main model class
    def __init__(self,modeltype,PCA=None,modelparams = None):
        self.information = {}
        if PCA:
            fitters = load(PCA)
            self.pca = fitters[0]
            self.scaler = fitters[1]
        else:
            self.pca = None
        if modeltype == 'Linear': #Linear Regression
            self.model = LinearRegression(n_jobs=-1)
        elif modeltype == 'SVM': #Support Vector Machine
            self.model= svm.SVR(cache_size=750,C=200)
        elif modeltype == 'LinearSVM': #Linear SVM
            self.model = svm.LinearSVR()
        elif modeltype == 'SGD': #Stochastic Gradient Descent
            self.model = SGDRegressor()
        elif modeltype == 'MLP': #Multi-layer Perceptron
            self.model = MLPRegressor(learning_rate='adaptive',max_iter=1000) 
        elif modeltype == 'KNN': #K Nearest Neighbour
            self.model = KNeighborsRegressor(n_neighbors=2,n_jobs=-1)
        elif modeltype == 'Tree': #Decision Tree
            self.model = DecisionTreeRegressor()
        elif modeltype == 'load': #Load a pre-existing model
            pass
        else: #Not supported
            print('Model type not recognised')
        if modelparams:
            self.model.set_params(**modelparams)

    def convert_arrays(self,datadf): #Convert pd dataframes to numpy ndarrays
        """
        Converts pd.Dataframe to np.ndarray
        Parameters
        ----------
        datadf : pd.Dataframe
            Dataframe of MP's and descriptors and SMILES
        Returns
        -------
        X : np.ndarray
            Descriptor values array
        Y : np.ndarray
            MP values
        """
        if isinstance(datadf, pd.DataFrame):
            Y = datadf['MP'].to_numpy()
            X = datadf.drop(['SMILES','MP'],axis=1)
            self.descrips = X.keys()
            X = X.to_numpy()
            #X,Y = shuffle(X,Y,random_state=None)
        else:
            X = datadf[0]
            Y = datadf[1]
        return X,Y

    def split_data(self,data,split):
        """
        Splits data into train and test data
        Parameters
        ----------
        split : float between 0 and 1
            Proportion of dataset to create train data
        Returns
        -------
        Training Data : list
            Training data as a list of [X,Y] where X and Y are numpy arrays of the descriptor and MP values respectively
        Test Data : list
            Test data as a list of [X,Y] where X and Y are numpy arrays of the descriptor and MP values respectively
        """
        X,Y = self.convert_arrays(data)
        datas = train_test_split(X,Y,train_size=split)
        return [datas[0],datas[2]],[datas[1],datas[3]]


    def printinfo(self,X):
        """
        Prints info about the current model
        Parameters
        ----------
        X : numpy array
            Dataset descriptors array
        Returns
        -------
        Prints various information about the model and dataset
        Dictionary holding all this information
        """
        modelname = type(self.model).__name__
        print("Model: "+modelname)
        parameters = self.model.get_params()
        print("Model Parameters: "+str(parameters))
        if self.pca:
            print("PCA: True")
            PCA = True
        else:
            print("PCA: False")
            PCA = False
        samples = np.size(X,0)
        features = np.size(X,1)
        print("Dataset # of samples: "+str(samples))
        print("Dataset # of features: "+str(features))
        return {'Model Type':modelname,'Model Parameters':parameters,'Samples':samples,'Features':features,'PCA':PCA}

    def crossValidate(self,training_data,folds=5): #Cross Validate model
        """
        Performs cross validation using the current model
        Parameters
        ----------
        training_data : np.array or pd.Dataframe
            Dataset to perform cross validation on
        folds : integer
            Number of folds
        """
        X,Y = self.convert_arrays(training_data)
        if self.pca:
            X_scaled = self.scaler.transform(X)
            X = self.pca.transform(X_scaled)
        modelPipeline = make_pipeline(preprocessing.StandardScaler(),self.model)
        kf = KFold(n_splits=folds,shuffle=True)
        cvScore = cross_val_score(modelPipeline,X,Y,scoring='neg_root_mean_squared_error',cv=kf,n_jobs=-1,verbose=1)
        print("CROSS VALIDATION")
        self.printinfo(X)
        print("Cross validated score (RMSE): "+str(cvScore))
        print("Mean = "+str(statistics.mean(cvScore))+"\n")
            

    def train_model(self,training_data): #Train model on inputted dataset
        """
        Trains model on inputted dataset
        Parameters
        ----------
        training_data : np.array or pd.Dataframe
            Data to train the model on
        """
        X,Y = self.convert_arrays(training_data)
        if self.pca:
            X_scaled = self.scaler.transform(X)
            X = self.pca.transform(X_scaled)

        else:
            self.scaler = preprocessing.StandardScaler().fit(X)
            X = self.scaler.transform(X)
        self.model.fit(X,Y)
        print("TRAINING")
        self.information['Training'] = self.printinfo(X)
        print("R^2 Score = "+str(self.model.score(X, Y)))
        predicted = self.model.predict(X)
        RMSE = mean_squared_error(Y, predicted,squared=False)
        print("RMSE = "+str(RMSE)+"\n")
        self.information['Training']['RMSE'] = RMSE
    
    def save_model(self,filepath): #Input filepath with filename included
        """
        Saves the model to a .joblib file
        Parameters
        ----------
        filepath : string
            The filepath to save the model to
        """
        #full_model = [self.model,self.scaler,self.descrips,self.pca]
        full_model = {'model':self.model,'scaler':self.scaler,'descriptors':self.descrips,'PCA':self.pca,'information':self.information}
        dump(full_model, filepath+'.joblib') #File extension is added automatically


    def load_model(self,file): #Load saved model
        """
        Loads a model from a .joblib file
        Parameters
        ----------
        file : string
            Filepath to load model from
        """
        models = load(file)
        self.model = models['model']
        self.scaler = models['scaler']
        self.descrips = models['descriptors']
        self.pca = models['PCA']
        self.information = models['information']

    def test_model(self,test_data): #Test model on test_data and return RMSE
        """
        Tests model on inputted dataset and returns the predicted values
        Parameters
        ----------
        test_data : np.array or pd.Dataframe
            Dataset to test the model on
        Returns
        -------
        Y : np.array
            Actual MP values
        predicted : np.array
            Predicted MP values
        """
        X,Y = self.convert_arrays(test_data)
        if self.pca:
            X_scaled = self.scaler.transform(X)
            X = self.pca.transform(X_scaled)
        else:
            X = self.scaler.transform(X)
        predicted = self.model.predict(X)
        print("TESTING")
        self.information['Testing'] = self.printinfo(X)
        print("R^2 = "+str(r2_score(Y,predicted)))
        RMSE = mean_squared_error(Y, predicted,squared=False)
        print("RMSE = "+str(RMSE)+"\n")
        self.information['Testing']['RMSE'] = RMSE
        return Y,predicted


    def gridsearch(self,test_data,params,save=None,graph=False): #Perform a gridsearch on test_data using params
        """
        Performs a cross validated gridsearch on a dataset with selected parameters
        Parameters
        ----------
        test_data : np.array or pd.Dataframe
            Dataset to use for the gridsearch
        params : dict
            Dictionary of parameter values to test
        save : string
            Filepath to save results to (defaults to None if not inputted)
        graph : boolean
            If true, creates graph of results (only works when one parameter is being varied)
        Returns
        -------
        Creates graph if graph = True
        Saves .txt of results if a save filepath is given
        """
        modelPipeline = make_pipeline(preprocessing.StandardScaler(),self.model)
        #print(modelPipeline.get_params().keys())
        gridcv = GridSearchCV(modelPipeline,param_grid=params,n_jobs=-1,scoring='neg_root_mean_squared_error',verbose=1)

        X,Y = self.convert_arrays(test_data)
        if self.pca:
            X_scaled = self.scaler.transform(X)
            X = self.pca.transform(X_scaled)
        gridcv.fit(X,Y)
        print("GRIDSEARCH")
        self.printinfo(X)
        print("Best Parameter : "+str(gridcv.cv_results_['params'][gridcv.best_index_]))
        print("RMSE: "+str(gridcv.cv_results_['mean_test_score'][gridcv.best_index_]))
        if graph == True:
            for param in params.keys():
                variable = param.split('__')[1]
                x_axis = (gridcv.cv_results_["param_"+param]).filled().astype(np.float64)
            y_axis = gridcv.cv_results_["mean_test_score"]
            std = gridcv.cv_results_["std_test_score"]
            sns.lineplot(x="param_"+param,y="mean_test_score",data=gridcv.cv_results_,color = 'red')
            plt.title("Gridsearch on "+type(self.model).__name__)
            plt.xlabel(variable)
            plt.ylabel("Negative RMSE /°C")
            
            plt.fill_between(x= x_axis,y1 = y_axis-std,y2 = y_axis+std,alpha=0.2,color= 'red')
            plt.show()
        if save: #Input filepath to save to if wanted
            pd.DataFrame.from_dict(gridcv.cv_results_, orient="index").to_csv(save+'.csv')

    def predictSingle(self,mol): #Return the predicted MP of single mol
        """
        Predicts the MP of a single molecule
        Parameters
        ----------
        mol : array
            Descriptor values for molecule
        Returns
        -------
        prediction : array
            Contains predicted MP of inputted molecule
        """
        if self.pca:
            X_scaled = self.scaler.transform(mol)
            X = self.pca.transform(X_scaled)
        else:
            X = self.scaler.transform(mol)
        prediction = self.model.predict(X)
        return prediction

    def getDescriptors(self): #Returns descriptors used in model
        """
        Returns the descriptors of the model as a list
        """
        return self.descrips
