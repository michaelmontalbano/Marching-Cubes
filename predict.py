import math
import numpy as np
import pandas as pd
from itertools import cycle
import matplotlib.pyplot as plt
import operator
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV
import sys
from pprint import pprint


# PLAN

# show model predicted size vs observed size
# mean absolute error
# bullet point about how well it performed
# human generated manual interrogated 
# merge with something more subjective that relies on the knowledge of a forecaster




df = pd.read_csv('C:\\Users\\User\\weather\\alldata.csv')

df = df.drop(columns = ["Unnamed: 0","BWER","Direction","Latitude","Longitude","TBSS","WER","b_height","eov","pointval","timeStamp","time"])

for col in df.columns:
    if col[-4:] == 'shes':
        df = df.drop(columns=[col])
    if col[-4:] == 'ects':
        df = df.drop(columns=[col])
    if col[:4] == 'SWDI':
        df = df.drop(columns=[col])
    if col[-5:] == 'width':
        df = df.drop(columns=[col])
    if col[-8:] == 'centroid':
        df = df.drop(columns=[col])

#clean_df = df[['maxEst','zdrColumn_numVol','zdrColumn_totalVolume','zdrColumn_MergedDZDR_max','zdrColumn_depth','zdrColumn_width','zdrColumn_specificVolume',
#        'zdrColumn_MergedDZDR_corr','zdrColumn_MergedReflectivityQC_corr','zdrColumn_MergedDRHO_corr']]

#clean_df = df[['maxEst','zdrColumn_totalVolume','zdrColumn_MergedDZDR_max','zdrColumn_specificVolume','zdrColumn_MergedReflectivityQC_corr','zdrColumn_MergedDZDR_corr','SWDI_VIL','SWDI_POSH','TBSS','WER']]

#df = clean_df
#clean_df[clean_df<0] = 0
# def label_hail(column):
#         if column[0] > 0.3:
#             return 1
#         if column[0] <= 0.3:
#             return 0

# sets binary column 'hail' which is 1 if hail, 0 otherwise

def rescale(df):
    rescale_df = df
    for (columnName, columnData) in rescale_df.iteritems(): 
        rescale_df[columnName] = (df[columnName] - min(df[columnName])) / (max(df[columnName]) - min(df[columnName]))
    return rescale_df

def split_data(df, prob):
    msk = np.random.rand(len(df)) < prob
    train = df[msk]
    test = df[~msk]
    return train, test

#rescale_df = rescale(clean_df)
#rescale_df['hail'] = rescale_df.apply (lambda column: label_hail(column), axis = 1)

# train, test = split_data(rescale_df, 0.75)

# y_train = train["maxEst"]
# x_train = train.drop(columns={"hail","maxEst"})
# x_test = test.drop(columns={"hail","maxEst"})
# y_test = test["maxEst"]

# dataset = train.drop(columns={'maxEst'})
# dataset = dataset.to_numpy()
# x_train = x_train.to_numpy()
# y_train = y_train.to_numpy()
# x_test = x_test.to_numpy()
# y_test = y_test.to_numpy()
# dataset = train.drop(columns={'maxEst'})


import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import (GradientBoostingClassifier,
                              GradientBoostingRegressor)
from sklearn.metrics import log_loss, mean_absolute_error
from sklearn.model_selection import StratifiedKFold, StratifiedShuffleSplit, train_test_split, GridSearchCV
​
##GET DATA AND CLEAN IT UP
#read in our data
#data = pd.read_csv('TheClusters_SHAVE_Soundings.csv', sep=",")
data = df
data = data.replace([-99900., -99903], 0.) # replace any WDSS-II missing values with 0
targets = data["maxEst"] #these are the data we are going to try to predict
#drop the RowName column, it's not helpful nor really data and also drop our targets out of the data


#need to make our targets a class
#this is needed for both the classifier and for splitting the data so we have representative samples (See below)
targets['maxEst'] = pd.cut(targets["maxEst"], bins=[-1,0.254,25.3,50.7,200.],include_lowest=True,right=False,labels=[0,1,2,3])
#targets['Common_Class'] = pd.cut(targets["SHAVE_Common"], bins=[-1,0.254,25.3,50.7,200.],include_lowest=True,right=False,labels=[0,1,2,3])
​
#split the data into the train/validate and test sets
data_train, data_test, target_train, target_test = train_test_split(data, targets['maxEst'], test_size = 0.2, random_state=42)
​
#### EDIT ABOVE HERE FOR YOUR DATA
#### EDIT BELOW HERE FOR MODEL SET UP
​
#choose which model architecture to use
gbr = GradientBoostingRegressor(max_features='sqrt') 
​
#hyperparamter grid for searching
param_grid = [{'n_estimators': [100, 200, 300, 400, 500], 'max_depth': [5, 10, 15]}]
​
#set up the train/validate cross-validation
grid_search = GridSearchCV(gbr, param_grid, cv=10, scoring='neg_mean_absolute_error', n_jobs=10) #make the CV object
results = grid_search.fit(data_train, target_train) #do the CV
​
#save the results for plotting
results_dataframe = pd.DataFrame.from_dict(grid_search.cv_results_)
results_dataframe.to_csv("Results_Max.csv") 
​
#get the best hyperparameters
bestparams = grid_search.best_estimator_
#run the best model on the test data
predictions = bestparams.predict(data_test) 
​
#calculate the MAE for the test set
calcmae = mean_absolute_error(target_test, predictions) 
​
#print the test set MAE
print(grid_search.best_params_)
print('Calculated MAE: target values/predicted values: ' + str(calcmae))
​
#save the test set obs and test set predictions for plotting
pd.DataFrame(predictions).to_csv("Predictions_Max.csv")
pd.DataFrame(target_test).to_csv("Targets_Max.csv")







# import numpy as np
# import pandas as pd
# import matplotlib
# matplotlib.use('agg')
# import matplotlib.pyplot as plt
# import seaborn as sns
# from sklearn.ensemble import (GradientBoostingClassifier,
#                               GradientBoostingRegressor)
# from sklearn.metrics import log_loss, mean_absolute_error
# from sklearn.model_selection import StratifiedKFold, StratifiedShuffleSplit, train_test_split, GridSearchCV
# ​
# ##GET DATA AND CLEAN IT UP
# #read in our data
# data = pd.read_csv('TheClusters_SHAVE_Soundings.csv', sep=",")
# data = data.replace([-99900., -99903], 0.) # replace any WDSS-II missing values with 0
# targets = data[["SHAVE_Common", "SHAVE_Max"]] #these are the data we are going to try to predict
# #drop the RowName column, it's not helpful nor really data and also drop our targets out of the data
# data = data.drop(columns=["RowName", "SHAVE_Common", "SHAVE_Max"])
# #need to make our targets a class
# #this is needed for both the classifier and for splitting the data so we have representative samples (See below)
# targets['Max_Class'] = pd.cut(targets["SHAVE_Max"], bins=[-1,0.254,25.3,50.7,200.],include_lowest=True,right=False,labels=[0,1,2,3])
# targets['Common_Class'] = pd.cut(targets["SHAVE_Common"], bins=[-1,0.254,25.3,50.7,200.],include_lowest=True,right=False,labels=[0,1,2,3])
# ​
# #split the data into the train/validate and test sets
# data_train, data_test, target_train, target_test = train_test_split(x_train, y_train, test_size = 0.2, random_state=42)
# ​
# #### EDIT ABOVE HERE FOR YOUR DATA
# #### EDIT BELOW HERE FOR MODEL SET UP
# ​
# #choose which model architecture to use
# gbr = GradientBoostingRegressor(max_features='sqrt') 
# ​
# #hyperparamter grid for searching
# param_grid = [{'n_estimators': [100, 200, 300, 400, 500], 'max_depth': [5, 10, 15]}]
# ​
# #set up the train/validate cross-validation
# grid_search = GridSearchCV(gbr, param_grid, cv=10, scoring='neg_mean_absolute_error', n_jobs=10) #make the CV object
# results = grid_search.fit(data_train, target_train) #do the CV
# ​
# #save the results for plotting
# results_dataframe = pd.DataFrame.from_dict(grid_search.cv_results_)
# results_dataframe.to_csv("Results_Max.csv") 
# ​
# #get the best hyperparameters
# bestparams = grid_search.best_estimator_
# #run the best model on the test data
# predictions = bestparams.predict(data_test) 
# ​
# #calculate the MAE for the test set
# calcmae = mean_absolute_error(target_test, predictions) 
# ​
# #print the test set MAE
# print(grid_search.best_params_)
# print('Calculated MAE: target values/predicted values: ' + str(calcmae))
# ​
# #save the test set obs and test set predictions for plotting
# pd.DataFrame(predictions).to_csv("Predictions_Max.csv")
# pd.DataFrame(target_test).to_csv("Targets_Max.csv")

# # metrics
# def accuracy(y_true, y_pred):
#     accuracy = np.sum(y_true == y_pred) / len(y_true)
#     return accuracy

# parameters = {'kernel':('linear', 'rbf')}


# regr = RandomForestRegressor(max_depth=3, random_state = 0)
# regr.fit(x_train, y_train)
# y_pred = regr.predict(x_test)
# print(regr.score(x_test,y_test))



def distance(x, y):
    # returns euclidean distance
    dist = np.sqrt(np.sum(x-y)**2)
    return dist

class knn:

    def __init__(self, k):
        # k: number of neighbors
        self.k = k
    
    def fit(self, x_train, y_train):
        self.x_train = x_train
        self.y_train = y_train
        
    def predict(self, x_train):
        y_pred = []
        for x in x_train:
            y_pred.append(self.prediction(x))
        return y_pred
    
    def prediction(self, x):
        # compute the distances 
        dists = []
        for x_ in self.x_train:
            dists.append(distance(x, self.x_train))
        
        # sort for the nearest k neighbors
        neighbors = np.argsort(dists)[:self.k]
        # get the types of these neighbors
        types = []
        for idx, n in enumerate(neighbors):
            types.append(self.y_train[idx])
        
        # pick type based on these neighbors
        # we will use the mode of types
        return int(max(set(types), key=types.count))

def distance(x, y):
    # returns euclidean distance
    dist = np.sqrt(np.sum(x-y)**2)
    return dist

k = 5
fitKNN = knn(k=3)
fitKNN.fit(x_train, y_train)
y_pred = fitKNN.predict(x_test)
print("accuracy of knn:", accuracy(y_pred,y_test))



#---------------------
# Logistic Regression

# Regression

class logisticRegression:
    
    def __init__(self,lr,n_iter):
        self.lr = lr
        self.n_iter = n_iter
        self.w = None
        self.b = None
        
    def fit(self, x_train, y_train):
        n_x, n_feat = x_train.shape
        self.w = np.zeros(n_feat)
        self.b = 0
        
        # now we use gradient descent
        for i in range(self.n_iter):
            y_pred = self.calculate(x_train)
            
            # get the gradients for w and b
            dw = (1/n_x) * np.dot(x_train.T, (y_pred - y_train))
            db = (1/n_x) * np.sum(y_pred - y_train)
            
            # update weights and bias
            self.w = self.w - self.lr * dw
            self.b = self.b - self.lr * db
            
    def calculate(self, x_train):
        predicted = self.sigmoid(np.dot(x_train, self.w) + self.b)
        return predicted
    
    def sigmoid(self, x):
        # simple sigmoidal function for logistic regression
        return 1 / (np.exp(-x) + 1)
    
    def predict(self, x_train):
        y_pred = self.sigmoid(np.dot(x_train, self.w) + self.b)
        y_class = [1 if i > 0.5 else 0 for i in y_pred]
        return np.array(y_class)

r = logisticRegression(0.01,100)
r.fit(x_train, y_train)
y_pred = r.predict(x_test)
print('Logistic regression:',accuracy(y_test,y_pred))

# Naive Bayes

class NaiveBayes:

    def fit(self, x_train, y):
        n_samples, n_features = x_train.shape
        # get the unique classes (0,1 in case of Hail dataset)
        self.classes = np.unique(y)
        n_classes = len(self.classes)

        # calculate mean, var, and prior for each class
        # cast each as a float
        # first, initialize
        self.mean = np.zeros((n_classes, n_features), dtype=np.float64)
        self.var = np.zeros((n_classes, n_features), dtype=np.float64)
        self.priors =  np.zeros(n_classes, dtype=np.float64)

        for idx, c in enumerate(self.classes):
            x_c = x_train[y==c]
            self.mean[idx, :] = x_c.mean(axis=0)
            self.var[idx, :] = x_c.var(axis=0)
            self.priors[idx] = x_c.shape[0] / float(n_samples)

    def predict(self, x_train):
            # cycle and find the most likely 
        y_pred = [self.find_most_likely(x) for x in x_train]
        return np.array(y_pred)

    def find_most_likely(self, x):
        probabilities = []

        # calculate probability for each class
        for idx, c in enumerate(self.classes):
            prior = np.log(self.priors[idx])
            probability = np.sum(np.log(self.getP(idx, x)))
            probability = prior + probability
            probabilities.append(probability)
            
        # return the class with the highest probability
        return self.classes[np.argmax(probabilities)]
            

    def getP(self, class_idx, x):
        #calculate probability P
        mean = self.mean[class_idx]
        var = self.var[class_idx]
        num = np.exp(- (x-mean)**2 / (2 * var))
        den = np.sqrt(2 * np.pi * var)
        return num / den

n = NaiveBayes()
n.fit(x_train, y_train)
y_pred = n.predict(x_test)

print("Naive Bayes accuracy:",accuracy(y_test,y_pred))
print("Naive Bayes y_pred",y_pred)

# Support Vector Machine

class support:

    def __init__(self, lr=0.001, lam=0.01, n=1000):
        # learning rate
        self.lr = lr
        # lambda
        self.lam = lam
        # number of iterations
        self.n = n
        # initialize weights and bias
        self.w = None
        self.b = None


    def fit(self, x_train, y):
        n_samples, n_features = x_train.shape
        # set to -1 if <= 0, +1 otherwise
        y_changed = np.where(y <= 0, -1, 1)
        
        self.w = np.zeros(n_features)
        self.b = 0

        # 'training' loop
        for i in range(self.n):
            for idx, x_i in enumerate(x_train):
                positive = y_changed[idx] * (np.dot(x_i, self.w) - self.b) >= 1
                if positive:
                    self.w -= self.lr * (2 * self.lam * self.w)
                else:
                    self.w -= self.lr * (2 * self.lam * self.w - np.dot(x_i, y_changed[idx]))
                    self.b -= self.lr * y_changed[idx]


    def predict(self, x_train):
        # predict using sign of x*w - b
        approx = np.dot(x_train, self.w) - self.b
        return np.sign(approx)

svm = support()
svm.fit(x_train, y_train)
y_pred = svm.predict(x_test)
y_pred = y_pred.clip(min=0)
print("Support Vector Machine accuracy:",accuracy(y_test,y_pred))

#--------------
# neural network

class neuralnet():
        # a 2 layer neural network

        # initialize layers, learning rate, and iterations by input
        # initialize parameters, loss, input set and output set
        def __init__(self,layers,lr,n):
                self.lr = lr
                self.n = n
                self.layers = layers
                self.sample_size = None
                self.loss = []
                self.params = {}
                self.x_ = None
                self.y_ = None
        
        # initialize weights
        def initialize_w(self):
                np.random.seed(141)
                self.params["w1"] = np.random.randn(self.layers[0], self.layers[1]) 
                self.params['b1']  =np.random.randn(self.layers[1],)
                self.params['w2'] = np.random.randn(self.layers[1],self.layers[2]) 
                self.params['b2'] = np.random.randn(self.layers[2],)

        # now we add our functions
        def relu(self, z):
                return np.maximum(0,z)
        
        def sigmoid(self,z):
                return 1.0 / (1.0 + np.exp(-z))

        def calculate_loss(self, y, ymean):
                num_samples = len(y)
                return -1/num_samples * (np.sum(np.multiply(np.log(ymean), y) + np.multiply((1 - y), np.log(1 - ymean))))

        def forward(self):
                # propagate inputs forward
                # use z = x*w + b
                z1 = self.x_.dot(self.params['w1']) + self.params['b1']
                act1 = self.relu(z1)
                # use output from first 
                z2 = act1.dot(self.params['w2']) + self.params['b2']
                ymean = self.sigmoid(z2)
                loss = self.calculate_loss(self.y_, ymean)

                # update parameters
                self.params['z1'] = z1
                self.params['z2'] = z2
                self.params['act1'] = act1

                return ymean, loss

        def backward(self, ymean):
                # get the derivatives and update the parameters

                # use derivative relu
                # below 0, slope is 0. above, it is 1
                def drelu(ex):
                        ex[ex<=0] = 0
                        ex[ex>0] = 1
                        return ex
                
                dl_ymean = -(np.divide(self.y_, ymean) - np.divide((1 - self.y_),(1-ymean)))
                dl_sig = ymean * (1-ymean)
                dl_z2 = dl_ymean * dl_sig

                dl_A1 = dl_z2.dot(self.params['w2'].T)
                dl_w2 = self.params['act1'].T.dot(dl_z2)
                dl_b2 = np.sum(dl_z2, axis=0)

                dl_z1 = dl_A1 * drelu(self.params['z1'])
                dl_w1 = self.x_.T.dot(dl_z1)
                dl_b1 = np.sum(dl_z1, axis=0)

                # now we update the weights and biases
                self.params['w1'] = self.params['w1'] - self.lr * dl_w1
                self.params['w2'] = self.params['w2'] - self.lr * dl_w2
                self.params['b1'] = self.params['b1'] - self.lr * dl_b1
                self.params['b2'] = self.params['b2'] - self.lr* dl_b2
        def fit(self, x_, y_):
                # train the neural network
                self.x_ = x_
                self.y_ = y_ 
                self.initialize_w()

                for i in range(self.n):
                        ymean, loss = self.forward()
                        self.backward(ymean)
                        self.loss.append(loss)
                
        def predict(self, x_):
                # predict y_pred based on x_test
                z1 = x_.dot(self.params['w1'] + self.params['b1'])
                act1 = self.relu(z1)
                z2 = act1.dot(self.params['w2']) + self.params['b2']
                y_pred = self.sigmoid(z2)
                return np.round(y_pred)

        def accuracy(self, y_, ymean):
                return int(sum(y_ == ymean) / len(y_) * 100)

y_train = y_train.reshape(x_train.shape[0],1)
y_test = y_test.reshape(x_test.shape[0],1)

nn = neuralnet(layers=[9,20,1],lr=0.001, n=1000)
nn.fit(x_train, y_train)
y_pred = nn.predict(x_test)
print(accuracy(y_pred,y_test))
