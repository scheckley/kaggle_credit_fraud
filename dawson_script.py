# script to run random forest "hands free" on Dawson

# initial setup
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.utils import shuffle
import pickle
pd.options.display.max_rows = 10
pd.options.display.float_format = '{:.1f}'.format

#for logistic regression and over sampling
cc = !nproc
#cc = !sysctl -n hw.ncpu
cc = int(cc[0])

####################################################################################
# timing helper functions

def tic():
    #Homemade version of matlab tic and toc functions
    import time
    global startTime_for_tictoc
    startTime_for_tictoc = time.time()

def toc():
    import time
    if 'startTime_for_tictoc' in globals():
        print("Elapsed time is " + str(time.time() - startTime_for_tictoc) + " seconds.")
    else:
        print("Toc: start time not set")

####################################################################################
# set up parallel cluster using ipyparallel engines.
# ipcluster start
import ipyparallel as ipp
from ipyparallel import require
c = ipp.Client(profile='default')
v = c[:]
#print("engines running:",v)
print("engines running:",len(v))

####################################################################################
data = pd.read_csv("data/creditcard.csv")
data.shape

normamount = np.array(data['Amount'])
normamount = StandardScaler().fit_transform(normamount.reshape(-1,1))
data['normamount'] = normamount
del data['Amount']
del data['Time']
data = shuffle(data)
data.head(10)
data.shape

####################################################################################
from imblearn.over_sampling import ADASYN#SMOTE
sm = ADASYN(n_jobs=cc)

X = data.loc[:, data.columns != 'Class']
y = data.loc[:, data.columns == 'Class']

from sklearn.decomposition import PCA
pca = PCA(n_components=2)
X_vis = pca.fit_transform(X)

X.columns = X.columns.astype(str)
y.columns = y.columns.astype(str)
####################################################################################
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size = 0.3, random_state = 0)

X_train_resampled = pickle.load(open( "X_train_resampled.pkl", "rb" ))
y_train_resampled = pickle.load(open( "y_train_resampled.pkl", "rb" ))

####################################################################################
# confusion matrix plotting function
import itertools

def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=0)
    plt.yticks(tick_marks, classes)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        #print("Normalized confusion matrix")
    else:
        1#print('Confusion matrix, without normalization')

    #print(cm)

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j],
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')

####################################################################################
# set up the random forest

from sklearn.ensemble import RandomForestClassifier
from sklearn.externals import joblib #for saving the trained model

# try a grid search to find best parameters
from sklearn.model_selection import GridSearchCV
tic()

# Grid search best parameters
rf = RandomForestClassifier(n_estimators=100, oob_score=True, n_jobs=cc)
grid_values = {'n_estimators': [100,500,1000,5000]}
rf = GridSearchCV(rf, param_grid=grid_values, n_jobs=cc)

rf.fit(X_train_resampled,y_train_resampled)

y_pred = rf.predict(X_test)
toc()

print(rf.best_params_)

#save the model
joblib.dump(rf, 'rf_model.pkl')




####################################################################################

####################################################################################

####################################################################################