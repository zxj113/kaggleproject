import csv
import numpy
from sklearn import preprocessing
from sklearn import tree
from sklearn import linear_model
from sklearn.cross_validation import KFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import GradientBoostingClassifier
import pandas as pd
from sklearn.externals import joblib
import time
from sklearn.naive_bayes import MultinomialNB

filename='/Users/jzhy/Downloads/train.csv'
data=pd.read_csv(filename)

X=numpy.zeros((len(data.x),4))
X[:,0]=data.x
X[:,1]=data.y
X[:,2]=data.accuracy
X[:,3]=data.time
Y=numpy.zeros((len(data.x),1))
Y=data.place_id

XX=preprocessing.scale(X)
YY=numpy.unique(Y)

kf=KFold(len(X),n_folds=len(Y)/10000+1)

clf=MultinomialNB()

i=0
for train, test in kf:
	clf.partial_fit(X[test,:],Y[test],YY.reshape((len(YY),-1)))
	i=i+1
	print i
joblib.dump(clf,'MultinomialNB.pkl')

exit()
