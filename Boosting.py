#Import Libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import model_selection
#from sklearn import grid_search
#from sklearn import preprocessing
import time
from sklearn.ensemble import AdaBoostClassifier

#Column names
features = ["X1","X2","X3","X4","X5","X6","X7","X8","Y"]
#features = ["X1","X2","X3","X4","X5","X6","X7","X8","X9","X10","X11","X12","X13","X14","X15","X16","Y"]
#features=["age","job","marital","education","default","housing","loan","contact","month","day_of_week","duration","campaign","pdays","previous","poutcome","emp.var.rate","cons.price.idx","cons.conf.idx","euribor3m","nr.employed","Y"]
#features=["fixed acidity","volatile acidity","citric acid","residual sugar","chlorides","free sulfur dioxide","total sulfur dioxide","density","pH","sulphates","alcohol","Y"]
#Read the file
df=pd.read_csv("diabetes.csv",header=None,names=features)
#df=pd.read_csv("bank.csv",header=None,names=features)
#df=pd.read_csv("bank_full.csv",header=None,names=features)
#df=pd.read_csv("winequality-red.csv",header=None,names=features)

#df=df.drop(df[(df.Y==3)].index)
#df=df.drop(df[(df.Y==8)].index)

#Label Encoding required for Bank Dataset only
#for i in range(len(features)):
#    if (type(df[features[i]][0])==str):
#        #print(i)        
#        le = preprocessing.LabelEncoder()
#        le.fit(df[features[i]])
#        df[features[i]]=le.transform(df[features[i]])

no_of_features=df.shape[1]-1
no_of_rows=df.shape[0]

X_df = df[features[:-1]]
Y_df = df['Y']

X_train, X_test, y_train, y_test = model_selection.train_test_split(X_df,Y_df,\
                        test_size=0.3,random_state=0)

"""
BOOSTING
"""

start_time=time.clock()

#clf = AdaBoostClassifier(n_estimators=100)
clf = AdaBoostClassifier()

clf.fit(X=X_train, y=y_train)

accuracy_train=clf.score(X_train,y_train)
accuracy_test=clf.score(X_test,y_test)

print("Training accuracy is: ",accuracy_train)
print("Test accuracy is: ",accuracy_test)

end_time=time.clock()

print("Total Time taken: ",end_time-start_time)

"""PLOTS"""

x_axis_vals=[]
accuracy_train=[]
accuracy_test=[]

for i in range(9,-1,-1):
#for i in range(1,50):
#for i in np.linspace(0.01,1,100):
#for i in ["SAMME","SAMME.R"]:

    print(i)
    #i=0
    X_train_sub, X_unused, y_train_sub, y_unused = model_selection.train_test_split(\
                    X_train,y_train,test_size=i*0.1,random_state=0)    
    
    #clf=AdaBoostClassifier(n_estimators=i,learning_rate=0.45,algorithm="SAMME") 
    #clf=AdaBoostClassifier(n_estimators=36,learning_rate=i,algorithm="SAMME")
    #clf=AdaBoostClassifier(algorithm=i,n_estimators=36,learning_rate=0.45)
    clf=AdaBoostClassifier(learning_rate=0.45,n_estimators=36)
          
    clf=clf.fit(X=X_train_sub, y=y_train_sub) 
    #clf=clf.fit(X=X_train, y=y_train)
    
    x_axis_vals.append(100-10*i)
    accuracy_train.append(clf.score(X_train_sub,y_train_sub))
    
    #x_axis_vals.append(i)    
    #accuracy_train.append(clf.score(X_train,y_train))
    
    accuracy_test.append(clf.score(X_test,y_test))


"""For Continuos type curve"""
plt.plot(x_axis_vals,accuracy_test)
plt.plot(x_axis_vals,accuracy_train)

"""For Discrete Type"""
#plt.plot(x_axis_vals,accuracy_test,linestyle='none', marker='^')
#plt.plot(x_axis_vals,accuracy_train,linestyle='none', marker='^')

plt.title("Learning Curve")
plt.xlabel("Percentage of Training Data Used")

#plt.xlabel("# of Estimators")      `     
#plt.xlabel("Learning rate")`
#plt.xlabel("Algorithm Type")
           
plt.ylabel("Accuracy")
plt.legend(['Test Accuracy','Training Accuracy'])



"""FINDING BEST PARAMETERS"""
start_time=time.clock()
parameters = {\
              #'n_estimators':list(range(1,40)),\
              #'n_estimators':[36],\
              'learning_rate':np.linspace(0.4,1,41),\
              #'algorithm':["SAMME","SAMME.R"]\
              'algorithm':["SAMME"]\
              }

algo=AdaBoostClassifier()

clf = model_selection.GridSearchCV(algo,parameters,cv=10)
clf.fit(X=X_train, y=y_train)
print (clf.best_score_, clf.best_params_)
clf = clf.best_estimator_

accuracy_train=clf.score(X_train,y_train)
accuracy_test=clf.score(X_test,y_test)
print("Training accuracy is: ",accuracy_train)
print("Test accuracy is: ",accuracy_test)

end_time=time.clock()
print("Total Time taken: ",end_time-start_time)