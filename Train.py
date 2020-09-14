#data preprocessing

#importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix,accuracy_score
from sklearn.externals import joblib

def train_model():
    #importing dataset
    dataset =pd.read_csv(r"C:\Users\priya\Downloads\PCOD_Disease_Prediction-master (1)\PCOD_Disease_Prediction-master\PCODdataset.csv")
    X=dataset.drop(['disease_present'],axis=1)
    Y=dataset['disease_present']

    #encoding categorical data
    labelencoder_y=LabelEncoder()
    Y=labelencoder_y.fit_transform(Y)

    #spliting training and test set
    X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=0.33,random_state=0)


    #fitting the classifier to the training set
    classifier = RandomForestClassifier(criterion='entropy',random_state=0)
    classifier.fit(X_train, Y_train)

    #predicting the test set results
    Y_pred=classifier.predict(X_test)

    # making the confusion matrix and calculating accuraccy
    cm=confusion_matrix(Y_test,Y_pred)
    accuracy = accuracy_score(Y_test, Y_pred)

    #dumping the classifier into model
    joblib.dump(classifier, 'pcod1final.model')
    print('Model Training Finished.\n\tAccuracy obtained: {}'.format(accuracy))
