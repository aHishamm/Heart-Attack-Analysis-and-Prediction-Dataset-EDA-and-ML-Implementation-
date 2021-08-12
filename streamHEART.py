import pandas as pd 
import numpy as np 
from sklearn.model_selection import train_test_split 
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import classification_report
from sklearn.metrics import average_precision_score
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import streamlit as st 

HDFrame = pd.read_csv('heart.csv')

#shuffling the dataset 
HDFrame = HDFrame.sample(frac=1).reset_index(drop=True)
#choosing the input and output 
x = HDFrame.iloc[:,:13].values
y = HDFrame.iloc[:,13:].values 
#setting the train test split 
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.20)

#feeding the data to a train a naive bayes model 
naiveClassifier = GaussianNB() 
naiveClassifier.fit(x_train,y_train.ravel()) 
y_pred = naiveClassifier.predict(x_test)
average_precision = average_precision_score(y_test.ravel(),y_pred)

#streamlit title 
st.title("Heart Attack Streamlit App")
#streamlit subheader 
st.subheader("This streamlit application is to display the accuracy of the model created.")

#streamlit table 
st.table(HDFrame.head())


#classification report and average precision score 
st.subheader('Classification report: ')
st.text(classification_report(y_pred,y_test.ravel()))
#print(classification_report(y_pred,y_test.ravel()))
st.subheader('average precision score: ')
st.text(average_precision)