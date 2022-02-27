## Import libraries
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression as LoR
from sklearn.ensemble import GradientBoostingClassifier,RandomForestClassifier
from sklearn.metrics import accuracy_score,confusion_matrix,f1_score,auc,classification_report 
import plotly.express as ex 
import pickle as pkl

## Contents
st.sidebar.header("Bees and Pest") # Create Header text


# Create Interractive widgets
Period = st.sidebar.selectbox('Period:',['JAN THRU MAR','APR THRU JUN','JUL THRU SEP','OCT THRU DEC']) # <-- Widget to select through Period categorical feature and save it in period
Percentage_of_colonies_impacted = st.sidebar.slider('Percentage of colonies impacted',0.00,100.00)# <-- Widget to slide through and save percentage impacted feature

codes = ['Alabama', 'Arizona', 'Arkansas', 'California', 'Colorado',
       'Connecticut', 'Florida', 'Georgia', 'Hawaii', 'Idaho', 'Illinois',
       'Indiana', 'Iowa', 'Kansas', 'Kentucky', 'Louisiana', 'Maine',
       'Maryland', 'Massachusetts', 'Michigan', 'Minnesota',
       'Mississippi', 'Missouri', 'Montana', 'Nebraska', 'New Jersey',
       'New Mexico', 'New York', 'North Carolina', 'North Dakota', 'Ohio',
       'Oklahoma', 'Oregon', 'Pennsylvania', 'South Carolina',
       'South Dakota', 'Tennessee', 'Texas', 'Utah', 'Vermont',
       'Virginia', 'Washington', 'West Virginia', 'Wisconsin', 'Wyoming']
State = st.sidebar.selectbox('States:',codes)#<--- Widget to select through and save State categorical column 
year = st.sidebar.selectbox('Year :',['2015','2016','2017','2018','2019'])#<--Widget to select through and save year categorical column



model_gradient = pkl.load(open('Gradient_bees1','rb'))
model_logistics = pkl.load(open('Logistics_bees1','rb'))
model_random = pkl.load(open('Random_bees1','rb'))

data = {'Period' : Period, # codes to convert the stored widget data to dictionary
		'Percentage_of_colonies_impacted' : Percentage_of_colonies_impacted,
		'State' : State,
		'year' : year}

features = pd.DataFrame(data, index=[0]) # Converting the data to dataframe

dummied_features = pd.get_dummies(features) # Dummying the features data

lists = [] # Codes to create a list for the features to predict data so as to have the same column length as the train feature data.

X = ['Percentage_of_colonies_impacted','Period_APR THRU JUN', 'Period_JAN THRU MAR', 'Period_JUL THRU SEP',
       'Period_OCT THRU DEC', 'State_Alabama', 'State_Arizona',
       'State_Arkansas', 'State_California', 'State_Colorado',
       'State_Connecticut', 'State_Florida', 'State_Georgia', 'State_Hawaii',
       'State_Idaho', 'State_Illinois', 'State_Indiana', 'State_Iowa',
       'State_Kansas', 'State_Kentucky', 'State_Louisiana', 'State_Maine',
       'State_Maryland', 'State_Massachusetts', 'State_Michigan',
       'State_Minnesota', 'State_Mississippi', 'State_Missouri',
       'State_Montana', 'State_Nebraska', 'State_New Jersey',
       'State_New Mexico', 'State_New York', 'State_North Carolina',
       'State_North Dakota', 'State_Ohio', 'State_Oklahoma', 'State_Oregon',
       'State_Pennsylvania', 'State_South Carolina', 'State_South Dakota',
       'State_Tennessee', 'State_Texas', 'State_Utah', 'State_Vermont',
       'State_Virginia', 'State_Washington', 'State_West Virginia',
       'State_Wisconsin', 'State_Wyoming', 'year_2015', 'year_2016',
       'year_2017', 'year_2018', 'year_2019']
for i in (X) : 
    if i in (dummied_features.columns) :
        lists.append(dummied_features[i].iloc[0])
    else :
        lists.append(0)

pred_lists = np.array([lists]).reshape(1,-1) # Reshaping it to the proper 2 dimentional


st.write(f'Logistic Regression predicts : {model_logistics.predict_proba(pred_lists)[:,1]}') # Predict and output it with logreg


st.write(f'Gradient Boosting predicts : {model_gradient.predict_proba(pred_lists)[:,1]}') # Predict and output with Gra

st.write(f'Random Forest predicts : {model_random.predict_proba(pred_lists)[:,1]}') # Predict and output with Random forest.