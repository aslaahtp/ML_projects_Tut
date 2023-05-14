# -*- coding: utf-8 -*-
"""
Created on Sun May 14 12:43:48 2023

@author: HP
"""

import numpy as np
import pickle
import streamlit as st

loaded_model=pickle.load(open('D:/ML projects/Diabetes prediction/trained_model.sav','rb'))
def diabetes_prediction(input_data):
    
    input_data_np=np.asarray(input_data)
    reshaped_array=input_data_np.reshape(1,-1) #to expect only one data point

    prediction=loaded_model.predict(reshaped_array)
    if prediction[0]==0:
      return "Non diabetic";
    else:
      return "Diabetic";
      
def main():
    st.title('Diabetes prediction WebApp')
    Pregnancies=st.text_input('Number of Pregnencies')
    Glucose=st.text_input('Current Glucose value')
    BloodPressure=st.text_input('Current BP')
    SkinThickness=st.text_input('Enter SkinThickness')
    Insulin=st.text_input('Enter Insulin value in body')
    DiabetesPedigreeFunction=st.text_input('Enter DP functon value')
    BMI=st.text_input('Enter BMI')
    Age=st.text_input('Enter Your Age')
    
	
    diagnosis=''
    
    if st.button('Test Result'):
        diagnosis=diabetes_prediction([Pregnancies,Glucose,BloodPressure,SkinThickness,Insulin,BMI,DiabetesPedigreeFunction,Age])
        
    st.success(diagnosis)
    
if __name__ == '__main__':
    main()
        