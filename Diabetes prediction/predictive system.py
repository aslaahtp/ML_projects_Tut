# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
import numpy as np
import pickle

loaded_model=pickle.load(open('D:/ML projects/Diabetes prediction/trained_model.sav','rb'))

input_data=(7,196,92,0,0,37.6,0.191,30) #diabetic

input_data_np=np.asarray(input_data)
reshaped_array=input_data_np.reshape(1,-1) #to expect only one data point
#standerdise input
#std_data=scaler.transform(reshaped_array)
prediction=loaded_model.predict(reshaped_array)
if prediction[0]==0:
  print("Non diabetic")
else:
  print("Diabetic")