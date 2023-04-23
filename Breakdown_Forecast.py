# -*- coding: utf-8 -*-
"""
Created on Sun Apr 16 19:40:21 2023

@author: purva
"""

import streamlit as st
import pandas as pd
import numpy as np
import pickle

# Load the trained model
with open('finalized_model.sav', 'rb') as file:
    model = pickle.load(file)

# Load the input data
input_file = st.file_uploader("Upload input file", type="csv")
if input_file is not None:
    input_data = pd.read_csv(input_file)

    # Make predictions using the input data
    predictions = model.predict(input_data)

    # Display the predictions
    st.write(predictions)
