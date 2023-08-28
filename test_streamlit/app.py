##Librairies to import
import pandas as pd
import numpy as np
import re
import matplotlib.pyplot as plt
import seaborn as sns
##Import Streamlit
import streamlit as st


#########################INITIALIZE STREAMLIT#######################################################################
st.title("Data Visualization web application for NLP diseases analysis")
st.write("In this section, we will explore the Disease Text dataSet.")

##Import data
st.header("Explore the dataframe")
data = pd.read_csv("/home/annemocoeur/Final_project_Artefact/NLP_Analysis/Description_diseases_raw_1.csv", encoding_errors="ignore", delimiter=";")

st.dataframe(data=data)

##Look for value count
##Data variable response
st.header("Text data Preprocessed")
st.write("Data from all columns are merged within line in a New column.")
st.write("Text processing is done including, keeping alphnumeric worlds, removing stop words, lemmetizing with Tagged words and tokenizing")

##Data SUBSET & PREPROCESSING
data_cleaned = pd.read_csv("/home/annemocoeur/Final_project_Artefact/NLP_Analysis/Data_cleaned.csv", encoding_errors="ignore", index_col=0)
st.dataframe(data=data_cleaned)

##Look for value count
##Data variable response
st.header("Creating BOW and TD_IDF")
st.write("From cleaned and tokenized data, BOW and TD-IDF are created with Genim & Sklearn tools.")
st.write("TDF-IDF corpus contains 716 words.")
st.write("TDF table obtained.")

td_idf = pd.read_csv("/home/annemocoeur/Final_project_Artefact/NLP_Analysis/TD_IDF_data.csv")
st.dataframe(data=td_idf)


##Look for value count
##Data variable response
st.header("Analysis of similarities")
st.write("Cosine similarities (pairwise) and Jaccard distances have been computed from TD-IDF between diseases description text")

##PLOTTING JACCARD 
cos  = pd.read_csv("/home/annemocoeur/Final_project_Artefact/NLP_Analysis/Cosine_similarities.csv", delimiter=";", index_col=0)

###Plotting Heatmapt
fig, ax = plt.subplots()
sns.heatmap(cos, annot=True)
plt.title("Jacquard Heatmap Distances" )
st.pyplot(fig)
