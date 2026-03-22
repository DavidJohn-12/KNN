# Diabetes Prediction using K-Nearest Neighbors
# Import libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, classification_report
from sklearn.metrics import roc_curve
import matplotlib.pyplot as plt


#Load the datasets
df1 = pd.read_csv("diabetes_012_health_indicators_BRFSS2015.csv")
df2 = pd.read_csv("diabetes_012_health_indicators_BRFSS2015_balanced.csv")
df3 = pd.read_csv("diabetes_binary_health_indicators_BRFSS2015.csv")