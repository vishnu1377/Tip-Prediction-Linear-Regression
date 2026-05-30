import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt 
import warnings
warnings.filterwarnings('ignore')
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression 
from sklearn.metrics import r2_score 
import pickle
import streamlit as st

st.title('Tip Prediction using Linear Regression')

tips = sns.load_dataset('tips')

st.subheader('Dataset Overview')
st.write(tips.head())

tips.info()
tips.head()
tips.describe()
tips.isnull().sum()
st.write(tips)

sns.scatterplot(x = 'total_bill', y = 'tip', data = tips)
sns.regplot(x = 'total_bill', y = 'tip', data = tips)

tips = pd.get_dummies(tips, drop_first = True)
sns.heatmap(tips.corr(), annot = True, cmap = 'coolwarm')
tips.corr()

st.pyplot(plt.gcf())

X = tips[['total_bill', 'size']]
y = tips['tip']

x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = LinearRegression()

model.fit(x_train, y_train)

y_pred = model.predict(x_test)

r2 = r2_score(y_test, y_pred)
st.write(f"R² Score: {r2:.4f}")

comparison = pd.DataFrame({
    'Actual Tip': y_test,
    'Predicted Tip': y_pred
})

st.write(comparison.head(10))

bill = st.number_input('Enter the total bill amount:')
size = st.number_input('Enter the size of the party:', min_value=1, step=1)
if st.button('Predict Tip'):
    new_data = pd.DataFrame({'total_bill': [bill], 'size' : [size]})
    prediction = model.predict(new_data)
    st.write(f'Predicted Tip: {prediction[0]:.2f}')