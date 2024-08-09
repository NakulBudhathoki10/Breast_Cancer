
import streamlit as st
import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split

# Load the data
@st.cache_data
def load_data():
    data = load_breast_cancer()
    df = pd.DataFrame(data.data, columns=data.feature_names)
    df['target'] = data.target
    return df

# Train the model
def train_model(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    clf = MLPClassifier(max_iter=1000)
    clf.fit(X_train, y_train)
    return clf

# Main Streamlit App
st.title('Breast Cancer Prediction App')

# Load the data
df = load_data()
st.write("Dataset Overview:", df.head())

if st.button('Train Model'):
    X = df.drop('target', axis=1)
    y = df['target']
    clf = train_model(X, y)
    st.success('Model trained successfully!')

    # User input for prediction
    st.subheader('Enter the feature values:')
    input_data = []
    for feature in df.columns[:-1]:
        input_val = st.number_input(f'{feature}', value=float(df[feature].mean()))
        input_data.append(input_val)
    
    # Predict based on user input
    if st.button('Predict'):
        input_df = pd.DataFrame([input_data], columns=df.columns[:-1])
        prediction = clf.predict(input_df)
        prediction_label = 'Malignant' if prediction[0] == 0 else 'Benign'
        st.write(f'Prediction: {prediction_label}')
