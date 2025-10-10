import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import ShuffleSplit, validation_curve, train_test_split
import warnings

warnings.filterwarnings("ignore", category=UserWarning, module="matplotlib")

# -----------------------------
# Helper Functions
# -----------------------------
def model_complexity(X, y):
    cv = ShuffleSplit(n_splits=10, test_size=0.2, random_state=0)
    max_depth = np.arange(1, 11)
    
    train_scores, test_scores = validation_curve(
        DecisionTreeRegressor(), X, y,
        param_name="max_depth", param_range=max_depth,
        cv=cv, scoring="r2"
    )
    
    train_mean = np.mean(train_scores, axis=1)
    test_mean = np.mean(test_scores, axis=1)
    
    fig, ax = plt.subplots(figsize=(6,4))
    ax.plot(max_depth, train_mean, 'o-', color='r', label='Training Score')
    ax.plot(max_depth, test_mean, 'o-', color='g', label='Validation Score')
    ax.fill_between(max_depth, train_mean - np.std(train_scores, axis=1),
                    train_mean + np.std(train_scores, axis=1), color='r', alpha=0.15)
    ax.fill_between(max_depth, test_mean - np.std(test_scores, axis=1),
                    test_mean + np.std(test_scores, axis=1), color='g', alpha=0.15)
    ax.set_xlabel("Maximum Depth")
    ax.set_ylabel("Score (R²)")
    ax.set_title("Model Complexity - Decision Tree Regressor")
    ax.legend()
    st.pyplot(fig)

def train_model(X_train, y_train, depth):
    reg = DecisionTreeRegressor(max_depth=depth)
    reg.fit(X_train, y_train)
    return reg

def predict_trials(X, y, fitter, data):
    prices = []
    for k in range(10):
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=k)
        reg = fitter(X_train, y_train)
        pred = reg.predict([data])[0]
        prices.append(pred)
    return prices


# -----------------------------
# Streamlit App UI
# -----------------------------
st.set_page_config(page_title="🏡 Boston Housing Predictor", layout="wide")

st.title("🏡 Boston Housing Price Prediction App")
st.markdown("This app trains a Decision Tree model to predict home prices based on features.")

# Upload or load dataset
uploaded_file = st.file_uploader("Upload your CSV dataset", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file)
else:
    st.info("No dataset uploaded. Using sample Boston Housing data.")
    from sklearn.datasets import load_boston
    boston = load_boston()
    df = pd.DataFrame(boston.data, columns=boston.feature_names)
    df["PRICE"] = boston.target

st.dataframe(df.head())

# Feature and target selection
features = st.multiselect("Select feature columns", df.columns[:-1], default=df.columns[:3])
target = st.selectbox("Select target column", df.columns, index=len(df.columns)-1)

if st.button("Show Model Complexity Plot"):
    X = df[features]
    y = df[target]
    model_complexity(X, y)

st.markdown("---")
st.header("🔮 Predict a House Price")

col1, col2, col3 = st.columns(3)
inputs = []
for i, feat in enumerate(features):
    with [col1, col2, col3][i % 3]:
        val = st.number_input(f"{feat}", value=float(df[feat].mean()))
        inputs.append(val)

depth = st.slider("Select Decision Tree Depth", 1, 10, 3)

if st.button("Predict Price"):
    X = df[features]
    y = df[target]
    
    def fitter(X_train, y_train):
        return train_model(X_train, y_train, depth)
    
    prices = predict_trials(X, y, fitter, inputs)
    st.success(f"Predicted Prices across 10 trials: ${np.mean(prices):,.2f}")
    st.write(f"Price range: ${min(prices):,.2f} - ${max(prices):,.2f}")
