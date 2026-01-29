import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

# -----------------------------
# Title
# -----------------------------
st.title("Hours Watched vs Data Consumed (Regression)")

st.write("Upload CSV or TXT file:")
st.write("Column 1 → Hours Watched")
st.write("Column 2 → Data Consumed (GB)")

# -----------------------------
# Upload File
# -----------------------------
uploaded_file = st.file_uploader("Upload File", type=["csv", "txt"])

if uploaded_file is not None:

    # -----------------------------
    # Read Data
    # -----------------------------
    df = pd.read_csv(uploaded_file, header=None)
    df = df.dropna()

    X = df.iloc[:, 0].values.reshape(-1, 1)
    Y = df.iloc[:, 1].values.reshape(-1, 1)

    st.success("File loaded successfully!")

    # -----------------------------
    # Train Regression Model
    # -----------------------------
    model = LinearRegression()
    model.fit(X, Y)

    # -----------------------------
    # Predictions
    # -----------------------------
    Y_pred = model.predict(X)

    # -----------------------------
    # Sort for clean line
    # -----------------------------
    idx = np.argsort(X.flatten())
    X_sorted = X[idx]
    Y_sorted = Y[idx]
    Y_pred_sorted = Y_pred[idx]

    # -----------------------------
    # Plot
    # -----------------------------
    fig, ax = plt.subplots()

    ax.scatter(X_sorted, Y_sorted, label="Uploaded Data")
    ax.plot(X_sorted, Y_pred_sorted, color="black", label="Best Fit Line")

    ax.set_xlabel("Watch Hours")
    ax.set_ylabel("Data Consumption (GB)")
    ax.set_title("Best Fit Line on Uploaded Dataset")
    ax.legend()
    ax.grid(True)

    st.pyplot(fig)

    # -----------------------------
    # Prediction Section
    # -----------------------------
    st.subheader("Predict Data Consumption")

    hours = st.number_input("Enter Hours Watched:", min_value=0.0)

    if st.button("Predict"):
        result = model.predict([[hours]])
        st.success(f"Predicted Data Consumption: {result[0][0]:.2f} GB")
