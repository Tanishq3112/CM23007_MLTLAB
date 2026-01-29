import numpy as np
import matplotlib.pyplot as plt

# -------------------------------
# 1. Load Data from File
# -------------------------------

file_name = input("Enter file name (with .csv or .txt): ")

data = np.loadtxt(file_name, delimiter=",")

X = data[:, 0].reshape(-1, 1)   # Watch Hours
Y = data[:, 1].reshape(-1, 1)   # Data Consumption

# -------------------------------
# 2. Gradient Descent Function
# -------------------------------

def train_linear_regression(X, Y, lr=0.01, epochs=1000):
    m = 0
    b = 0
    n = len(X)

    for i in range(epochs):
        Y_pred = m * X + b

        dm = (-2/n) * np.sum(X * (Y - Y_pred))
        db = (-2/n) * np.sum(Y - Y_pred)

        m = m - lr * dm
        b = b - lr * db

    return m, b

# -------------------------------
# 3. Train Model
# -------------------------------

m, b = train_linear_regression(X, Y)

print("\nSlope (m):", m)
print("Intercept (b):", b)

# -------------------------------
# 4. Predictions
# -------------------------------

Y_predicted = m * X + b

# -------------------------------
# 5. Plot Graph
# -------------------------------

plt.scatter(X, Y, label="Actual Data")
plt.plot(X, Y_predicted, color="black", label="Best Fit Line")
plt.xlabel("Watch Hours")
plt.ylabel("Data Consumption (GB)")
plt.title("Watch Hours vs Data Consumption")
plt.legend()
plt.show()

# -------------------------------
# 6. Predict Unseen Input
# -------------------------------

new_hours = float(input("Enter watch hours to predict data usage: "))
prediction = m * new_hours + b
print("Predicted Data Consumption:", prediction, "GB")
