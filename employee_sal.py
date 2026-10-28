
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor

# 1. Load the dataset
dataset = pd.read_csv(
    r"C:\Users\ybali\Downloads\emp_sal.csv"
)

# 2. Select features and target
x = dataset.iloc[:, 1:2].values
y = dataset.iloc[:, 2].values

# Position level to predict
new_level = [[6.5]]

# 3. Linear Regression
lin_reg = LinearRegression()
lin_reg.fit(x, y)

linear_prediction = lin_reg.predict(new_level)

# 4. Polynomial Regression
poly_reg = PolynomialFeatures(degree=5)
x_poly = poly_reg.fit_transform(x)

poly_model = LinearRegression()
poly_model.fit(x_poly, y)

polynomial_prediction = poly_model.predict(
    poly_reg.transform(new_level)
)

# Plot a smooth polynomial curve
x_grid = np.arange(x.min(), x.max(), 0.01).reshape(-1, 1)

plt.scatter(x, y, color="red", label="Actual salaries")
plt.plot(
    x_grid,
    poly_model.predict(poly_reg.transform(x_grid)),
    color="blue",
    label="Polynomial Regression"
)
plt.title("Polynomial Regression")
plt.xlabel("Position level")
plt.ylabel("Salary")
plt.legend()
plt.show()

# 5. Support Vector Regression
# SVR generally benefits from feature scaling
x_scaler = StandardScaler()
y_scaler = StandardScaler()

x_scaled = x_scaler.fit_transform(x)
y_scaled = y_scaler.fit_transform(y.reshape(-1, 1)).ravel()

svr_reg = SVR(kernel="poly", degree=5)
svr_reg.fit(x_scaled, y_scaled)

svr_scaled_prediction = svr_reg.predict(
    x_scaler.transform(new_level)
)

svr_prediction = y_scaler.inverse_transform(
    svr_scaled_prediction.reshape(-1, 1)
).ravel()

# 6. K-Nearest Neighbors Regression
knn_reg = KNeighborsRegressor(n_neighbors=4)
knn_reg.fit(x, y)

knn_prediction = knn_reg.predict(new_level)

# 7. Decision Tree Regression
dt_reg = DecisionTreeRegressor(random_state=0)
dt_reg.fit(x, y)

dt_prediction = dt_reg.predict(new_level)

# 8. Random Forest Regression
rf_reg = RandomForestRegressor(
    n_estimators=100,
    random_state=0
)
rf_reg.fit(x, y)

rf_prediction = rf_reg.predict(new_level)

# 9. Compare all predictions
results = pd.DataFrame({
    "Algorithm": [
        "Linear Regression",
        "Polynomial Regression",
        "SVR",
        "KNN",
        "Decision Tree",
        "Random Forest"
    ],
    "Predicted Salary": [
        linear_prediction[0],
        polynomial_prediction[0],
        svr_prediction[0],
        knn_prediction[0],
        dt_prediction[0],
        rf_prediction[0]
    ]
})

print(results.to_string(index=False))
