import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor

st.set_page_config(page_title="Employee Salary Predictor", page_icon="💼", layout="wide")

st.title("💼 Employee Salary Prediction")
st.write(
    "Explore regression algorithms and predict salary from an employee's position level. "
    "Upload your CSV dataset to get started."
)

with st.sidebar:
    st.header("1. Upload data")
    uploaded_file = st.file_uploader(  r"C:\Users\ybali\Downloads\emp_sal.csv", type=["csv"])
    st.caption("Expected: one numeric position-level column and one numeric salary column.")

if uploaded_file is None:
    st.info("Upload your `emp_sal.csv` file in the sidebar. Your file is processed in this app session.")
    st.code("Example columns:\nPositionLevel, Salary\n1, 45000\n2, 50000", language="text")
    st.stop()

try:
    df = pd.read_csv(uploaded_file)
except Exception as exc:
    st.error(f"Could not read the CSV file: {exc}")
    st.stop()

if df.empty or len(df.columns) < 2:
    st.error("Your CSV needs at least two columns: a position level and a salary.")
    st.stop()

st.subheader("2. Preview your dataset")
st.dataframe(df.head(10), use_container_width=True)
st.caption(f"{len(df)} rows · {len(df.columns)} columns")

with st.sidebar:
    st.header("3. Configure model")
    
    feature_default = (
    df.columns.get_loc("Level")
    if "Level" in df.columns
    else 0
)

target_default = (
    df.columns.get_loc("Salary")
    if "Salary" in df.columns
    else min(1, len(df.columns) - 1)
)

feature_col = st.selectbox(
    "Position-level / feature column",
    df.columns,
    index=feature_default
)

target_col = st.selectbox(
    "Salary / target column",
    df.columns,
    index=target_default
)



algorithm = st.selectbox(
        "Regression algorithm",
        [
            "Polynomial Regression (degree 5)",
            "Linear Regression",
            "Support Vector Regression (polynomial kernel)",
            "K-Nearest Neighbors",
            "Decision Tree",
            "Random Forest",
        ],
    )
degree = st.slider("Polynomial degree", min_value=2, max_value=5, value=5)
predict_value = st.number_input("Position level to predict", value=6.5, step=0.5)

if feature_col == target_col:
    st.error("Choose different columns for the feature and salary.")
    st.stop()

clean = df[[feature_col, target_col]].copy()
clean[feature_col] = pd.to_numeric(clean[feature_col], errors="coerce")
clean[target_col] = pd.to_numeric(clean[target_col], errors="coerce")
clean = clean.dropna()

if len(clean) < 2:
    st.error("At least two rows with numeric feature and salary values are required.")
    st.stop()

X = clean[[feature_col]].to_numpy()
y = clean[target_col].to_numpy()

try:
    if algorithm == "Polynomial Regression (degree 5)":
        model = make_pipeline(PolynomialFeatures(degree=degree), LinearRegression())
    elif algorithm == "Linear Regression":
        model = LinearRegression()
    elif algorithm == "Support Vector Regression (polynomial kernel)":
        model = make_pipeline(
            StandardScaler(),
            SVR(kernel="poly", degree=degree)
        )
        # Scale target values too, then reverse scaling for displayed predictions.
        y_scaler = StandardScaler()
        y_fit = y_scaler.fit_transform(y.reshape(-1, 1)).ravel()
        model.fit(X, y_fit)
        prediction_scaled = model.predict(np.array([[predict_value]]))
        prediction = float(y_scaler.inverse_transform(prediction_scaled.reshape(-1, 1))[0, 0])
        fitted_scaled = model.predict(X)
        fitted_values = y_scaler.inverse_transform(fitted_scaled.reshape(-1, 1)).ravel()
    elif algorithm == "K-Nearest Neighbors":
        neighbors = min(4, len(clean))
        model = KNeighborsRegressor(n_neighbors=neighbors)
    elif algorithm == "Decision Tree":
        model = DecisionTreeRegressor(random_state=0)
    else:
        model = RandomForestRegressor(n_estimators=100, random_state=0)

    if algorithm != "Support Vector Regression (polynomial kernel)":
        model.fit(X, y)
        prediction = float(model.predict(np.array([[predict_value]]))[0])
        fitted_values = model.predict(X)

except Exception as exc:
    st.error(f"Model training failed: {exc}")
    st.stop()

st.subheader("4. Prediction")
left, right = st.columns(2)
with left:
    st.metric("Input position level", f"{predict_value:g}")
with right:
    st.metric("Predicted salary", f"{prediction:,.2f}")

st.caption("This is a learning demonstration. Predictions depend on the dataset and may not reflect real-world salaries.")

st.subheader("5. Actual values and fitted model")
fig, ax = plt.subplots(figsize=(9, 5))
ax.scatter(X[:, 0], y, color="red", label="Actual salary")

x_grid = np.linspace(X.min(), X.max(), 300).reshape(-1, 1)
if algorithm == "Support Vector Regression (polynomial kernel)":
    grid_scaled = model.predict(x_grid)
    y_scaler_for_grid = y_scaler
    y_grid = y_scaler_for_grid.inverse_transform(grid_scaled.reshape(-1, 1)).ravel()
else:
    y_grid = model.predict(x_grid)

ax.plot(x_grid[:, 0], y_grid, color="blue", label=algorithm)
ax.set_xlabel(feature_col)
ax.set_ylabel(target_col)
ax.set_title("Salary prediction model")
ax.legend()
ax.grid(alpha=0.2)
st.pyplot(fig)
plt.close(fig)

st.subheader("6. Model output")
output_df = clean.copy()
output_df["Model prediction"] = fitted_values
output_df["Residual (actual - predicted)"] = y - fitted_values
st.dataframe(output_df, use_container_width=True)

csv_bytes = output_df.to_csv(index=False).encode("utf-8")
st.download_button(
    "Download predictions as CSV",
    data=csv_bytes,
    file_name="salary_predictions.csv",
    mime="text/csv",
)

with st.expander("What do the algorithms do?"):
    st.markdown("""
    - **Linear Regression:** fits a straight line.
    - **Polynomial Regression:** adds powers of the feature to model curves.
    - **SVR:** fits a support-vector regression function; scaling is important.
    - **KNN:** predicts using nearby training examples.
    - **Decision Tree:** predicts using feature-based splits.
    - **Random Forest:** averages predictions from many decision trees.

    Compare models carefully. A visually close curve does not automatically mean a model will generalize well.
    """)