
# Employee Salary Prediction

A beginner-friendly machine-learning web app built with Python, scikit-learn, pandas, Matplotlib, and Streamlit. Upload a CSV, choose the feature and salary columns, select a regression algorithm, and predict a salary for a chosen position level.

## Features

- CSV upload and data preview
- Select feature and target columns
- Six regression options:
  - Linear Regression
  - Polynomial Regression
  - Support Vector Regression (polynomial kernel)
  - K-Nearest Neighbors
  - Decision Tree
  - Random Forest
- Visualize actual data and the fitted model
- Inspect fitted predictions and residuals
- Download model outputs as CSV

## Requirements

- Python 3.10 or newer recommended
- pip

## Run locally

1. Download or clone this repository.
2. Open a terminal in the project folder.
3. Create and activate a virtual environment (recommended).

### Windows

```powershell
python -m venv .venv
.venv\Scripts\activate
python -m pip install -r requirements.txt
streamlit run app.py
```

4. Open the local URL printed in the terminal.
5. Upload your `emp_sal.csv` dataset using the sidebar.

## Dataset format

Your CSV should contain at least two numeric columns: a position-level/feature column and a salary/target column. You can select their names in the app. Non-numeric and missing values in those two columns are dropped.

Example:

```csv
PositionLevel,Salary
1,45000
2,50000
3,60000
4,75000
```

Use your own dataset; the example values above are illustrative only.

## Project structure

```text
employee_salary_prediction/
├── app.py
├── requirements.txt
├── README.md
└── .gitignore
```

## Upload to GitHub

1. Create a new repository on GitHub named `employee-salary-prediction`.
2. Do not initialize it with another README if you already have this one locally.
3. Open a terminal in this project folder and run:

```bash
git init
git add .
git commit -m "Build employee salary prediction app"
git branch -M main
git remote add origin https://github.com/YOUR-USERNAME/employee-salary-prediction.git
git push -u origin main
```

Replace `YOUR-USERNAME` with your GitHub username.

## Deploy (optional)

You can deploy the repository using Streamlit Community Cloud. Connect your GitHub account, select this repository and `app.py`, and deploy. The app asks users to upload the CSV, so you do not need to commit your local dataset.

## Limitations

- This is an educational demonstration, not a validated salary-setting tool.
- A small dataset can lead to unstable predictions and overfitting.
- Use held-out data or cross-validation and appropriate regression metrics before comparing models for real applications.
