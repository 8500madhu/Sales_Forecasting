Project Title:
      Sales Forecasting Using Machine Learning

Objective:
      The objective is to predict future sales using historical sales data.

Historical Sales
      ↓
Data Cleaning
      ↓
Feature Engineering
      ↓
Train ML Model
      ↓
Evaluate Model
      ↓
Future Sales Prediction



Problem Type:
        Sales forecasting is generally a regression/time-series forecasting problem.

Technologies:
        Python
        Pandas
        NumPy
        Matplotlib
        Scikit-learn
        Flask
        HTML
        CSS
        Pickle

Algorithms:
        Linear Regression
        Random Forest 
         XGBoost


Project Structure:
        Sales_Forecasting/
          │
          ├── sales.csv
          ├── train.py
          ├── app.py
          ├── model.pkl
          │
          ├── templates/
          │   └── index.html
          │
          └── static/
              └── style.css


Implementation process:
        sales.csv
            ↓
      Date preprocessing
            ↓
      Exploratory Data Analysis
            ↓
      Feature Engineering
            ↓
      Train/Test Split
            ↓
      Linear Regression
            ↓
      Random Forest
            ↓
          XGBoost
            ↓
      MAE / RMSE / R²
            ↓
      Flask Web App

# 📈 Sales Forecasting Using Machine Learning

A machine-learning web application that predicts future sales for individual stores using historical sales data.

The project uses **Date, Store, and Sales** information to create time-based features and lag features. A **Random Forest Regression** model is trained to forecast future sales.

The trained model is deployed using **Flask** with an HTML/CSS frontend.

---

## 🚀 Features

* Historical sales data processing
* Multi-store sales forecasting
* Date-based feature engineering
* Lag feature generation
* Rolling average feature generation
* Random Forest Regression
* MAE, RMSE, and R² evaluation
* Custom forecasting horizon
* Flask web application
* Store selection through web interface
* Future sales prediction
* Forecasted sales table
* Total forecasted sales calculation

---

## 🧠 Machine Learning Workflow

Historical Sales Data
        ↓
Data Loading
        ↓
Data Cleaning
        ↓
Date Conversion
        ↓
Feature Engineering
        ↓
Lag Features
        ↓
Rolling Average
        ↓
Time-Based Train/Test Split
        ↓
Random Forest Regression
        ↓
Model Evaluation
        ↓
Model Serialization
        ↓
Flask Deployment
        ↓
Future Sales Forecast



## 📊 Dataset

The dataset contains three main columns:

| Column | Description                   |
| ------ | ----------------------------- |
| Date   | Date of the sales transaction |
| Store  | Store identifier              |
| Sales  | Sales amount                  |

Example:


Date,Store,Sales
2024-01-01,Store_1,25000
2024-01-02,Store_1,27000
2024-01-03,Store_1,23000
2024-01-01,Store_2,18000
2024-01-02,Store_2,21000


## 🛠️ Technologies Used

### Programming Language

* Python

### Data Processing

* Pandas
* NumPy

### Machine Learning

* Scikit-learn
* Random Forest Regression

### Web Development

* Flask
* HTML
* CSS

### Model Saving

* Pickle



## 📁 Project Structure

Sales_Forecasting/
│
├── sales.csv
├── train.py
├── app.py
├── model.pkl
│
├── templates/
│   └── index.html
│
├── static/
│   └── style.css
│
└── README.md




## 🔧 Feature Engineering

The Date column is converted into several numerical features.

### Date Features


Year
Month
Day
DayOfWeek
DayOfYear
WeekOfYear


### Lag Features

The project uses previous sales values:


Lag_1
Lag_7


Where:

* `Lag_1` = previous day's sales
* `Lag_7` = sales from seven days earlier

### Rolling Feature


Rolling_7

represents the seven-day rolling average of historical sales.

These features provide the model with information about recent sales behavior.


## 🤖 Machine Learning Model

The project uses:


RandomForestRegressor(
    n_estimators=200,
    random_state=42,
    n_jobs=-1
)


Random Forest Regression combines multiple decision trees to make numerical predictions.

The target variable is:


Sales



## 📚 Train/Test Split

Because this is a time-series forecasting problem, the project uses a chronological split rather than randomly shuffling the dataset.


Older historical data
        ↓
Training Data

Newer historical data
        ↓
Testing Data


This better represents how the model will be used in the real world.


## 📏 Model Evaluation

The following metrics are calculated:

### MAE

Mean Absolute Error measures the average absolute difference between actual and predicted sales.


MAE = Average |Actual - Predicted|

Lower MAE is better.

### RMSE

Root Mean Squared Error gives greater weight to larger prediction errors.

Lower RMSE is better.

### R² Score

R² indicates how well the model explains variation in the target variable.

A value closer to 1 generally indicates better fit.



## 🌐 Flask Web Application

The Flask application allows users to:

1. Select a store
2. Enter a custom forecast horizon
3. Generate future sales predictions
4. View predicted sales for each future date
5. View total forecasted sales

Example:

Store:
Store_1

Forecast Horizon:
30 days

    ↓

Forecast Results

Date          Predicted Sales
2026-08-12    25430.20
2026-08-13    26120.75
2026-08-14    24980.40





## ⚙️ Installation

Clone the repository:


git clone YOUR_GITHUB_REPOSITORY_URL


Move into the project directory:

cd Sales_Forecasting

Install dependencies:


py -m pip install pandas numpy scikit-learn flask



## ▶️ Run the Project

### Step 1 — Train the model


py train.py


This generates:


model.pkl


### Step 2 — Start Flask


py app.py

### Step 3 — Open the application

Open your browser and visit:


http://127.0.0.1:5000



## 🖥️ Application Flow


User
 ↓
Select Store
 ↓
Enter Forecast Horizon
 ↓
Flask
 ↓
Historical Store Data
 ↓
Feature Generation
 ↓
Random Forest Model
 ↓
Future Sales Prediction
 ↓
Display Results




## 🔮 Custom Forecast Horizon

The application supports a custom number of future days.

For example:


7  → 7-day forecast
15 → 15-day forecast
30 → 30-day forecast
60 → 60-day forecast
90 → 90-day forecast


The user can enter the desired number of days directly into the web application.


## 📈 Example Use Cases

This project can be used for:

* Retail sales forecasting
* Store-level planning
* Inventory planning
* Demand forecasting
* Business decision support
* Sales trend analysis
* Stock management


## ⚠️ Limitations

* Forecast accuracy depends heavily on the quality and quantity of historical data.
* Random Forest does not inherently model temporal patterns like specialized time-series models.
* Unexpected events can cause significant forecasting errors.
* The current implementation uses a recursive forecasting approach for multi-day predictions.
* External factors such as holidays, promotions, weather, and competitor activity are not included unless present in the dataset.


## 🚀 Future Improvements

Possible improvements include:

* XGBoost
* LightGBM
* Gradient Boosting
* Random Forest optimization
* Hyperparameter tuning
* Holiday features
* Promotion features
* Weather information
* Product-level forecasting
* Inventory integration
* Interactive sales charts
* Automated model retraining
* LSTM
* GRU
* Temporal Fusion Transformer
* Prophet or other dedicated forecasting approaches

---

## 🎯 Learning Outcomes

Through this project, you can learn:

* Time-series data preprocessing
* Date feature engineering
* Lag features
* Rolling averages
* Regression
* Train/test splitting for temporal data
* Model evaluation
* Forecasting
* Model serialization using Pickle
* Flask deployment
* HTML/CSS integration
* Building an end-to-end machine-learning application

---

## 👨‍💻 Project

**Sales Forecasting Using Machine Learning**

Built using:

Python + Pandas + NumPy + Scikit-learn + Flask + HTML + CSS

---

## 📜 License

This project is intended for educational and learning purposes.


