# Bangalore Real Estate Price Prediction

![License](https://img.shields.io/badge/license-MIT-blue.svg)
![Python](https://img.shields.io/badge/python-3.8%2B-blue.svg)
![Real Estate](https://img.shields.io/badge/Real-Estate%20Prediction-green.svg)

## Table of Contents

- [Introduction](#introduction)
- [Project Overview](#project-overview)
- [Technologies Used](#technologies-used)
- [Data Collection](#data-collection)
- [Exploratory Data Analysis (EDA)](#exploratory-data-analysis-eda)
- [Data Processing](#data-processing)
  - [Handling Missing Values](#handling-missing-values)
  - [Feature Engineering](#feature-engineering)
- [Modeling](#modeling)
- [Model Evaluation](#model-evaluation)
- [Results](#results)
- [Conclusion](#conclusion)
- [Project Structure](#project-structure)
- [License](#license)
- [Contact](#contact)

## Introduction

This project aims to predict real estate prices in Bangalore, using various features of properties such as location, square footage, number of bedrooms, and other relevant data. By applying machine learning, this project helps provide an estimate for property prices, which can assist potential buyers, investors, and real estate professionals.

## Project Overview

1. **Data Collection**: Compiling property data, including details such as location, area, price, number of bedrooms, and more.
2. **EDA**: Analyzing patterns in property prices based on location and other factors.
3. **Data Processing**: Handling missing values and performing feature engineering for optimal model performance.
4. **Modeling**: Training predictive models to estimate property prices.
5. **Evaluation**: Assessing model performance with metrics like Root Mean Squared Error (RMSE) and Mean Absolute Error (MAE).

## Technologies Used

- **Python**
- **Libraries**:
  - **Data Analysis**: Pandas, NumPy
  - **Visualization**: Matplotlib, Seaborn
  - **Machine Learning**: Scikit-Learn
  - **Metrics**: RMSE, MAE

## Data Collection

The dataset contains details of real estate properties in Bangalore, including:

- **Location**: City area where the property is located
- **Size**: Square footage or number of bedrooms
- **Price**: The target variable representing property price
- **Additional Features**: Information like bathrooms, balconies, amenities, etc.

This data was pre-processed and cleaned for effective analysis.

## Exploratory Data Analysis (EDA)

Key insights were drawn from the data by exploring relationships between features like:

- Location and price distribution
- Price per square foot
- Trends in property prices for different numbers of bedrooms

### Sample EDA Code

```python
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Load data
data = pd.read_csv('bangalore_real_estate.csv')

# Price distribution
sns.histplot(data['price'], bins=30, kde=True)
plt.title("Price Distribution")
plt.show()
```
## Data Processing
### Handling Missing Values
Missing values were handled by techniques like imputing with mean/median values for numerical columns and mode for categorical columns.

### Feature Engineering
Some key transformations:

- Location Encoding: Converting categorical locations into numerical formats.
- Outliers Removal: Removing properties with unusually high or low prices.
- Normalization: Normalizing numeric features to improve model training
## Modeling
Various regression models were applied to predict the price of a property:

- Linear Regression: A basic model that helps establish a baseline.
- Decision Tree Regression: A non-linear model capable of capturing complex patterns.
- Random Forest Regression: An ensemble method that combines multiple decision trees for robust predictions.
- XGBoost Regression: A powerful model that uses gradient boosting for enhanced performance.
## Model Evaluation
The models were evaluated using metrics like:

- Root Mean Squared Error (RMSE): Measures the average magnitude of prediction error.
- Mean Absolute Error (MAE): Represents the average absolute error between predictions and actual values.
## Results
A summary of model performance:

|Model|	RMSE|	MAE|
|------|--------|------|
|Linear Regression	| ... |	...|
|Decision Tree|	...|	...|
|Random Forest	|...	|...|
|XGBoost|...	|...|
## Conclusion
This project demonstrated the prediction of real estate prices using various machine learning models, with Random Forest or XGBoost typically yielding the most accurate results. The project illustrates effective feature engineering and modeling for real estate data.
## Project Structure
```css
bangalore-real-estate-prediction/
│
├── data/
│   ├── bangalore_real_estate.csv
│
├── notebooks/
│   ├── EDA.ipynb
│   ├── Model_Training.ipynb
│
├── src/
│   ├── preprocessing.py
│   ├── model_training.py
│
├── README.md
└── LICENSE
```
## License
This project is licensed under the Apache 2.0 License. See the LICENSE file for details.
## Contact
Mohammed Ammaruddin
md.ammaruddin2020@gmail.com
