# Diamond Price Prediction

This project aims to predict the prices of diamonds using machine learning techniques. The dataset contains various features of diamonds such as carat, cut, color, clarity, and dimensions. The goal is to build a predictive model that can accurately estimate diamond prices based on these features.

## Table of Contents

1. [Installation](#installation)
2. [Data Description](#data-description)
3. [Preprocessing and Feature Engineering](#preprocessing-and-feature-engineering)
4. [Model Training](#model-training)
5. [Model Evaluation](#model-evaluation)
6. [Prediction on Test Data](#prediction-on-test-data)
7. [File Descriptions](#file-descriptions)
8. [Acknowledgments](#acknowledgments)

## Installation

To run this project, you need to have the following software and libraries installed:

- Python 3.x
- pandas
- h2o
- H2OAutoML

You can install the required Python libraries using pip:

```bash
pip install pandas h2o
```

## Data Description

The dataset consists of the following columns:

- `carat`: The weight of the diamond.
- `cut`: The quality of the cut (Fair, Good, Very Good, Premium, Ideal).
- `color`: The color of the diamond (ranging from D to J).
- `clarity`: The clarity of the diamond (I1, SI2, SI1, VS2, VS1, VVS2, VVS1, IF).
- `depth`: The depth percentage of the diamond.
- `table`: The width of the diamond's top relative to its widest point.
- `x`: Length in mm.
- `y`: Width in mm.
- `z`: Depth in mm.
- `price`: The price of the diamond (target variable).

## Preprocessing and Feature Engineering

We performed the following preprocessing and feature engineering steps:

1. **Ordinal Encoding**: Converted categorical variables (cut, color, clarity) to ordinal values.
2. **Feature Creation**: Created new features such as volume, density, surface area, depth percentage, and length ratios (xy, xz, yz).

Here is the code for preprocessing and feature engineering:

```python
import pandas as pd

df = pd.read_csv('train.csv')

# Ordinal encoding
c = {"Fair": 1, "Good": 2, "Very Good": 3, "Premium": 4, "Ideal": 5}
df["cut_ordinal"] = df['cut'].map(c)
b = {'D': 7, 'E': 6, 'F': 5, 'G': 4, 'H': 3, 'I': 2, 'J': 1}
df['color_ordinal'] = df['color'].map(b)
e = {'I1': 1, 'SI2': 2, 'SI1': 2, 'VS2': 3, 'VS1': 4, 'VVS2': 5, 'VVS1': 6, 'IF': 7}
df['clarity_ordinal'] = df['clarity'].map(e)

# Feature creation
df['volume'] = df['x'] * df['y'] * df['z']
df['density'] = df['carat'] / df['volume']
df['surface_area'] = 2 * (df['x'] * df['y'] + df['x'] * df['z'] + df['y'] * df['z'])
df['depth_percentage'] = (df['depth'] / ((df['x'] + df['y'] + df['z']) / 3)) * 100
df['length_ratio_xy'] = df['x'] / df['y']
df['length_ratio_xz'] = df['x'] / df['z']
df['length_ratio_yz'] = df['y'] / df['z']

df = df.drop(['Id'], axis=1)
```

## Model Training

We used H2O's AutoML to train the model. The AutoML process includes automated training and tuning of multiple models within a specified time limit, selecting the best-performing model.

```python
import h2o
from h2o.automl import H2OAutoML

h2o.init()

h2o_data = h2o.H2OFrame(df)
h2o_data['clarity'] = h2o_data['clarity'].asfactor()
h2o_data['color'] = h2o_data['color'].asfactor()
h2o_data['cut'] = h2o_data['cut'].asfactor()

aml = H2OAutoML(max_runtime_secs=700, seed=789)
aml.train(y="price", training_frame=h2o_data)

best_model = aml.leader
print(best_model)
```

## Model Evaluation

The best model can be evaluated using the H2O framework's built-in functions. The model's performance is measured based on metrics such as RMSE, MAE, and R².

## Prediction on Test Data

To make predictions on the test data, we follow similar preprocessing steps and use the trained model to predict diamond prices.

```python
test_data = pd.read_csv('test.csv')

# Preprocessing on test data
test_data["cut_ordinal"] = test_data['cut'].map(c)
test_data['color_ordinal'] = test_data['color'].map(b)
test_data['clarity_ordinal'] = test_data['clarity'].map(e)
test_data['volume'] = test_data['x'] * test_data['y'] * test_data['z']
test_data['density'] = test_data['carat'] / test_data['volume']
test_data['surface_area'] = 2 * (test_data['x'] * test_data['y'] + test_data['x'] * test_data['z'] + test_data['y'] * test_data['z'])
test_data['depth_percentage'] = (test_data['depth'] / ((test_data['x'] + test_data['y'] + test_data['z']) / 3)) * 100
test_data['length_ratio_xy'] = test_data['x'] / test_data['y']
test_data['length_ratio_xz'] = test_data['x'] / test_data['z']
test_data['length_ratio_yz'] = test_data['y'] / test_data['z']

h2o_train = h2o.H2OFrame(test_data)
h2o_train['clarity'] = h2o_train['clarity'].asfactor()
h2o_train['color'] = h2o_train['color'].asfactor()
h2o_train['cut'] = h2o_train['cut'].asfactor()

predictions = best_model.predict(h2o_train)
predictions_df = predictions.as_data_frame()
predictions_df['Id'] = test_data['Id']
predictions_df = predictions_df[['Id', 'predict']]
predictions_df = predictions_df.rename(columns={'predict': 'price'})

predictions_df.to_csv('h2o_final.csv', index=False)
```

## File Descriptions

- `train.csv`: Training dataset containing features and target variable (price).
- `test.csv`: Test dataset containing features without target variable.
- `h2o_final.csv`: Output file containing predicted prices for the test dataset.
- `README.md`: This README file.

## Acknowledgments

This project utilizes the H2O.ai machine learning platform for training and evaluating the predictive model. Special thanks to the contributors of the dataset and the developers of H2O for providing robust tools for machine learning.
