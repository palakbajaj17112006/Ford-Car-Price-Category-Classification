# Ford Used-Car Price Category Classification

A machine learning classification project that predicts the price category of used Ford cars based on vehicle attributes.

## Project Overview

This project converts the original continuous vehicle price into three categories and trains multiple machine learning classification models to predict those categories.

The workflow includes:

* Data loading
* Dataset inspection
* Missing-value analysis
* Exploratory Data Analysis
* Correlation analysis
* Categorical encoding
* Price-category creation
* Train-test splitting
* Feature scaling
* Classification
* Cross-validation
* Hyperparameter tuning using GridSearchCV
* Model comparison

## Dataset

The project uses a Ford used-car dataset stored in:

```text
ford.csv
```

The dataset contains **17,966 records**.

### Features

| Feature      | Description        |
| ------------ | ------------------ |
| model        | Ford vehicle model |
| year         | Manufacturing year |
| price        | Vehicle price      |
| transmission | Transmission type  |
| mileage      | Vehicle mileage    |
| fuelType     | Fuel type          |
| tax          | Vehicle tax        |
| mpg          | Miles per gallon   |
| engineSize   | Engine size        |

The target variable is derived from the original `price` column.

## Data Inspection

The notebook checks the dataset for missing values.

The recorded missing-value check shows zero missing values for the listed columns.

## Exploratory Data Analysis

The notebook performs several visual analyses.

### Box Plots

Box plots are created for:

* Model vs Price
* Price
* Mileage

### Scatter Plots

Price is compared against:

* Year
* Mileage
* Tax
* MPG
* Engine Size

## Correlation Analysis

The notebook records the following correlations with price:

| Feature    | Correlation |
| ---------- | ----------: |
| year       |    0.636009 |
| mileage    |   -0.530659 |
| engineSize |    0.411178 |
| tax        |    0.406857 |
| mpg        |   -0.346419 |

`year` has the strongest positive correlation with price, while `mileage` has the strongest negative correlation among the listed numerical features.

## Categorical Encoding

Categorical columns are converted into numerical values using `LabelEncoder`.

The notebook applies encoding to variables such as:

```text
model
transmission
fuelType
```

This allows the machine learning algorithms to process categorical information numerically.

## Creating Price Categories

The continuous `price` variable is converted into three quantile-based categories using:

```python
pd.qcut(
    df["price"],
    q=3,
    labels=[0, 1, 2]
)
```

The resulting target column is:

```text
price_category
```

The original `price` column is separated from the input features.

Therefore, this project performs **three-class classification** rather than exact price prediction.

## Train-Test Split

The notebook uses:

```python
train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)
```

Recorded dimensions:

```text
X_train: 14372 rows
X_test:   3594 rows
y_train: 14372 rows
y_test:   3594 rows
```

## Feature Scaling

`StandardScaler` is used to create:

```text
X_train_scaled
X_test_scaled
```

The scaler is fitted on the training data and then applied to the test data.

## Machine Learning Models

Four classification algorithms are evaluated.

### Logistic Regression

Initial test accuracy:

```text
75.49%
```

GridSearchCV tunes:

* C
* penalty
* solver

Best recorded parameters:

```text
C = 0.1
penalty = l2
solver = saga
```

Best recorded cross-validation accuracy:

```text
76.25%
```

Tuned test accuracy:

```text
75.51%
```

### Decision Tree

Initial test accuracy:

```text
82.75%
```

GridSearchCV tunes:

* criterion
* max_depth
* min_samples_split
* min_samples_leaf

Best recorded parameters:

```text
criterion = gini
max_depth = 10
min_samples_leaf = 1
min_samples_split = 10
```

Best recorded cross-validation accuracy:

```text
84.76%
```

### K-Nearest Neighbors

Initial configuration:

```text
n_neighbors = 5
```

Initial test accuracy:

```text
86.37%
```

GridSearchCV tunes:

* Number of neighbors
* Weighting
* Distance metric

Best recorded configuration:

```text
n_neighbors = 15
weights = uniform
metric = manhattan
```

Best recorded cross-validation accuracy:

```text
85.99%
```

### Support Vector Classifier

Initial test accuracy:

```text
84.61%
```

GridSearchCV searches over:

* C
* kernel
* gamma

The notebook performs the grid search, but does not record a final best SVC parameter set and tuned test accuracy in the displayed final output.

## Model Comparison

The recorded initial test accuracies are:

| Model               | Accuracy |
| ------------------- | -------: |
| KNN                 |   86.37% |
| SVC                 |   84.61% |
| Decision Tree       |   82.75% |
| Logistic Regression |   75.49% |

Based on the recorded initial test results, **KNN achieved the highest test accuracy at approximately 86.37%**.

## Technologies Used

* Python
* Jupyter Notebook
* Pandas
* NumPy
* Matplotlib
* Seaborn
* Scikit-learn

## Installation

Create a virtual environment:

```bash
python -m venv .venv
```

Activate it on Windows:

```bash
.venv\Scripts\activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

## Running the Project

Start Jupyter Notebook:

```bash
jupyter notebook
```

Open:

```text
ml(3).ipynb
```

Make sure:

```text
ford.csv
```

is available.

The notebook loads the dataset using:

```python
df = pd.read_csv("ford.csv")
```

## Project Structure

```text
Ford-Used-Car-Classification/
│
├── README.md
├── requirements.txt
├── ml(3).ipynb
└── ford.csv
```

## Machine Learning Concepts Demonstrated

* Multiclass classification
* Exploratory Data Analysis
* Label encoding
* Feature scaling
* Train-test splitting
* Cross-validation
* GridSearchCV
* Hyperparameter tuning
* Logistic Regression
* Decision Trees
* K-Nearest Neighbors
* Support Vector Classification

## Limitations

The model predicts three price categories rather than an exact vehicle price.

The three categories are generated using quantile-based binning, so their boundaries depend on the price distribution of the dataset.

The results may change when a different dataset or preprocessing approach is used.

## Future Improvements

* Use a complete preprocessing pipeline
* Compare additional classification algorithms
* Add precision, recall and F1-score
* Add confusion matrices
* Add classification reports
* Experiment with alternative categorical encoding
* Save the best trained model
* Build a simple prediction interface
* Add automated testing

## Disclaimer

This project is intended for educational and experimental purposes.

The predicted price categories should not be treated as guaranteed market valuations.
