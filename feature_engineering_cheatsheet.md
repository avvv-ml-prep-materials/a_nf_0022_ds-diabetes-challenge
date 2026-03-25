# Feature Engineering Cheatsheet

A concise guide to feature engineering techniques in Python using scikit-learn.

---

## Introduction to Feature Engineering

Feature engineering involves transforming raw data into meaningful features that better represent the underlying problem to predictive models, resulting in improved model performance.

---

## Best Practices  
1. **Early Train-Test Split**: Split your dataset into training and testing sets as early as possible to avoid data leakage.

2. **Consistent Transformation**: Apply the same feature engineering steps to both training and testing data.

3. **Pipeline Usage**: Utilize `Pipeline` and `ColumnTransformer` to streamline preprocessing and ensure consistency.

---

## Common Feature Engineering Techniques

### 1. Imputation

Handling missing data by filling in or removing the missing values.

#### **Techniques**

- **SimpleImputer**: Fill missing values with a constant, mean, median, or most frequent value.
  
```python
  from sklearn.impute import SimpleImputer
  
  # For numerical data
  num_imputer = SimpleImputer(strategy='mean')
  
  # For categorical data
  cat_imputer = SimpleImputer(strategy='most_frequent')
```

- **KNNImputer**: Use k-Nearest Neighbors to impute missing values based on feature similarity.
  
```python
  from sklearn.impute import KNNImputer
  
  knn_imputer = KNNImputer(n_neighbors=5)
```

### 2. Categorical Encoding

Converting categorical variables into numerical format for model compatibility.

#### **Techniques**

- **OneHotEncoder**: Convert categorical variables into one-hot encoded vectors.
  
```python
  from sklearn.preprocessing import OneHotEncoder
  
  ohe = OneHotEncoder(drop='first', handle_unknown='ignore')
```

- **get_dummies**: Quickly create dummy variables using pandas.
  
```python
  import pandas as pd
  
  dummies = pd.get_dummies(data['categorical_feature'], drop_first=True)
```

### 3. Feature Scaling

Rescaling features to ensure that they contribute equally to the model.

#### **Techniques**

- **StandardScaler**: Standardize features by removing the mean and scaling to unit variance.
  
```python
  from sklearn.preprocessing import StandardScaler
  
  scaler = StandardScaler()
```

- **MinMaxScaler**: Scale features to a specified range, typically [0, 1].
  
```python
  from sklearn.preprocessing import MinMaxScaler
  
  scaler = MinMaxScaler()
```

### 4. Feature Expansion

Creating new features from existing ones to capture non-linear relationships.

#### **Techniques**

- **PolynomialFeatures**: Generate polynomial and interaction features.
  
```python
  from sklearn.preprocessing import PolynomialFeatures
  
  poly = PolynomialFeatures(degree=2, include_bias=False)
```

### 5. Discretization

Transforming continuous variables into discrete bins.

#### **Techniques**

- **KBinsDiscretizer**: Bin continuous data into intervals.
  
```python
  from sklearn.preprocessing import KBinsDiscretizer
  
  discretizer = KBinsDiscretizer(n_bins=5, encode='onehot-dense', strategy='quantile')
```

---

## Implementing Feature Engineering with Pipelines

Utilize `Pipeline` and `ColumnTransformer` to apply transformations efficiently.

### **Example: Preprocessing Pipeline**

```python
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler, PolynomialFeatures

# Define numerical and categorical features
numerical_features = ['num_feature1', 'num_feature2']
categorical_features = ['cat_feature1', 'cat_feature2']

# Numerical pipeline
num_pipeline = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='mean')),
    ('scaler', StandardScaler()),
    ('poly_features', PolynomialFeatures(degree=2, include_bias=False))
])

# Categorical pipeline
cat_pipeline = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(drop='first', handle_unknown='ignore'))
])

# Combine pipelines using ColumnTransformer
preprocessor = ColumnTransformer(transformers=[
    ('num', num_pipeline, numerical_features),
    ('cat', cat_pipeline, categorical_features)
])
```

### **Integrate with Model**

```python
from sklearn.linear_model import LinearRegression

# Complete pipeline with model
model_pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('regressor', LinearRegression())
])

# Fit the model
model_pipeline.fit(X_train, y_train)

# Evaluate the model
train_score = model_pipeline.score(X_train, y_train)
test_score = model_pipeline.score(X_test, y_test)
```

---

## Practical Tips

- **Handle Missing Values First**: Always impute or remove missing values before applying other transformations.
- **Scale After Encoding**: For pipelines, ensure that encoding of categorical variables is done before scaling.
- **Use `include_bias=False`**: When using `PolynomialFeatures`, set `include_bias=False` to avoid adding an extra bias term.
- **Check for Data Leakage**: Be cautious not to use information from the test set during training.

---

## Additional Code Snippets

### **Imputing Missing Values**

```python
# Impute missing numerical values with mean
num_imputer = SimpleImputer(strategy='mean')
X_train_num = num_imputer.fit_transform(X_train[numerical_features])

# Impute missing categorical values with the most frequent value
cat_imputer = SimpleImputer(strategy='most_frequent')
X_train_cat = cat_imputer.fit_transform(X_train[categorical_features])
```

### **Encoding Categorical Variables**

```python
# One-hot encoding
ohe = OneHotEncoder(drop='first', handle_unknown='ignore')
X_train_cat_encoded = ohe.fit_transform(X_train_cat)
```

### **Scaling Features**

```python
# Standard scaling
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_num)

# Min-max scaling
minmax_scaler = MinMaxScaler()
X_train_minmax_scaled = minmax_scaler.fit_transform(X_train_num)
```

### **Generating Polynomial Features**

```python
# Polynomial features of degree 2
poly = PolynomialFeatures(degree=2, include_bias=False)
X_train_poly = poly.fit_transform(X_train_scaled)
```

### **Discretizing Features**

```python
# Discretize numerical features into bins
discretizer = KBinsDiscretizer(n_bins=4, encode='ordinal', strategy='uniform')
X_train_binned = discretizer.fit_transform(X_train[['num_feature']])
```

---

## Common Pitfalls to Avoid

- **Not Applying the Same Transformations to Test Data**: Always fit transformers on training data and apply them to both training and test sets.
- **Ignoring Data Types**: Ensure that numerical and categorical features are correctly identified.
- **Overfitting with Polynomial Features**: Using high-degree polynomials can lead to overfitting; use cross-validation to find the optimal degree.

---

## Useful Functions and Classes

- **Imputers**: `SimpleImputer`, `KNNImputer`
- **Encoders**: `OneHotEncoder`
- **Scalers**: `StandardScaler`, `MinMaxScaler`
- **Feature Generators**: `PolynomialFeatures`
- **Discretizers**: `KBinsDiscretizer`
- **Pipelines**: `Pipeline`, `ColumnTransformer`

---

## Example Workflow

```python
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

# Split data
X = data.drop('target', axis=1)
y = data['target']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

# Define features
numerical_features = ['num_feature1', 'num_feature2']
categorical_features = ['cat_feature1', 'cat_feature2']

# Build preprocessing pipelines (as shown above)

# Build complete pipeline with model
model_pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('regressor', LinearRegression())
])

# Train model
model_pipeline.fit(X_train, y_train)

# Evaluate model
print(f"Training Score: {model_pipeline.score(X_train, y_train):.4f}")
print(f"Test Score: {model_pipeline.score(X_test, y_test):.4f}")
```

---

## Conclusion

Feature engineering is a critical step in the machine learning pipeline. By systematically applying the techniques outlined in this cheatsheet and utilizing scikit-learn's powerful tools, you can enhance model performance and gain deeper insights from your data.