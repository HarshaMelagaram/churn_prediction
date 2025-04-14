import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
import joblib

# Load dataset
df = pd.read_csv('data/churn.csv')

# Drop unnecessary columns
df.drop('customerID', axis=1, inplace=True)

# Fix TotalCharges
df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
df['TotalCharges'].fillna(df['TotalCharges'].median(), inplace=True)

# Encode target column
df['Churn'] = df['Churn'].map({'Yes': 1, 'No': 0})

# Define features and target
features = ['gender', 'Partner', 'Dependents', 'PhoneService', 'PaperlessBilling',
            'tenure', 'MonthlyCharges', 'TotalCharges']
target = 'Churn'

X = df[features]
y = df[target]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Column definitions
numerical_cols = ['tenure', 'MonthlyCharges', 'TotalCharges']
categorical_cols = ['gender', 'Partner', 'Dependents', 'PhoneService', 'PaperlessBilling']

# Preprocessors
numeric_transformer = Pipeline(steps=[
    ('scaler', StandardScaler())
])
categorical_transformer = Pipeline(steps=[
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])
preprocessor = ColumnTransformer(transformers=[
    ('num', numeric_transformer, numerical_cols),
    ('cat', categorical_transformer, categorical_cols)
])

# Full pipeline
pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', LogisticRegression())
])

# Fit the model
pipeline.fit(X_train, y_train)

# Accuracy
accuracy = pipeline.score(X_test, y_test)
print(f"Model Accuracy on Test Set: {accuracy:.4f}")

# Save model
joblib.dump(pipeline, 'model/churn_model_pipeline.pkl')
