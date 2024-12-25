import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report, accuracy_score
import joblib

# Load the dataset
df = pd.read_csv('fake_job_postings.csv')

# Split data into features and target
X = df.drop(columns=['fraudulent'])
y = df['fraudulent']

# Split data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define preprocessing steps for numeric and categorical features
numeric_features = ['salary_range']
numeric_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])

categorical_features = ['department', 'employment_type', 'required_experience', 'required_education', 'industry', 'function']
categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])

preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_features),
        ('cat', categorical_transformer, categorical_features)
    ])

# Append classifiers to preprocessing pipeline
clf_rf = Pipeline(steps=[('preprocessor', preprocessor),
                         ('classifier', RandomForestClassifier(random_state=42))])

clf_dt = Pipeline(steps=[('preprocessor', preprocessor),
                         ('classifier', DecisionTreeClassifier(random_state=42))])

clf_adaboost = Pipeline(steps=[('preprocessor', preprocessor),
                               ('classifier', AdaBoostClassifier(random_state=42))])

# Fit the models
clf_rf.fit(X_train, y_train)
clf_dt.fit(X_train, y_train)
clf_adaboost.fit(X_train, y_train)

# Predict on test data
y_pred_rf = clf_rf.predict(X_test)
y_pred_dt = clf_dt.predict(X_test)
y_pred_adaboost = clf_adaboost.predict(X_test)

# Evaluate the models
print("Random Forest Classifier Accuracy:", accuracy_score(y_test, y_pred_rf))
print("Random Forest Classifier Classification Report:\n", classification_report(y_test, y_pred_rf))

print("Decision Tree Classifier Accuracy:", accuracy_score(y_test, y_pred_dt))
print("Decision Tree Classifier Classification Report:\n", classification_report(y_test, y_pred_dt))

print("AdaBoost Classifier Accuracy:", accuracy_score(y_test, y_pred_adaboost))
print("AdaBoost Classifier Classification Report:\n", classification_report(y_test, y_pred_adaboost))

# Save the trained models
joblib.dump(clf_rf, 'trained_model_rf.joblib')
joblib.dump(clf_dt, 'trained_model_dt.joblib')
joblib.dump(clf_adaboost, 'trained_model_adaboost.joblib')
