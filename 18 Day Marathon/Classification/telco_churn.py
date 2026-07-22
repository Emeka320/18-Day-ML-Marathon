import time
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from itertools import combinations
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score
from sklearn.metrics import mean_squared_error
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report


#-------------------------------------------------------------------------
# Load Dataset

data = pd.read_csv('Data/WA_Fn-UseC_-Telco-Customer-Churn.csv')
print(f"Dataset: \n{data}")

backup_data = data.copy()

#-------------------------------------------------------------------------
# Inspection

print(data.columns)
print(data.dtypes)
print(data.isna().isnull().sum())


columns = ['customerID']

data = data.drop(columns=columns)
data.insert(0, 'ID', range(1, len(data) + 1))
print(data)

#-------------------------------------------------------------------------
# Feature Engineering

one_hot = OneHotEncoder(drop='first', sparse_output=False)

# Replace gender column with encoded values
data['Churn'] = one_hot.fit_transform(data[['Churn']]).astype(int)
data['gender'] = one_hot.fit_transform(data[['gender']]).astype(int)
data['Partner'] = one_hot.fit_transform(data[['Partner']]).astype(int)
data['Dependents'] = one_hot.fit_transform(data[['Dependents']]).astype(int)
data['PhoneService'] = one_hot.fit_transform(data[['PhoneService']]).astype(int)
data['PaperlessBilling'] = one_hot.fit_transform(data[['PaperlessBilling']]).astype(int)

one_hot_columns = ['MultipleLines', 'InternetService','OnlineSecurity', 'OnlineBackup', 
    'DeviceProtection', 'TechSupport', 'StreamingTV', 'StreamingMovies', 'Contract', 'PaymentMethod', 'TotalCharges']

one_hot = OneHotEncoder(drop='first', sparse_output=False)

def encode(df, one_hot_columns):
    encoded_array = one_hot.fit_transform(df[one_hot_columns])

    encoded_df = pd.DataFrame(
        encoded_array,
        columns=one_hot.get_feature_names_out(one_hot_columns),
        index=df.index
    )

    return encoded_df

encoded_df = encode(data, one_hot_columns)
id = data['ID']
encoded_df.insert(0, 'ID', id)


prefixes = ['MultipleLines', 'InternetService','OnlineSecurity', 'OnlineBackup', 
    'DeviceProtection', 'TechSupport', 'StreamingTV', 'StreamingMovies', 'Contract', 'PaymentMethod', 'TotalCharges']

for prefix in prefixes:

    cols = [c for c in encoded_df.columns if c.startswith(prefix + '_')]

    encoded_df[prefix] = (
        encoded_df[cols]
        .idxmax(axis=1)
        .str.replace(prefix + '_', '', regex=False)
        .astype('category')
        .cat.codes
    )

    encoded_df.drop(columns=cols, inplace=True)

#-------------------------------------------------------------------------------------
# Encoded Data

columnss = ['MultipleLines', 'InternetService','OnlineSecurity', 'OnlineBackup', 
    'DeviceProtection', 'TechSupport', 'StreamingTV', 'StreamingMovies', 'Contract', 'PaymentMethod', 'TotalCharges']

dropped_data = data.drop(columns=columnss)


full_encoded = pd.merge(dropped_data, encoded_df, on="ID")
drop = ['ID']
full_encoded = full_encoded.drop(columns=drop)
print(f"Fully Encoded Data: \n{full_encoded}")

#-------------------------------------------------------------------------------------
# Cleaned Data

def get_duplicates(df):
    duplicates = df.duplicated()
    result = print(duplicates)
    return result

def drop_duplicates(df, subset=None):
    dup_mask = df.duplicated(subset=subset, keep='first')
    return df[~dup_mask]

cleaned_data = drop_duplicates(full_encoded)
cleaned_data = cleaned_data.dropna()
print(f"Fully Cleaned Data: \n{cleaned_data}")

#-------------------------------------------------------------------------------------
# Correlaton
columns = list(cleaned_data.columns)
print(columns)

corr = cleaned_data.corr()
print(corr)

print("\n")

for one, two in combinations(columns, 2):
    correlation = cleaned_data[one].corr(cleaned_data[two])

    print(f"{one} vs {two} -> Correlation: {correlation}")

#-------------------------------------------------------------------------
# Exploratory Data Analysis (EDA)
print(f"Description DataFrame: \n{data.describe()}")

print(f"\nInfo DataFrame: \n{data.info()}")

# Outlier detection using Interquartile range
def iqr_outlier(df):
    results = {}

    for col in df.columns:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1

        lower = Q1 - 1.5 * IQR
        upper = Q3 + 1.5 * IQR

        outliers = df[(df[col] < lower) | (df[col] > upper)]

        results[col] = {
            "lower_bound": lower,
            "upper_bound": upper,
            "outliers": outliers
        }

    return results

outlier_df = cleaned_data[[
    'SeniorCitizen', 'tenure', 'MonthlyCharges'
]]

outliers = iqr_outlier(outlier_df)

print(f"\nOutlier detection: \n{outliers}")

#-------------------------------------------------------------------------------------
# Feature Selcetion


X = cleaned_data[['gender', 'SeniorCitizen', 'Partner', 'Dependents', 'tenure', 'PhoneService', 'PaperlessBilling', 
                  'MonthlyCharges', 'MultipleLines', 'InternetService', 'OnlineSecurity', 'OnlineBackup', 'DeviceProtection', 
                  'TechSupport', 'StreamingTV', 'StreamingMovies', 'Contract', 'PaymentMethod', 'TotalCharges']]

y = cleaned_data['Churn']

print(f"\nFeatures variable: \n{X}")
print(f"Target variable: \n{y}")

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

#-----------------------------------------------------------------------------
# 1. KNeighbors Classifier (try different k values)


print("\n" + "="*60)
print("KNEIGHBORS CLASSIFIER - TUNING K VALUES")
print("="*60)

k_values = range(1, 21)
knn_scores = []
knn_cv_scores = []

# Measure knn train time
knn_train_start = time.perf_counter()

for k in k_values:
    # Initialize KNN
    knn = KNeighborsClassifier(n_neighbors=k, weights='distance')
    
    # Train the model
    knn.fit(X_train_scaled, y_train)
    
    # Predict on test set
    y_pred = knn.predict(X_test_scaled)
    
    # Calculate accuracy
    accuracy = accuracy_score(y_test, y_pred)
    knn_scores.append(accuracy)
    
    # Cross-validation score (more robust)
    cv_score = cross_val_score(knn, X_train_scaled, y_train, cv=5, scoring='accuracy').mean()
    knn_cv_scores.append(cv_score)
    
    print(f"k={k}: Test Accuracy = {accuracy:.4f} | CV Accuracy = {cv_score:.4f}")

knn_train_end = time.perf_counter()

# Best k
best_k = k_values[np.argmax(knn_scores)]
best_cv_k = k_values[np.argmax(knn_cv_scores)]

print(f"\n{'='*60}")
print(f"Best k by Test Accuracy: {best_k} (Score: {max(knn_scores):.4f})")
print(f"Best k by Cross-Validation: {best_cv_k} (Score: {max(knn_cv_scores):.4f})")

#-----------------------------------------------------------------------------
# 2. LOGISTIC REGRESSION
print("\n" + "="*60)
print("LOGISTIC REGRESSION - TUNING C VALUES")
print("="*60)

c_values = [0.001, 0.01, 0.1, 1, 10, 100, 1000]
lr_scores = []
lr_cv_scores = []

# Measure log regression train time
log_reg_train_start = time.perf_counter()
for c in c_values:
    lr = LogisticRegression(C=c, max_iter=1000, random_state=42)
    lr.fit(X_train_scaled, y_train)
    y_pred = lr.predict(X_test_scaled)
    accuracy = accuracy_score(y_test, y_pred)
    lr_scores.append(accuracy)
    cv_score = cross_val_score(lr, X_train_scaled, y_train, cv=5, scoring='accuracy').mean()
    lr_cv_scores.append(cv_score)
    print(f"C={c}: Test Accuracy = {accuracy:.4f} | CV Accuracy = {cv_score:.4f}")

log_reg_train_end = time.perf_counter()

best_c = c_values[np.argmax(lr_scores)]
best_cv_c = c_values[np.argmax(lr_cv_scores)]
print(f"\nBest C by Test Accuracy: {best_c} (Score: {max(lr_scores):.4f})")
print(f"Best C by Cross-Validation: {best_cv_c} (Score: {max(lr_cv_scores):.4f})")

#-----------------------------------------------------------------------------
# 3. RANDOM FOREST
print("\n" + "="*60)
print("RANDOM FOREST - TUNING N_ESTIMATORS")
print("="*60)

n_values = [50, 100, 150, 200, 250, 300]
rf_scores = []
rf_cv_scores = []

# Measure random forest train time
forest_train_time_start = time.perf_counter()
for n in n_values:
    rf = RandomForestClassifier(n_estimators=n, random_state=42, n_jobs=-1)
    rf.fit(X_train_scaled, y_train)
    y_pred = rf.predict(X_test_scaled)
    accuracy = accuracy_score(y_test, y_pred)
    rf_scores.append(accuracy)
    cv_score = cross_val_score(rf, X_train_scaled, y_train, cv=5, scoring='accuracy').mean()
    rf_cv_scores.append(cv_score)
    print(f"n_estimators={n}: Test Accuracy = {accuracy:.4f} | CV Accuracy = {cv_score:.4f}")

forest_train_time_end = time.perf_counter()

best_n = n_values[np.argmax(rf_scores)]
best_cv_n = n_values[np.argmax(rf_cv_scores)]
print(f"\nBest n_estimators by Test Accuracy: {best_n} (Score: {max(rf_scores):.4f})")
print(f"Best n_estimators by Cross-Validation: {best_cv_n} (Score: {max(rf_cv_scores):.4f})")

#-----------------------------------------------------------------------------
# 4. RANDOM FOREST - TUNING MAX_DEPTH
print("\n" + "="*60)
print("RANDOM FOREST - TUNING MAX_DEPTH")
print("="*60)

depth_values = [5, 10, 15, 20, 25, 30, None]
rf_depth_scores = []
rf_depth_cv_scores = []

random_forest_train_time_start = time.perf_counter()
for depth in depth_values:
    rf = RandomForestClassifier(n_estimators=best_cv_n, max_depth=depth, random_state=42, n_jobs=-1)
    rf.fit(X_train_scaled, y_train)
    y_pred = rf.predict(X_test_scaled)
    accuracy = accuracy_score(y_test, y_pred)
    rf_depth_scores.append(accuracy)
    cv_score = cross_val_score(rf, X_train_scaled, y_train, cv=5, scoring='accuracy').mean()
    rf_depth_cv_scores.append(cv_score)
    print(f"max_depth={depth}: Test Accuracy = {accuracy:.4f} | CV Accuracy = {cv_score:.4f}")

random_forest_train_time_end = time.perf_counter()

best_depth = depth_values[np.argmax(rf_depth_scores)]
best_cv_depth = depth_values[np.argmax(rf_depth_cv_scores)]
print(f"\nBest max_depth by Test Accuracy: {best_depth} (Score: {max(rf_depth_scores):.4f})")
print(f"Best max_depth by Cross-Validation: {best_cv_depth} (Score: {max(rf_depth_cv_scores):.4f})")

#-----------------------------------------------------------------------------
# 5. FINAL COMPARISON
print("\n" + "="*60)
print("FINAL MODEL COMPARISON")
print("="*60)

# Best models
best_knn = KNeighborsClassifier(n_neighbors=best_cv_k, weights='distance')
best_knn.fit(X_train_scaled, y_train)
knn_final_acc = accuracy_score(y_test, best_knn.predict(X_test_scaled))

best_lr = LogisticRegression(C=best_cv_c, max_iter=1000, random_state=42)
best_lr.fit(X_train_scaled, y_train)
lr_final_acc = accuracy_score(y_test, best_lr.predict(X_test_scaled))

best_rf = RandomForestClassifier(n_estimators=best_cv_n, max_depth=best_cv_depth, random_state=42, n_jobs=-1)
best_rf.fit(X_train_scaled, y_train)
rf_final_acc = accuracy_score(y_test, best_rf.predict(X_test_scaled))

print(f"\nKNN (k={best_cv_k}):          {knn_final_acc:.4f}")
print(f"Logistic Regression (C={best_cv_c}): {lr_final_acc:.4f}")
print(f"Random Forest (n={best_cv_n}, depth={best_cv_depth}): {rf_final_acc:.4f}")

# Best overall
best_model_name = max([('KNN', knn_final_acc), ('Logistic Regression', lr_final_acc), ('Random Forest', rf_final_acc)], key=lambda x: x[1])
print(f"\n✅ Best Model: {best_model_name[0]} (Accuracy: {best_model_name[1]:.4f})")

#-----------------------------------------------------------------------------
# 6. CONFUSION MATRICES FOR BEST MODELS

print("\n" + "="*60)
print("CONFUSION MATRICES")
print("="*60)

print("\nKNN Classifier:")
print(confusion_matrix(y_test, best_knn.predict(X_test_scaled)))
print(classification_report(y_test, best_knn.predict(X_test_scaled), target_names=['No Churn', 'Churn Yes']))

print("\nLogistic Regression:")
print(confusion_matrix(y_test, best_lr.predict(X_test_scaled)))
print(classification_report(y_test, best_lr.predict(X_test_scaled), target_names=['No Churn', 'Churn Yes']))

print("\nRandom Forest:")
print(confusion_matrix(y_test, best_rf.predict(X_test_scaled)))
print(classification_report(y_test, best_rf.predict(X_test_scaled), target_names=['No Churn', 'Churn Yes']))

#-----------------------------------------------------------------------------
# 7. Measure training time

print("\n" + "="*60)

print(f"KNN Train Time : {knn_train_end - knn_train_start:.2f} sec")
print(f"Logistic Regression Train Time : {log_reg_train_end - log_reg_train_start:.2f} sec")
print(f"RANDOM FOREST N_ESTIMATORS Train TIme : {forest_train_time_end - forest_train_time_start:.2f} sec")
print(f"RANDOM FOREST MAX_DEPTH Train Time : {random_forest_train_time_end - random_forest_train_time_start:.2f} sec")

print("="*60)