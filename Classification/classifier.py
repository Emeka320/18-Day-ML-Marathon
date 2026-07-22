import numpy as np
import pandas as pd
from itertools import combinations
from sklearn.metrics import accuracy_score
from sklearn.metrics import mean_squared_error
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report


#-------------------------------------------------------------------------

data = pd.read_csv('Classification/shopping_trends.csv')
print(f"Dataset: \n{data}")

backup_data = data.copy()

#-------------------------------------------------------------------------

columns = ['Promo Code Used']

data = data.drop(columns=columns)
print(data)


#-------------------------------------------------------------------------
# Feature Engineering

one_hot = OneHotEncoder(drop='first', sparse_output=False)

# Replace gender column with encoded values
data['Gender'] = one_hot.fit_transform(data[['Gender']]).astype(int)
data['Discount Applied'] = one_hot.fit_transform(data[['Discount Applied']]).astype(int)
data['Subscription Status'] = one_hot.fit_transform(data[['Subscription Status']]).astype(int)


one_hot_columns = ['Item Purchased', 'Category', 'Location', 'Size', 'Color', 'Season', 'Payment Method', 'Shipping Type',
                   'Preferred Payment Method', 'Frequency of Purchases']

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
id = data['Customer ID']
encoded_df.insert(0, 'Customer ID', id)


prefixes = ['Item Purchased', 'Category', 'Location', 'Size', 'Color', 'Season', 'Payment Method', 'Shipping Type',
            'Preferred Payment Method', 'Frequency of Purchases']

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

#-------------------------------------------------------------------------

columnss = ['Item Purchased', 'Category', 'Location', 'Size', 'Color', 'Season', 'Payment Method', 'Shipping Type',
            'Preferred Payment Method', 'Frequency of Purchases']

dropped_data = data.drop(columns=columnss)


full_encoded = pd.merge(dropped_data, encoded_df, on="Customer ID")
drop = ['Customer ID']
full_encoded = full_encoded.drop(columns=drop)
print(f"Fully Encoded Data: \n{full_encoded}")

#-------------------------------------------------------------------------

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

#----------------------------------------------------------------------------

columns = list(cleaned_data.columns)
print(columns)

corr = cleaned_data.corr()
print(corr)

for one, two in combinations(columns, 2):
    correlation = cleaned_data[one].corr(cleaned_data[two])

    print(f"{one} vs {two} -> Correlation: {correlation}")


#-----------------------------------------------------------------------------------------------------------------------------------
# Data Preparations

X = cleaned_data[['Age', 'Gender', 'Purchase Amount (USD)', 'Review Rating', 'Subscription Status', 'Previous Purchases', 
                'Item Purchased', 'Category', 'Location', 'Size', 'Color', 'Season', 'Payment Method', 'Shipping Type', 
                'Preferred Payment Method', 'Frequency of Purchases']]

y = cleaned_data['Discount Applied']

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

#-----------------------------------------------------------------------------------------------------------------------------------
# 1. KNeighbors Classifier (try different k values)

print("\n" + "="*60)
print("KNEIGHBORS CLASSIFIER - TUNING K VALUES")
print("="*60)

k_values = range(1, 21)
knn_scores = []
knn_cv_scores = []

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

for c in c_values:
    lr = LogisticRegression(C=c, max_iter=1000, random_state=42)
    lr.fit(X_train_scaled, y_train)
    y_pred = lr.predict(X_test_scaled)
    accuracy = accuracy_score(y_test, y_pred)
    lr_scores.append(accuracy)
    cv_score = cross_val_score(lr, X_train_scaled, y_train, cv=5, scoring='accuracy').mean()
    lr_cv_scores.append(cv_score)
    print(f"C={c}: Test Accuracy = {accuracy:.4f} | CV Accuracy = {cv_score:.4f}")

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

for n in n_values:
    rf = RandomForestClassifier(n_estimators=n, random_state=42, n_jobs=-1)
    rf.fit(X_train_scaled, y_train)
    y_pred = rf.predict(X_test_scaled)
    accuracy = accuracy_score(y_test, y_pred)
    rf_scores.append(accuracy)
    cv_score = cross_val_score(rf, X_train_scaled, y_train, cv=5, scoring='accuracy').mean()
    rf_cv_scores.append(cv_score)
    print(f"n_estimators={n}: Test Accuracy = {accuracy:.4f} | CV Accuracy = {cv_score:.4f}")

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

for depth in depth_values:
    rf = RandomForestClassifier(n_estimators=best_cv_n, max_depth=depth, random_state=42, n_jobs=-1)
    rf.fit(X_train_scaled, y_train)
    y_pred = rf.predict(X_test_scaled)
    accuracy = accuracy_score(y_test, y_pred)
    rf_depth_scores.append(accuracy)
    cv_score = cross_val_score(rf, X_train_scaled, y_train, cv=5, scoring='accuracy').mean()
    rf_depth_cv_scores.append(cv_score)
    print(f"max_depth={depth}: Test Accuracy = {accuracy:.4f} | CV Accuracy = {cv_score:.4f}")

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
print(classification_report(y_test, best_knn.predict(X_test_scaled), target_names=['No Discount', 'Discount Used']))

print("\nLogistic Regression:")
print(confusion_matrix(y_test, best_lr.predict(X_test_scaled)))
print(classification_report(y_test, best_lr.predict(X_test_scaled), target_names=['No Discount', 'Discount Used']))

print("\nRandom Forest:")
print(confusion_matrix(y_test, best_rf.predict(X_test_scaled)))
print(classification_report(y_test, best_rf.predict(X_test_scaled), target_names=['No Discount', 'Discount Used']))