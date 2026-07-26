import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from itertools import combinations
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import root_mean_squared_error,  mean_squared_error, r2_score
from sklearn.linear_model import Ridge, Lasso, RidgeCV, LassoCV, LinearRegression



#-------------------------------------------------------------------------------------
# Load datasets

sim_data = pd.read_csv('Data/mortgage_loan_dataset.csv')
sim_data.insert(0, 'ID', range(1, len(sim_data) + 1))
print("="*60)
print(f"MORTGAGE LOAN DATASET: \n{sim_data}")
print("="*60)

print(f"Data Info: \n{sim_data.info()}")

print(f"\nData Description: \n{sim_data.describe()}")

#-------------------------------------------------------------------------------------
# Feature engineering

one_hot = OneHotEncoder(drop='first', sparse_output=False)

# Replace binary columns with encoded values
sim_data['Gender'] = one_hot.fit_transform(sim_data[['Gender']]).astype(int)
sim_data['Married'] = one_hot.fit_transform(sim_data[['Married']]).astype(int)


one_hot_columns = ['Education', 'Job', 'Area']

one_hot = OneHotEncoder(drop='first', sparse_output=False)


def encode(df, one_hot_columns):
    encoded_array = one_hot.fit_transform(df[one_hot_columns])

    encoded_df = pd.DataFrame(
        encoded_array,
        columns=one_hot.get_feature_names_out(one_hot_columns),
        index=df.index
    )

    return encoded_df

encoded_df = encode(sim_data, one_hot_columns)
id = sim_data['ID']
encoded_df.insert(0, 'ID', id)

prefixes = ['Education', 'Job', 'Area']

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
# Cleaned Data

columnss = ['Education', 'Job', 'Area']

dropped_data = sim_data.drop(columns=columnss)


full_encoded = pd.merge(dropped_data, encoded_df, on="ID")
drop = ['ID']
full_encoded = full_encoded.drop(columns=drop)
print(f"Fully Encoded Data: \n{full_encoded}")

#-------------------------------------------------------------------------------------

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

columns = list(cleaned_data.columns)
print(columns)

print("\n")

corr = cleaned_data.corr()
print(corr)

print("\n")

for one, two in combinations(columns, 2):
    correlation = cleaned_data[one].corr(cleaned_data[two])

    print(f"{one} vs {two} -> Correlation: {correlation}")

plt.figure(figsize=(14, 12))

# Create heatmap
sns.heatmap(corr, 
            annot=True,           # Show correlation values
            fmt='.2f',            # 2 decimal places
            cmap='RdBu_r',        # Red-Blue color scheme (red=positive, blue=negative)
            center=0,             # Center colormap at 0
            square=True,          # Square cells
            linewidths=0.5,       # Grid lines
            cbar_kws={'shrink': 0.8, 'label': 'Correlation Coefficient'},
            vmin=-1, vmax=1)

plt.title('Feature Correlation Heatmap', fontsize=16, fontweight='bold')
plt.tight_layout()
plt.savefig('correlation_heatmap.png', dpi=150, bbox_inches='tight')
plt.show()

#-------------------------------------------------------------------------------------
# Data Preparations

X = cleaned_data[['Gender', 'Age', 'Married', 'Employment Years', 'Annual Income (USD)', 'Interest Rate', 'Down Payment (USD)', 
                   'Credit Score', 'Existing Monthly Debt (USD)', 'Loans Repaid', 'Education', 'Job', 'Area']]

y = cleaned_data['Max Loan Amount (USD)']

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)


#-------------------------------------------------------------------------------------
# Model training


# BASE LINEAR REGRESSION


print("\n" + "="*60)
print("LINEAR REGRESSION")
print("="*60)

model = LinearRegression()

model.fit(X_train_scaled, y_train)
lin_pred = model.predict(X_test_scaled)

print(f"R² Score (Base Linear): {r2_score(y_test, lin_pred) * 100:.2f}%")
print(f"MSE (Base Linear): {mean_squared_error(y_test, lin_pred)}")
print(f"RMSE (Base Linear): {root_mean_squared_error(y_test, lin_pred)}")

#-------------------------------------------------------------------------------------
# LASSO REGRESSION

print("\n" + "="*60)
print("LASSO REGRESSION")
print("="*60)

alpha_range = [0.0001, 0.001, 0.01, 0.1, 1, 10, 100]

# Use LassoCV to find best alpha
lasso_cv = LassoCV(alphas=alpha_range, cv=5, random_state=42, max_iter=1000)
lasso_cv.fit(X_train_scaled, y_train)

print(f"Optimal Alpha: {lasso_cv.alpha_:.2f}")

# Train final model with best alpha
lasso = Lasso(alpha=lasso_cv.alpha_)
lasso.fit(X_train_scaled, y_train)

# Predict (use scaled data!)
lasso_pred = lasso.predict(X_test_scaled)  # ⚠️ Use X_test_scaled, not X_test

print(f"R² Score (Lasso): {r2_score(y_test, lasso_pred) * 100:.2f}%")
print(f"MSE (Lasso): {mean_squared_error(y_test, lasso_pred)}")
print(f"RMSE (Lasso): {root_mean_squared_error(y_test, lasso_pred)}")

# Check how many features Lasso kept vs. zeroed out
feature_importance = pd.DataFrame({
    'feature': X_train.columns,
    'coefficient': lasso.coef_
})
lasso_feature_importance = feature_importance[feature_importance['coefficient'] != 0].sort_values('coefficient', key=abs, ascending=False)

print(f"Features retained (Lasso): {len(lasso_feature_importance)} out of {len(lasso.coef_)}")
print(f"\nTop 10 most important features (Lasso):")
print(feature_importance.head(10))


#-------------------------------------------------------------------------------------
# RIDGE REGRESSION


print("\n" + "="*60)
print("RIDGE REGRESSION")
print("="*60)


# Test different alpha values for Ridge
alpha_values = [0.0001, 0.001, 0.01, 0.1, 1, 10, 50, 100, 200, 500]

# RidgeCV with cross-validation
ridge_cv = RidgeCV(alphas=alpha_values, scoring='r2', cv=5)
ridge_cv.fit(X_train_scaled, y_train)

print(f"Optimal Alpha: {ridge_cv.alpha_}")

# Final Ridge model
ridge = Ridge(alpha=ridge_cv.alpha_)
ridge.fit(X_train_scaled, y_train)
y_pred_ridge = ridge.predict(X_test_scaled)

ridge_r2 = r2_score(y_test, y_pred_ridge)
print(f"R² Score (Ridge): {r2_score(y_test, y_pred_ridge) * 100:.2f}%")
print(f"MSE (Ridge): {mean_squared_error(y_test, y_pred_ridge)}")
print(f"RMSE (Ridge): {root_mean_squared_error(y_test, y_pred_ridge)}")


# Feature importance for Ridge
ridge_importance = pd.DataFrame({
    'feature': X_train.columns,
    'coefficient': ridge.coef_
}).sort_values('coefficient', key=abs, ascending=False)

print(f"Features retained (Ridge): {len(ridge_importance)} out of {len(lasso.coef_)}")
print(f"\nTop 10 most important features (Ridge):")
print(ridge_importance.head(10))

# Compare coefficient magnitudes between Lasso and Ridge
print("\n" + "="*60)
print("COEFFICIENT COMPARISON: LASSO vs RIDGE")
print("="*60)

comparison = pd.DataFrame({
    'feature': X_train.columns,
    'Lasso': lasso.coef_,
    'Ridge': ridge.coef_
})
comparison['diff_pct'] = (comparison['Ridge'] - comparison['Lasso']) / comparison['Lasso'] * 100
print(comparison.round(2).to_string())