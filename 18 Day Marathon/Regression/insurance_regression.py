import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from itertools import combinations
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import OneHotEncoder, LabelEncoder, StandardScaler


# ----------------------------------------------------------------------
# Section 1: Preprocess Car Insurance data to include claims probability

print("\n" + "="*60)
print("SECTION 1: PREPROCESS CAR INSURANCE DATA TO INCLUDE CLAIMS PROBABILITY")
print("="*60)

# Load the dataset
df = pd.read_csv('Data/car_insurance.csv')

# ----------------------------------------------------------------------
# 1. Define base risk scores for each feature
# ----------------------------------------------------------------------

# Age risk (higher age = lower risk)
age_risk = {
    0: 0.85,   # 0-9y (youngest = highest risk)
    1: 0.70,   # 10-19y
    2: 0.50,   # 20-29y
    3: 0.30    # 30y+ (oldest = lowest risk)
}
df['age_risk'] = df['age'].map(age_risk)

# Gender risk (empirical: males often slightly higher risk)
gender_risk = {0: 0.45, 1: 0.55}
df['gender_risk'] = df['gender'].map(gender_risk)

# Driving experience risk (more experience = lower risk)
experience_risk = {
    '0-9y': 0.80,
    '10-19y': 0.55,
    '20-29y': 0.35,
    '30y+': 0.20
}
df['experience_risk'] = df['driving_experience'].map(experience_risk)

# Education risk (higher education = lower risk)
education_risk = {
    'none': 0.75,
    'high school': 0.60,
    'university': 0.40
}
df['education_risk'] = df['education'].map(education_risk)

# Income risk (higher income = lower risk? Or higher claims? Let's use it as a modifier)
income_risk = {
    'poverty': 0.70,
    'working class': 0.60,
    'middle class': 0.50,
    'upper class': 0.40
}
df['income_risk'] = df['income'].map(income_risk)

# Credit score risk (higher credit score = lower risk)
# Already a float between 0-1, but lower is riskier
# We'll invert it: 1 - credit_score gives risk
df['credit_risk'] = 1 - df['credit_score'].fillna(df['credit_score'].mean())

# Vehicle ownership (owning = lower risk? Or higher because they drive more?)
# We'll keep it neutral and use it as a small modifier
df['ownership_risk'] = df['vehicle_ownership'] * 0.05  # slight increase for ownership

# Vehicle year (newer = lower risk)
df['vehicle_year_risk'] = df['vehicle_year'].map({'before 2015': 0.60, 'after 2015': 0.40})

# Married (married = lower risk)
df['married_risk'] = df['married'] * 0.05  # slight decrease for married

# Children (more children = higher risk? Or lower? We'll make it neutral)
df['children_risk'] = df['children'] * 0.02

# Speeding violations (more = higher risk)
df['speeding_risk'] = np.clip(df['speeding_violations'] * 0.10, 0, 0.50)

# DUIs (more = much higher risk)
df['dui_risk'] = np.clip(df['duis'] * 0.15, 0, 0.60)

# Past accidents (more = much higher risk)
df['accident_risk'] = np.clip(df['past_accidents'] * 0.12, 0, 0.50)

# Annual mileage (more miles = higher risk)
# Normalize by assuming 10000 miles is average
df['mileage_risk'] = (df['annual_mileage'].fillna(df['annual_mileage'].mean()) / 10000) * 0.15

# ----------------------------------------------------------------------
# 2. Calculate base probability (sum of weighted risks)
# ----------------------------------------------------------------------

# Define weights for each risk factor (can be tuned)
weights = {
    'age_risk': 0.20,
    'gender_risk': 0.05,
    'experience_risk': 0.20,
    'education_risk': 0.10,
    'income_risk': 0.05,
    'credit_risk': 0.15,
    'ownership_risk': 0.02,
    'vehicle_year_risk': 0.05,
    'married_risk': 0.02,
    'children_risk': 0.02,
    'speeding_risk': 0.05,
    'dui_risk': 0.05,
    'accident_risk': 0.05,
    'mileage_risk': 0.02
}

# Calculate weighted sum
df['claims_probability'] = 0
for feature, weight in weights.items():
    df['claims_probability'] += df[feature] * weight

# Scale to 0-1 range
df['claims_probability'] = np.clip(df['claims_probability'], 0, 1)

# Add some random noise to make it more realistic (keep within 0-1)
np.random.seed(42)
noise = np.random.normal(0, 0.03, len(df))
df['claims_probability'] = np.clip(df['claims_probability'] + noise, 0, 1)

# ----------------------------------------------------------------------
# 3. GENERATE OUTCOME BASED ON PROBABILITY (NON-LINEAR, PROBABILISTIC)
# ----------------------------------------------------------------------

# For each customer, the probability is used as the chance of claiming
# Even with 20% probability, they CAN still claim—it's just less likely
np.random.seed(42)
df['outcome'] = np.random.binomial(1, df['claims_probability'])

# ----------------------------------------------------------------------
# 4. Convert to percentage scale and round to two decimal places
# ----------------------------------------------------------------------

df['claims_probability'] = (df['claims_probability'] * 100).round(2)

# ----------------------------------------------------------------------
# 5. Verify the relationship
# ----------------------------------------------------------------------

print("Claims Probability Distribution:")
print(df['claims_probability'].describe())
print("\n")
print(f"Mean: {df['claims_probability'].mean():.4f}")
print(f"Std: {df['claims_probability'].std():.4f}")
print(f"Min: {df['claims_probability'].min():.4f}")
print(f"Max: {df['claims_probability'].max():.4f}")

# Check correlation with actual outcome (should be moderate, not perfect)
corr = df['claims_probability'].corr(df['outcome'])
print(f"\nCorrelation with outcome: {corr:.4f}")

# Check relationship between probability and outcome by binning
df['prob_bin'] = pd.cut(df['claims_probability'], bins=10, labels=False)
outcome_by_bin = df.groupby('prob_bin')['outcome'].mean()
print("\nOutcome rate by probability bin:")
print(outcome_by_bin)

# Check if there are customers with low probability who still claimed
low_prob_claimers = df[(df['claims_probability'] < 0.20) & (df['outcome'] == 1)]
print(f"\nCustomers with probability < 20% who still claimed: {len(low_prob_claimers)}")

# ----------------------------------------------------------------------
# 6. Create the final dataset
# ----------------------------------------------------------------------

# Keep the original features + the new probability column + the outcome
# Drop intermediate risk columns
risk_cols = [col for col in df.columns if col.endswith('_risk')]
final_df = df.drop(columns=risk_cols)

# Ensure outcome is the last column
final_df = final_df[['id', 'age', 'gender', 'driving_experience', 'education', 'income', 
                     'credit_score', 'vehicle_ownership', 'vehicle_year', 'married', 
                     'children', 'postal_code', 'annual_mileage', 'vehicle_type', 
                     'speeding_violations', 'duis', 'past_accidents', 
                     'claims_probability', 'outcome']]

print(f"\nFinal dataset shape: {final_df.shape}")
print("Columns:", final_df.columns.tolist())

# Save the dataset
final_df.to_csv('Data/insurance_data_with_probability.csv', index=False)




# ----------------------------------------------------------------------
# Section 2: Use preprocessed Data for modelling

print("\n" + "="*60)
print("SECTION 2: USE PREPROCESSED DATA FOR MODELLING")
print("="*60)

# Data Reading

data = pd.read_csv('Data/insurance_data_with_probability.csv')
print(f"Car Insurance Data (with probability): \n{data}")

#----------------------------------------------------------------------------
# Data Inspection

print(f"Null columns: \n{data.isna().isnull().sum()}")
print(f"Data Columns: \n{data.columns}")
print(f"Columns types: \n{data.dtypes}")

pred_balance = data['outcome'].value_counts()
print("\n")
print(pred_balance)

string = data.select_dtypes(include='str')
string = pd.DataFrame(data=string)
string_columns = string.columns
print(string_columns)

#----------------------------------------------------------------------------
# Features Encoding

one_hot = OneHotEncoder(drop='first', sparse_output=False)
# Replace Vehcile type column with encoded values
data['vehicle_type'] = one_hot.fit_transform(data[['vehicle_type']]).astype(int)
data['vehicle_year'] = one_hot.fit_transform(data[['vehicle_year']]).astype(int)

def label_encode(df):
    label = LabelEncoder()
    df['driving_experience'] = label.fit_transform(df['driving_experience'])
    df['education'] = label.fit_transform(df['education'])
    df['income'] = label.fit_transform(df['income'])
    return df


print(f"\nData after One Hot Encode: \n{data}")
label_encode_data = label_encode(data)
print(f"\nData after Label Encode: \n{label_encode_data}")


cleaned_data = label_encode_data.copy()
cleaned_data = cleaned_data.dropna()
cleaned_data = cleaned_data.drop(columns='id')
print(f"\nFully Cleaned Data: \n{cleaned_data}")

print(cleaned_data.dtypes)

#----------------------------------------------------------------------------
# Feature Selection

claims_corr = cleaned_data.corr()

claims_column = cleaned_data.columns
print(claims_column)

for one, two in combinations(claims_column, 2):
    correlation = cleaned_data[one].corr(cleaned_data[two])
    print(f"{one} vs {two} -> Correlation: {correlation}")

#----------------------------------------------------------------------------
# Feature Selection - (Features for regression)

X_reg = cleaned_data[['age', 'gender', 'driving_experience', 'education', 'income', 
            'credit_score', 'vehicle_ownership', 'vehicle_year', 'married', 'children', 
            'annual_mileage', 'speeding_violations', 'duis', 'past_accidents']]

y_reg = cleaned_data['claims_probability']  # Continuous target (0-100)

# Split data
X_train, X_test, y_train, y_test = train_test_split(X_reg, y_reg, test_size=0.3, random_state=42)

# Scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

#----------------------------------------------------------------------------
# Model Training: RANDOM FOREST - TUNING N_ESTIMATORS
print("\n" + "="*60)
print("RANDOM FOREST - TUNING N_ESTIMATORS")
print("="*60)

n_values = [20, 50, 100, 150, 200, 250]
r2_scores = []
r2_cv_scores = []
mse_scores = []
mse_cv_scores = []

for n in n_values:
    rf = RandomForestRegressor(n_estimators=n, random_state=42, n_jobs=-1)
    rf.fit(X_train_scaled, y_train)
    y_pred = rf.predict(X_test_scaled)
    accuracy = r2_score(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    r2_scores.append(accuracy)
    mse_scores.append(mse)

    cv_score = cross_val_score(rf, X_train_scaled, y_train, cv=5, scoring='r2').mean()
    r2_cv_scores.append(cv_score)

    cv_score_mse = cross_val_score(rf, X_train_scaled, y_train, cv=5, scoring='neg_mean_squared_error').mean()
    mse_cv_scores.append(cv_score_mse)

    print(f"n_estimators={n}: R2 Score = {accuracy:.4f} | CV (R2) = {cv_score:.4f} | CV (MSE) = {cv_score_mse:.2f}")

best_n = n_values[np.argmax(r2_scores)]
best_cv_n = n_values[np.argmax(r2_cv_scores)]
best_cv_mse_n = n_values[np.argmax(mse_cv_scores)]

print(f"\nBest n_estimators by R2 Score: {best_n} (Score: {max(r2_scores):.4f})")
print(f"Best n_estimators by Cross-Validation (R2): {best_cv_n} (Score: {max(r2_cv_scores):.4f})")
print(f"Best n_estimators by Cross-Validation (MSE): {best_cv_mse_n} (Score: {max(mse_cv_scores):.4f})")



# ----------------------------------------------------------------------
# Section 3: Calibration Curve

print("\n" + "="*60)
print("SECTION 3: CALIBRATION CURVE")
print("="*60)

# Predict probabilities
y_pred_proba = rf.predict(X_test)

# Create bins for predicted probabilities
bins = np.linspace(0, 100, 11)  # 10 bins from 0 to 100
bin_indices = np.digitize(y_pred_proba, bins)

# Calculate mean predicted and actual for each bin
bin_pred_means = []
bin_actual_means = []

for i in range(1, len(bins)):
    mask = bin_indices == i
    if np.sum(mask) > 0:
        bin_pred_means.append(np.mean(y_pred_proba[mask]))
        bin_actual_means.append(np.mean(y_test[mask]))

# Plot calibration curve
plt.figure(figsize=(8, 6))
plt.plot(bin_pred_means, bin_actual_means, marker='o', linestyle='-', label='Model')
plt.plot([0, 100], [0, 100], 'k--', label='Perfect Calibration')
plt.xlabel('Predicted Probability (%)')
plt.ylabel('Actual Probability (%)')
plt.title('Calibration Curve: Regression Model')
plt.legend()
plt.grid(True)
plt.show()

# Also calculate calibration metrics
from sklearn.metrics import mean_absolute_error

mae = mean_absolute_error(y_test, y_pred_proba)
print(f"Output from calibration curve: \n")
print(f"Mean Absolute Error: {mae:.4f} percentage points")
print(f"RMSE: {np.sqrt(mean_squared_error(y_test, y_pred_proba)):.4f} percentage points")

