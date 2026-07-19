import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from itertools import combinations
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler, OneHotEncoder, PolynomialFeatures
from sklearn.model_selection import train_test_split, cross_val_score, RandomizedSearchCV



#-------------------------------------------------------------------------------------
# Load datasets

data = pd.read_csv('Data/EV Energy Efficiency Dataset.csv')
data.insert(0, 'ID', range(1, len(data) + 1))
print("="*60)
print(f"EV EFFICIENCY DATASET: \n{data}")
print("="*60)

#-------------------------------------------------------------------------------------
# 

print(data.columns, "\n", data.dtypes)

one_hot_columns = ['Make', 'Model', 'Vehicle class']

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

prefixes = ['Make', 'Model', 'Vehicle class']

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

columnss = ['Make', 'Model', 'Vehicle class']

dropped_data = data.drop(columns=columnss)


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
# plt.savefig('correlation_heatmap.png', dpi=150, bbox_inches='tight')
plt.show()

#-------------------------------------------------------------------------------------
# Data Preparations

X = cleaned_data[['Model year', 'Motor (kW)', 'Recharge time (h)', 'Make', 'Model', 'Vehicle class']]
y = cleaned_data['Energy Efficiency (km/kWh)']


# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# ============================================================
# VISUALIZE NON-LINEAR RELATIONSHIPS
# ============================================================

fig, axes = plt.subplots(2, 2, figsize=(12, 10))

# 1. Motor (kW) vs Energy Efficiency
axes[0, 0].scatter(cleaned_data['Motor (kW)'], cleaned_data['Energy Efficiency (km/kWh)'], alpha=0.5, edgecolors='k', linewidth=0.3)
axes[0, 0].set_xlabel('Motor Power (kW)')
axes[0, 0].set_ylabel('Energy Efficiency (km/kWh)')
axes[0, 0].set_title('Motor Power vs Efficiency')
axes[0, 0].axhline(y=cleaned_data['Energy Efficiency (km/kWh)'].mean(), color='r', linestyle='--', alpha=0.5)

# 2. Recharge time (h) vs Energy Efficiency
axes[0, 1].scatter(cleaned_data['Recharge time (h)'], cleaned_data['Energy Efficiency (km/kWh)'], alpha=0.5, edgecolors='k', linewidth=0.3)
axes[0, 1].set_xlabel('Recharge Time (hours)')
axes[0, 1].set_ylabel('Energy Efficiency (km/kWh)')
axes[0, 1].set_title('Recharge Time vs Efficiency')

# 3. Model year vs Energy Efficiency
axes[1, 0].scatter(cleaned_data['Model year'], cleaned_data['Energy Efficiency (km/kWh)'], alpha=0.5, edgecolors='k', linewidth=0.3)
axes[1, 0].set_xlabel('Model Year')
axes[1, 0].set_ylabel('Energy Efficiency (km/kWh)')
axes[1, 0].set_title('Model Year vs Efficiency (Trend Over Time)')

# 4. Motor (kW) vs Recharge time (colored by efficiency)
scatter = axes[1, 1].scatter(cleaned_data['Motor (kW)'], cleaned_data['Recharge time (h)'], 
                              c=cleaned_data['Energy Efficiency (km/kWh)'], 
                              cmap='viridis', alpha=0.6, edgecolors='k', linewidth=0.3)
axes[1, 1].set_xlabel('Motor Power (kW)')
axes[1, 1].set_ylabel('Recharge Time (h)')
axes[1, 1].set_title('Motor vs Recharge (Color = Efficiency)')
plt.colorbar(scatter, ax=axes[1, 1], label='Efficiency (km/kWh)')

plt.tight_layout()
# plt.savefig('ev_nonlinear_relationships.png', dpi=150)
# plt.show()

#-------------------------------------------------------------------------------------
# Random Forest Regressor

print("\n" + "="*60)
print("RANDOM FOREST REGRESSOR")
print("="*60)


print("N_ESTIMATORS ITERATION...")
print("="*60)

n_range = [100, 150, 200, 250, 300, 350, 400, 450, 500]
forest_scores = []

for n in n_range:
    rf = RandomForestRegressor(n_estimators=n, max_depth=10, random_state=42)
    rf.fit(X_train, y_train)
    y_pred = rf.predict(X_test)
    r2 = r2_score(y_test, y_pred)
    forest_scores.append(r2)
    print(f"n_estimators={n}: R² = {r2 * 100:.2f}%")

# Best n
best_n = n_range[np.argmax(forest_scores)]
print(f"\nBest n_estimators: {best_n} (R²: {max(forest_scores) * 100:.2f}%)")

# Train final model with best n
best_rf = RandomForestRegressor(n_estimators=best_n, max_depth=10, random_state=42)
best_rf.fit(X_train, y_train)
y_pred_best_n = best_rf.predict(X_test)

r2_order1 = r2_score(y_test, y_pred_best_n)

print(f"\nBest n_estimator Model R²: {r2_order1 * 100:.2f}%")


print("\n" + "="*60)
print("MAX_DEPTH ITERATION...")
print("="*60)

depth_range = range(5, 15)
depth_scores = []

for d in depth_range:
    depth_rf = RandomForestRegressor(n_estimators=best_n, max_depth=d, random_state=42)
    depth_rf.fit(X_train, y_train)
    depth_pred = depth_rf.predict(X_test)
    r2 = r2_score(y_test, depth_pred)
    depth_scores.append(r2)
    print(f"max_depth={d}: R² = {r2 * 100:.2f}%")

# Best depth
best_d = depth_range[np.argmax(depth_scores)]
print(f"\nBest max_depth: {best_d} (R²: {max(depth_scores) * 100:.2f}%)")

# Train final model with best n
final_rf = RandomForestRegressor(n_estimators=best_n, max_depth=best_d, random_state=42)
final_rf.fit(X_train, y_train)
y_pred_final = final_rf.predict(X_test)

print(f"\nFinal Model R²: {r2_score(y_test, y_pred_final) * 100:.2f}%")
