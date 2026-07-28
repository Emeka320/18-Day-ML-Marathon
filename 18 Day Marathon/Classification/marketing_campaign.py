import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.svm import SVC
import matplotlib.pyplot as plt
from itertools import combinations
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score, mean_squared_error, confusion_matrix, classification_report


#-------------------------------------------------------------------------
# Load Dataset

data = pd.read_csv('Data/marketing_campaign.csv', sep='\t')
print(f"Dataset: \n{data}")

backup_data = data.copy()

#-------------------------------------------------------------------------
# Inspection

print(data.columns)
print(data.dtypes)
print(data.isna().isnull().sum())


columns = ['ID']

data = data.drop(columns=columns)
data.insert(0, 'ID', range(1, len(data) + 1))
print(data)

print(f"Data Info: \n{data.info()}")

print(f"\nData Description: \n{data.describe()}")

#-------------------------------------------------------------------------
# Feature Engineering

one_hot_columns = ['Education', 'Marital_Status', 'Dt_Customer']

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


prefixes = ['Education', 'Marital_Status', 'Dt_Customer']

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

columnss = ['Education', 'Marital_Status', 'Dt_Customer']

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


#-------------------------------------------------------------------------------------
# Model 1: Channel Preference
# This model focuses on which customers prefer web vs store? Then a secondary question of which channel drives more sales?

print("\n" + "="*60)
print("MODEL 1: CHANNEL PREFERENCE...")
print("="*60)

#-------------------------------------------------------------------------------------
# Creating dataset copy
df = cleaned_data.copy()

# Create age from Year_Birth
from datetime import datetime
current_year = datetime.now().year
df['Age'] = current_year - df['Year_Birth']

# Create total spending column
spending_cols = ['MntWines', 'MntFruits', 'MntMeatProducts', 
                 'MntFishProducts', 'MntSweetProducts', 'MntGoldProds']
df['MntTotal'] = df[spending_cols].sum(axis=1)

# Create channel preference targets (single column)
# 0 = Web preferrer, 1 = Store preferrer, 2 = Balanced (ties with purchases)
df['ChannelPref'] = 2  # Default to balanced
df.loc[df['NumWebPurchases'] > df['NumStorePurchases'], 'ChannelPref'] = 0  # Web
df.loc[df['NumStorePurchases'] > df['NumWebPurchases'], 'ChannelPref'] = 1  # Store
# Ties with zero purchases remain 2, but exclude from analysis if needed

# Create separate flags for analysis
df['PrefersWeb'] = (df['ChannelPref'] == 0).astype(int)
df['PrefersStore'] = (df['ChannelPref'] == 1).astype(int)

# Create feature set
preference_data = df[[
    'Age', 'Income', 'Education', 'Marital_Status',
    'Kidhome', 'Teenhome', 'Recency', 'NumDealsPurchases', 
    'NumWebVisitsMonth', 'ChannelPref', 'MntTotal', 'NumCatalogPurchases'
]].copy()


print(f"\nData on Preferred Purchase Channel: \n{preference_data}")
print(f"\nShape: {preference_data.shape}")
print(preference_data.dtypes)


# Check class balance
print(f"\nChannel Preference Distribution:")
print(f"Web preferrers (0): {(df['ChannelPref'] == 0).sum()}")
print(f"Store preferrers (1): {(df['ChannelPref'] == 1).sum()}")
print(f"Balanced shoppers (2): {(df['ChannelPref'] == 2).sum()}")

# Which channel drives more sales?
web_avg = df[df['ChannelPref'] == 0]['MntTotal'].mean()
store_avg = df[df['ChannelPref'] == 1]['MntTotal'].mean()
balanced_avg = df[df['ChannelPref'] == 2]['MntTotal'].mean()

print(f"\nAvg spend - Web preferrers: ${web_avg:.2f}")
print(f"Avg spend - Store preferrers: ${store_avg:.2f}")
print(f"Avg spend - Balanced shoppers: ${balanced_avg:.2f}")

#-------------------------------------------------------------------------------------
# Features correlation
columns = list(preference_data.columns)
corr = preference_data.corr()
print(corr)

print("\n")

for one, two in combinations(columns, 2):
    correlation = preference_data[one].corr(preference_data[two])

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
plt.savefig('channel_preference_data_correlation_heatmap.png', dpi=150, bbox_inches='tight')
plt.show()
#-------------------------------------------------------------------------------------
# Features Selection and Scaling

X = preference_data[['Age', 'Income', 'Education', 'Marital_Status','Kidhome', 
    'Teenhome', 'Recency','NumDealsPurchases', 'NumCatalogPurchases', 'NumWebVisitsMonth', 'MntTotal']]

y = preference_data['ChannelPref']

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)


#-------------------------------------------------------------------------------------

print("\n" + "="*60)
print("GRADIENT BOOSTING - TUNING MAX_DEPTH")
print("="*60)

depth_values = [5, 10, 15, 20, 25, 30, None]
rf_depth_scores = []
rf_depth_cv_scores = []


for depth in depth_values:
    gbc_model = GradientBoostingClassifier(loss='log_loss', learning_rate=0.1, n_estimators=100, max_depth=depth, random_state=42)
    gbc_model.fit(X_train_scaled, y_train)
    y_pred = gbc_model.predict(X_test_scaled)
    accuracy = accuracy_score(y_test, y_pred)
    rf_depth_scores.append(accuracy)
    cv_score = cross_val_score(gbc_model, X_train_scaled, y_train, cv=5, scoring='accuracy').mean()
    rf_depth_cv_scores.append(cv_score)
    print(f"max_depth={depth}: Test Accuracy = {accuracy:.4f} | CV Accuracy = {cv_score:.4f}")

best_depth = depth_values[np.argmax(rf_depth_scores)]
best_cv_depth = depth_values[np.argmax(rf_depth_cv_scores)]
print(f"\nBest max_depth by Test Accuracy: {best_depth} (Score: {max(rf_depth_scores):.4f})")
print(f"Best max_depth by Cross-Validation: {best_cv_depth} (Score: {max(rf_depth_cv_scores):.4f})")

#-------------------------------------------------------------------------------------
# Model 2: High-Value Wine Buyer
# This model focuses on which customers are high-value wine buyers

print("\n" + "="*60)
print("MODEL 2: HIGH VALUE WINE BUYER...")
print("="*60)

# Top 25% wine spenders
threshold = df['MntWines'].quantile(0.75)
df['HighWineBuyer'] = (df['MntWines'] > threshold).astype(int)


wine_buy_data = df[[
    'Age', 'Income', 'Education', 'Marital_Status',
    'Kidhome', 'Teenhome', 'NumDealsPurchases', 'NumCatalogPurchases', 
    'NumWebVisitsMonth', 'HighWineBuyer'
]].copy()

print(wine_buy_data)

print(f"\nWine Buyer Distribution:")
print(f"Low Buyer (0): {(df['HighWineBuyer'] == 0).sum()}")
print(f"High Buyer (1): {(df['HighWineBuyer'] == 1).sum()}")

#-------------------------------------------------------------------------------------
# Features correlation

columns = list(wine_buy_data.columns)
wine_corr = wine_buy_data.corr()
print(wine_corr)

print("\n")

for one, two in combinations(columns, 2):
    correlation = wine_buy_data[one].corr(wine_buy_data[two])

    print(f"{one} vs {two} -> Correlation: {correlation}")

plt.figure(figsize=(14, 12))

# Create heatmap
sns.heatmap(wine_corr, 
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
plt.savefig('high-value_wine_buyer_data_correlation_heatmap.png', dpi=150, bbox_inches='tight')
plt.show()

#-------------------------------------------------------------------------------------
# Features Selection and Scaling


X = wine_buy_data[['Age', 'Income', 'Education', 'Marital_Status',
    'Kidhome', 'Teenhome', 'NumDealsPurchases', 'NumCatalogPurchases', 
    'NumWebVisitsMonth']]

y = wine_buy_data['HighWineBuyer']

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)


#-------------------------------------------------------------------------------------
# Model Training

print("\n" + "="*60)
print("GRADIENT BOOSTING - TUNING MAX_DEPTH")
print("="*60)

depth_values = [5, 10, 15, 20, 25, 30, None]
rf_depth_scores = []
rf_depth_cv_scores = []


for depth in depth_values:
    gbc_model = GradientBoostingClassifier(loss='log_loss', learning_rate=0.1, n_estimators=100, max_depth=depth, random_state=42)
    gbc_model.fit(X_train_scaled, y_train)
    y_pred = gbc_model.predict(X_test_scaled)
    accuracy = accuracy_score(y_test, y_pred)
    rf_depth_scores.append(accuracy)
    cv_score = cross_val_score(gbc_model, X_train_scaled, y_train, cv=5, scoring='accuracy').mean()
    rf_depth_cv_scores.append(cv_score)
    print(f"max_depth={depth}: Test Accuracy = {accuracy:.4f} | CV Accuracy = {cv_score:.4f}")

best_depth = depth_values[np.argmax(rf_depth_scores)]
best_cv_depth = depth_values[np.argmax(rf_depth_cv_scores)]
print(f"\nBest max_depth by Test Accuracy: {best_depth} (Score: {max(rf_depth_scores):.4f})")
print(f"Best max_depth by Cross-Validation: {best_cv_depth} (Score: {max(rf_depth_cv_scores):.4f})")



#-------------------------------------------------------------------------------------
# Model 3: Campaign Response (Experimental)

'''
# For campaign response, I'm thinking of using clustering to cluster customers into AcceptedCmp1 to 5, that way, i can cluster incoming customer's cause remember there won't be any data on which type of campaign they'd accept
camp_data = cleaned_data.copy()
prefixes = ['AcceptedCmp1', 'AcceptedCmp2', 'AcceptedCmp3', 'AcceptedCmp4', 'AcceptedCmp5']

for prefix in prefixes:

    cols = [c for c in camp_data.columns if c.startswith(prefix)]

    camp_data[prefix] = (
        camp_data[cols]
        .idxmax(axis=1)
        .str.replace(prefix, '', regex=False)
        .astype('category')
        .cat.codes
    )

    camp_data.drop(columns=cols, inplace=True)

print(camp_data)
'''






