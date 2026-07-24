import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from itertools import combinations
from scipy.cluster.hierarchy import dendrogram, linkage
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.preprocessing import OneHotEncoder, LabelEncoder, StandardScaler
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score

#-------------------------------------------------------------------------
# Load Dataset
data = pd.read_csv('Data/fraud_data.csv')
print(f"Dataset: \n{data}")
backup_data = data.copy()

#-------------------------------------------------------------------------
# Inspection
print(data.columns)
print(data.dtypes)
print(data.isna().isnull().sum())


data.insert(0, 'ID', range(1, len(data) + 1))
print(data)

data_counts = data['is_fraud'].value_counts()
print(data_counts)

#-------------------------------------------------------------------------

train_data = data[['ID', 'merchant', 'category', 'amt', 'city',
       'lat', 'long', 'city_pop', 'merch_lat', 'merch_long', 'is_fraud']].copy()

print(train_data)

#-------------------------------------------------------------------------

one_hot_columns = ['merchant', 'category', 'city', 'is_fraud']

one_hot = OneHotEncoder(drop='first', sparse_output=False)

#--------------------------------------------------------------------------
def encode(df, one_hot_columns):
    encoded_array = one_hot.fit_transform(df[one_hot_columns])

    encoded_df = pd.DataFrame(
        encoded_array,
        columns=one_hot.get_feature_names_out(one_hot_columns),
        index=df.index
    )

    return encoded_df

encoded_df = encode(train_data, one_hot_columns)
id = data['ID']
encoded_df.insert(0, 'ID', id)

prefixes = ['merchant', 'category', 'city', 'is_fraud']

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

#--------------------------------------------------------------------------

columnss = ['merchant', 'category', 'city', 'is_fraud']
dropped_data = train_data.drop(columns=columnss)


full_encoded = pd.merge(dropped_data, encoded_df, on="ID")

print(f"Fully Encoded Data: \n{full_encoded}")

#----------------------------------------------------------------------------

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

columns_type = cleaned_data.dtypes
print(columns_type)

corr = cleaned_data.corr()
print(corr)

for one, two in combinations(columns, 2):
    correlation = cleaned_data[one].corr(cleaned_data[two])

    print(f"{one} vs {two} -> Correlation: {correlation}")

#----------------------------------------------------------------------------

new_data = cleaned_data[['amt', 'city_pop', 'merch_lat', 'merch_long', 'merchant', 'category', 'city']]
print(new_data)
print(len(new_data))

print(new_data.isna().isnull().sum())


scaler = StandardScaler()
scaled_data = scaler.fit_transform(new_data)

print(f"Scaled Data Length: {len(scaled_data)}")

#----------------------------------------------------------------------------
# 1. K-Means Clustering

k = 2

Kmeans = KMeans(n_clusters=k, init='k-means++', algorithm='lloyd', random_state=42)
k_labels = Kmeans.fit_predict(scaled_data)
silo_score = silhouette_score(scaled_data, k_labels)

print(f"k={k}: Silhouette Score = {silo_score:.4f}")


print(k_labels)
print(len(k_labels))


"""Add best clustering results back to original dataframe"""
data_with_clusters = cleaned_data.copy()
print(len(data_with_clusters))

kmeans_labels = pd.DataFrame(k_labels)
print(kmeans_labels)

data_with_clusters.insert(0, 'KMeans_Cluster', kmeans_labels)
print(data_with_clusters)

#----------------------------------------------------------------------------
# Cluster Validtion

# 1. Calculate the average fraud rate for each cluster
cluster_0_fraud_rate = data_with_clusters[data_with_clusters['KMeans_Cluster'] == 0]['is_fraud'].mean()
cluster_1_fraud_rate = data_with_clusters[data_with_clusters['KMeans_Cluster'] == 1]['is_fraud'].mean()

print(f"Cluster 0 Fraud Rate: {cluster_0_fraud_rate:.2f}")
print(f"Cluster 1 Fraud Rate: {cluster_1_fraud_rate:.2f}")

# 2. Determine which cluster corresponds to "Fraud" (1) and which to "No Fraud" (0)
if cluster_1_fraud_rate > cluster_0_fraud_rate:
    # Cluster 1 is the Fraud cluster
    aligned_predictions = np.where(data_with_clusters['KMeans_Cluster'] == 1, 1, 0)
else:
    # Cluster 0 is the Fraud cluster
    aligned_predictions = np.where(data_with_clusters['KMeans_Cluster'] == 0, 1, 0)

# 3. Calculate accuracy with aligned labels
matches = (aligned_predictions == data_with_clusters['is_fraud']).sum()
pct_match = (aligned_predictions == data_with_clusters['is_fraud']).mean() * 100

print(f"Number of matching predictions: {matches} out of {len(data_with_clusters)} rows")
print(f"Percentage of matching predictions: {pct_match:.2f}%")
