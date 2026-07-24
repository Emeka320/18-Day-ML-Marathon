import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from itertools import combinations
from scipy.cluster.hierarchy import dendrogram, linkage
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.preprocessing import OneHotEncoder, LabelEncoder, StandardScaler
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score

#-----------------------------------------------------------------------------------------------------------------------------------

data = pd.read_csv("Data/Customer Segmentation.csv")

backup_data = data.copy()

columns = ['Unnamed: 0', 'Segmentation']

data = data.drop(columns=columns)
print(data)

# Feature Engineering

label_columns = ['Ever_Married', 'Graduated']
one_hot_columns = ['Profession', 'Spending_Score', 'Var_1']

#-------------------------------------------------------------------------

label = LabelEncoder()

for l_col in label_columns:
    data[l_col] = label.fit_transform(data[l_col])

# Convert numerical float columns properly
data['Work_Experience'] = data['Work_Experience'].astype('Int64')
data['Family_Size'] = data['Family_Size'].astype('Int64')

#--------------------------------------------------------------------------

one_hot = OneHotEncoder(drop='first', sparse_output=False)

# Replace gender column with encoded values
data['Gender'] = one_hot.fit_transform(data[['Gender']]).astype(int)

#--------------------------------------------------------------------------
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

prefixes = ['Profession', 'Spending_Score', 'Var_1']

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

columnss = ['Profession', 'Spending_Score', 'Var_1']
dropped_data = data.drop(columns=columnss)


full_encoded = pd.merge(dropped_data, encoded_df, on="ID")
drop = ['ID']
full_encoded = full_encoded.drop(columns=drop)
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

corr = cleaned_data.corr()
print(corr)

for one, two in combinations(columns, 2):
    correlation = cleaned_data[one].corr(cleaned_data[two])

    print(f"{one} vs {two} -> Correlation: {correlation}")

#----------------------------------------------------------------------------

scaler = StandardScaler()

scaled_data = scaler.fit_transform(cleaned_data)



# Sample for dendrogram (2000 samples max for readability)
sample_size = min(2000, len(scaled_data))
np.random.seed(42)
sample_idx = np.random.choice(len(scaled_data), sample_size, replace=False)
sample_data = scaled_data[sample_idx]

# Generate linkage matrix
linkage_matrix = linkage(sample_data, method='ward')

# Plot dendrogram
plt.figure(figsize=(14, 8))
dendrogram(
    linkage_matrix,
    truncate_mode='lastp',
    p=30,
    leaf_rotation=90.,
    leaf_font_size=10.,
    show_contracted=True
)
plt.title('Hierarchical Clustering Dendrogram (Ward Linkage)', fontsize=14)
plt.xlabel('Cluster Index', fontsize=12)
plt.ylabel('Distance', fontsize=12)
plt.tight_layout()
plt.show()



#-----------------------------------------------------------------------------------------------------------------------------------

# Function to evaluate clustering metrics
def evaluate_clustering(labels, data_scaled, algorithm_name):
    """Calculate multiple clustering metrics"""
    if len(set(labels)) > 1:  # Need at least 2 clusters for metrics
        sil_score = silhouette_score(data_scaled, labels)
        db_score = davies_bouldin_score(data_scaled, labels)
        ch_score = calinski_harabasz_score(data_scaled, labels)
        
        print(f"\n{'='*50}")
        print(f"{algorithm_name} Results:")
        print(f"{'='*50}")
        print(f"Number of clusters: {len(set(labels))}")
        print(f"Silhouette Score: {sil_score:.4f} (higher = better, range: -1 to 1)")
        print(f"Davies-Bouldin Score: {db_score:.4f} (lower = better)")
        print(f"Calinski-Harabasz Score: {ch_score:.2f} (higher = better)")
        
        # Count samples per cluster
        unique, counts = np.unique(labels, return_counts=True)
        print(f"\nCluster sizes:")
        for cluster, count in zip(unique, counts):
            print(f"  Cluster {cluster}: {count} samples ({count/len(labels)*100:.1f}%)")
        
        return sil_score, db_score, ch_score
    else:
        print(f"\n{algorithm_name}: Only {len(set(labels))} cluster found. Cannot compute metrics.")
        return None, None, None

#-----------------------------------------------------------------------------------------------------------------------------------
# 1. K-Means Clustering (try different k values)

print("\n" + "="*60)
print("K-MEANS CLUSTERING")
print("="*60)

k_values = range(2, 11)
kmeans_scores = []

for k in k_values:
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels = kmeans.fit_predict(scaled_data)
    sil_score = silhouette_score(scaled_data, labels)
    kmeans_scores.append(sil_score)
    print(f"k={k}: Silhouette Score = {sil_score:.4f}")

# Best k
best_k = k_values[np.argmax(kmeans_scores)]
print(f"\nBest k for K-Means: {best_k} (Silhouette Score: {max(kmeans_scores):.4f})")

# Run K-Means with best k
final_kmeans = KMeans(n_clusters=best_k, random_state=42, n_init=10)
kmeans_labels = final_kmeans.fit_predict(scaled_data)
evaluate_clustering(kmeans_labels, scaled_data, f"K-Means (k={best_k})")

#-----------------------------------------------------------------------------------------------------------------------------------
# 2. Agglomerative (Hierarchical) Clustering
print("\n" + "="*60)
print("AGGLOMERATIVE CLUSTERING")
print("="*60)

for c in [3, 4, 5, 6, 7, 8]:
    hier = AgglomerativeClustering(n_clusters=c, linkage='ward')
    hier_labels = hier.fit_predict(scaled_data)
    sil_score = silhouette_score(scaled_data, hier_labels)
    print(f"c={c}: Silhouette Score = {sil_score:.4f}")

# Best k from above
best_c_hier = 4  # Based on typical results, adjust as needed
hier = AgglomerativeClustering(n_clusters=best_c_hier)
hier_labels = hier.fit_predict(scaled_data)
evaluate_clustering(hier_labels, scaled_data, f"Agglomerative (k={best_c_hier})")

#-----------------------------------------------------------------------------------------------------------------------------------
# 3. DBSCAN (density-based)
print("\n" + "="*60)
print("DBSCAN CLUSTERING")
print("="*60)

# Try different eps values
eps_values = [0.5, 1.0, 1.5, 2.0, 2.5, 3.0]
min_samples_values = [5, 10, 15, 20]

best_dbscan_score = -1
best_params = None

for eps in eps_values:
    for min_samples in min_samples_values:
        dbscan = DBSCAN(eps=eps, min_samples=min_samples)
        dbscan_labels = dbscan.fit_predict(scaled_data)
        
        n_clusters = len(set(dbscan_labels)) - (1 if -1 in dbscan_labels else 0)
        n_noise = list(dbscan_labels).count(-1)
        
        if n_clusters >= 2:
            sil_score = silhouette_score(scaled_data[dbscan_labels != -1], 
                                        dbscan_labels[dbscan_labels != -1])
            print(f"eps={eps}, min_samples={min_samples}: {n_clusters} clusters, {n_noise} noise, Silhouette={sil_score:.4f}")
            
            if sil_score > best_dbscan_score:
                best_dbscan_score = sil_score
                best_params = (eps, min_samples)
        else:
            print(f"eps={eps}, min_samples={min_samples}: Only {n_clusters} cluster(s), {n_noise} noise")

if best_params:
    print(f"\nBest DBSCAN parameters: eps={best_params[0]}, min_samples={best_params[1]}")
    dbscan_final = DBSCAN(eps=best_params[0], min_samples=best_params[1])
    dbscan_labels = dbscan_final.fit_predict(scaled_data)
    evaluate_clustering(dbscan_labels, scaled_data, f"DBSCAN (eps={best_params[0]}, min_samples={best_params[1]})")
else:
    print("No valid DBSCAN clustering found with 2+ clusters")

#-----------------------------------------------------------------------------------------------------------------------------------
# 4. Compare all algorithms
print("\n" + "="*60)
print("FINAL COMPARISON")
print("="*60)

print("\nRecommendation based on metrics:")
print("- Silhouette Score > 0.5: Good clustering")
print("- Silhouette Score > 0.7: Strong clustering")
print("- Silhouette Score < 0.2: Poor clustering (data may not have natural clusters)")

#-----------------------------------------------------------------------------------------------------------------------------------
# Optional: Visualize cluster distributions (if you want to add this)

"""Add best clustering results back to original dataframe"""
data_with_clusters = backup_data.copy()

kmeans_labels = pd.DataFrame(kmeans_labels)
hier_labels = pd.DataFrame(hier_labels)
dbscan_labels = pd.DataFrame(dbscan_labels)


data_with_clusters.insert(0, 'KMeans_Cluster', kmeans_labels)
data_with_clusters.insert(1, 'Hierarchical_Cluster', hier_labels)
data_with_clusters.insert(2, 'DBSCAN_Cluster', dbscan_labels)




# Uncomment to save results
final_data = data_with_clusters
print(final_data)
# final_data.to_csv("Customer_Segmentation_Clustered.csv", index=False)
# print("\nClustered data saved to 'Customer_Segmentation_Clustered.csv'")




