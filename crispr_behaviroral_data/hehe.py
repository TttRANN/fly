
import numpy as np
import pandas as pd
from sklearn.decomposition import FastICA
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import seaborn as sns


import numpy as np
import pandas as pd
from sklearn.decomposition import FastICA
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import seaborn as sns

import numpy as np
import pandas as pd
from sklearn.decomposition import FastICA
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import seaborn as sns

def iterative_ica_clustering(data, max_iterations=10, n_clusters=2, kmeans_max_iter=500):
    iteration = 0
    # Ensure parent_labels starts as a NumPy array to use Boolean indexing directly
    data_to_analyze = [(data, np.arange(len(data)), np.array(['Root'] * len(data)))]
    clusters_history = {i: [] for i in range(max_iterations)}
    
    while data_to_analyze and iteration < max_iterations:
        new_data_to_analyze = []
        iteration_clusters = []

        for subset, indices, parent_labels in data_to_analyze:
            if len(subset) < 2:
                continue  # Skip if fewer than 2 samples, cannot cluster further
            
            # Perform ICA
            ica = FastICA(n_components=2, random_state=42)
            ica_components = ica.fit_transform(subset)
            ica_df = pd.DataFrame(ica_components, columns=['IC1', 'IC2'])
            
            # Perform clustering
            kmeans = KMeans(n_clusters=n_clusters, random_state=42, max_iter=kmeans_max_iter)
            labels = kmeans.fit_predict(ica_df)

            # Calculate distances to cluster centers for split decision
            distances = np.linalg.norm(ica_df - kmeans.cluster_centers_[labels], axis=1)
            
            for cluster_id in range(n_clusters):
                cluster_mask = labels == cluster_id
                if cluster_mask.any():  # Check if any element is True in the mask
                    cluster_distances = distances[cluster_mask]
                    cluster_indices = indices[cluster_mask]
                    cluster_subset = subset[cluster_mask]
                    cluster_labels = parent_labels[cluster_mask]  # Direct use without error

                    # Update labels for new clusters
                    cluster_labels = np.array([f"{pl}-sub{cluster_id}" for pl in cluster_labels])

                    # Determine if this cluster should be split further by checking the variance of distances
                    if np.var(cluster_distances) > 0.2 and len(cluster_subset) > 2:  # Threshold variance and minimum size
                        new_data_to_analyze.append((cluster_subset, cluster_indices, cluster_labels))

            # Store the history of indices clustered in this iteration
            iteration_clusters.append((indices, labels))

        # Update data for the next iteration only if new splits are found
        if new_data_to_analyze:
            data_to_analyze = new_data_to_analyze
        else:
            print(f"Stopping at Iteration {iteration}: No further splits warranted.")
            break
                # Save indices to CSV file
        for idx, (indices, labels) in enumerate(iteration_clusters):
            df = pd.DataFrame({'Original_Index': indices, 'Cluster_Label': labels})
            df.to_csv(f'clusters_iteration_{iteration}_cluster_{idx}.csv', index=False)
        
        # Visualize the clustering result of this iteration
        plt.figure(figsize=(12, 6))
        plt.subplot(1, 2, 1)
        sns.scatterplot(x='IC1', y='IC2', hue=parent_labels, palette='viridis', data=ica_df, style=labels, markers=['o', 's', 'D'], alpha=0.6)
        plt.title(f'ICA Components with Clusters - Iteration {iteration}')
        plt.xlabel('IC1')
        plt.ylabel('IC2')
        plt.grid(True)

        plt.subplot(1, 2, 2)
        sns.kdeplot(x='IC1', y='IC2', hue=parent_labels, fill=True, common_norm=False, palette='viridis', data=ica_df)
        plt.title(f'Density Plot of IC1 vs IC2 - Iteration {iteration}')
        plt.xlabel('IC1')
        plt.ylabel('IC2')
        plt.grid(True)
        plt.tight_layout()
        plt.show()

        # Move to the next iteration
        clusters_history[iteration] = iteration_clusters
        iteration += 1

    return clusters_history



# Example usage with dummy data
file_path = '/Users/tairan/Documents/bbba.csv'
data =  pd.read_csv(file_path,header=None)

threshold = 0.001
data = data.loc[:, data.isnull().mean() < threshold]
# # #Transpose the data if needed, to ensure more rows than columns
if data.shape[0] < data.shape[1]:
      data = data.T

# # # Convert data to numpy array

data = data.to_numpy()

# Perform iterative ICA clustering
n_clusters = 2
clusters_history = iterative_ica_clustering(data, max_iterations=4, n_clusters=2, kmeans_max_iter=100)

# Plot clusters from each iteration using original features
# plot_clusters_from_iterations(clusters_history)
