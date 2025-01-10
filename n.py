import time
import numpy as np
import matplotlib.pyplot as plt
from pyclustering.cluster.kmedoids import kmedoids
from scipy.spatial.distance import cdist

# Set random seed for reproducibility
np.random.seed(42)

# Generate random data
data = np.random.rand(10000, 2) * 100

# Subsample data for initial clustering (e.g., use 2000 points)
subsample_size = 2000
indices = np.random.choice(data.shape[0], subsample_size, replace=False)
data_subsample = data[indices]

# Initialize K-Medoids model with initial medoids from the subsample
initial_medoids = np.random.choice(range(subsample_size), size=5, replace=False).tolist()

# Create K-Medoids instance
kmedoids_instance = kmedoids(data_subsample.tolist(), initial_medoids)

# Measure the time for fitting the model
t0 = time.process_time()
kmedoids_instance.process()
t1 = time.process_time()

# Calculate and print the time taken
tt = t1 - t0
print("Total Time for subsample:", tt)

# Get the medoid centers and cluster labels for the subsample
centers = kmedoids_instance.get_medoids()
labels_subsample = kmedoids_instance.get_clusters()

# Prepare cluster labels for the full dataset
full_labels = np.empty(data.shape[0], dtype=int)
for cluster_id, cluster in enumerate(labels_subsample):
    for index in cluster:
        full_labels[indices[index]] = cluster_id

# Assign the remaining points to the nearest medoid
medoid_points = data_subsample[centers]
distances = cdist(data, medoid_points, metric='euclidean')
full_labels[~np.isin(np.arange(data.shape[0]), indices)] = np.argmin(distances[~np.isin(np.arange(data.shape[0]), indices)], axis=1)

# Define colors for each cluster using the updated method
num_clusters = len(labels_subsample)
colors = plt.colormaps['hsv'](np.linspace(0, 1, num_clusters))

# Plot each data point with the corresponding color
for i in range(len(data)):
    plt.scatter(data[i][0], data[i][1], color=colors[full_labels[i]], s=10, alpha=0.5)

# Plot the medoid centers
plt.scatter(data_subsample[centers, 0], data_subsample[centers, 1], marker="o", s=100, linewidths=2, color='black', label='Medoids')

# Add title and legend
plt.title("K-Medoids Clustering")
plt.xlabel("Feature 1")
plt.ylabel("Feature 2")
plt.legend()
plt.grid()

# Save the plot as an image file
plt.savefig("kmedoids_plot_optimized.png")

# Show the plot
plt.show()
