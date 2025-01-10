import time
import numpy as np
import matplotlib.pyplot as plt
from pyclustering.cluster.kmedoids import kmedoids

# Generate random data
data = np.random.rand(10000, 2) * 100

# Initialize K-Medoids model
# Randomly choose initial medoids
initial_medoids = np.random.choice(range(data.shape[0]), size=5, replace=False).tolist()

# Create K-Medoids instance
kmedoids_instance = kmedoids(data.tolist(), initial_medoids)

# Measure the time for fitting the model
t0 = time.process_time()
kmedoids_instance.process()
t1 = time.process_time()

# Calculate and print the time taken
tt = t1 - t0
print("Total Time:", tt)

# Get the medoid centers and cluster labels
centers = kmedoids_instance.get_medoids()
labels = kmedoids_instance.get_clusters()

# Prepare cluster labels for visualization
cluster_labels = np.empty(data.shape[0], dtype=int)
for cluster_id, cluster in enumerate(labels):
    for index in cluster:
        cluster_labels[index] = cluster_id

# Define colors and markers for each cluster
colors = ["g", "r", "b", "y", "m"]
markers = ["+", "x", "*", ".", "d"]

# Plot each data point with the corresponding color and marker based on the cluster
for i in range(len(data)):
    plt.plot(data[i][0], data[i][1], color=colors[cluster_labels[i]], marker=markers[cluster_labels[i]])

# Plot the medoid centers
plt.scatter(data[centers, 0], data[centers, 1], marker="o", s=50, linewidths=5)

# Save the plot as an image file
plt.savefig("kmedoids_plot.png")

# Show the plot
plt.show()

