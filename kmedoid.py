
#K-Medoid Algorithm


import time
import numpy as np
import matplotlib.pyplot as plt
from sklearn_extra.cluster import KMedoids  # Use sklearn_extra for KMedoids

# Generate random data
data = np.random.rand(10000, 2) * 100

# Initialize K-Medoids model with 5 clusters and Euclidean distance
km = KMedoids(n_clusters=5, metric='euclidean', init='random')

# Measure the time for fitting the model
t0 = time.process_time()
km.fit(data)
t1 = time.process_time()

# Calculate and print the time taken
tt = t1 - t0
print("Total Time:", tt)

# Get the medoid centers and cluster labels
centers = km.cluster_centers_
labels = km.labels_

print("Cluster Centers:", centers)

# Define colors and markers for each cluster
colors = ["g", "r", "b", "y", "m"]
markers = ["+", "x", "*", ".", "d"]

# Plot each data point with the corresponding color and marker based on the cluster
for i in range(len(data)):
    plt.plot(data[i][0], data[i][1], color=colors[labels[i]], marker=markers[labels[i]])

# Plot the medoid centers
plt.scatter(centers[:, 0], centers[:, 1], marker="o", s=50, linewidths=5)

# Save the plot as an image file
plt.savefig("kmedoids_plot.png")

# Show the plot
plt.show()