# K - means clustering #
# Define the number of clusters to the split the data which is K
# Select K random points within the data
# 3 - Calculate distance between centroids and other points
# 4 - Assign the points to the closest centroid
# 5 - Calculate the centre of each cluster
# Repeat 3-5

from sklearn.cluster import KMeans
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os

col_names = ['a1','a2','a3','a4','a5','a6','a7','a8','a9']
data = pd.read_csv("kmeans/dataset/clustering_data.csv",skiprows=1,header=None,names=col_names)


# min-max normalization
numerical_columns = data.columns
scaler = MinMaxScaler()
data[numerical_columns] = scaler.fit_transform(data[numerical_columns])
k_cluster_number = 5
k_means = KMeans(n_clusters=k_cluster_number,init='k-means++',random_state=42)

k_means.fit(data)

data['cluster'] = k_means.labels_
cluster_counts = data['cluster'].value_counts()

# Save cluster counts to result.txt
file_path = 'kmeans/output/result.txt'

if not os.path.exists(file_path):
    with open(file_path, 'w') as result_file:
        for index,row in data.iterrows():
            str_index =str(index+1)
            result_file.write(f"Record {str_index}:    ")
            result_file.write(f"Cluster {row['cluster']}\n")
else:
    with open(file_path, 'w') as result_file:
        for index,row in data.iterrows():
            str_index =str(index+1)
            result_file.write(f"Record {str_index}:    ")
            result_file.write(f"Cluster {int(row['cluster'])}\n")

cluster_centers = k_means.cluster_centers_

# Calculate WCSS and BCSS
# KMeans package provides wcss with inertia_
wcss = k_means.inertia_
bcss = 0  # Initialize BCSS
bcss2= 0



# overall_mean ==> sum up all data points divide by size
overall_mean = np.sum(data[numerical_columns].values)/data[numerical_columns].size

# total sum of squares ==> sum of all squared distances between data points and overall_mean 
tss = ((data[numerical_columns].values-overall_mean)**2).sum()

# tss = bcss + wcss
bcss = tss- wcss

# to check tss = bcss + wcss, I calculated bcss again as bcss2 with given formula
# bcss and bcss2 values are equivalent, therefore calculation is correct
bcss2 = np.sum([len(data[data['cluster'] == i]) * np.sum((cluster_centers[i] - overall_mean)**2) for i in range(k_cluster_number)])

# Print WCSS and BCSS and TSS
print("WCSS:", wcss)
print("BCSS:", bcss)
print("BCSS2: ", bcss2)
print("TSS: ", tss)

print("BCSS/TSS: ", bcss/tss )

# Dunn index calculations
# First initalize dunn index
# Then calculate intra_cluster_distance
# intra_cd = Intra-cluster distance is the distance between a data point and the cluster centroid within a cluster.


# cluster_key is the cluster number from 0 to 4
max_intra_cluster_distance = 0
for cluster_key in cluster_counts.keys():
    
    given_cluster_centroids = cluster_centers[cluster_key]
    data_for_given_cluster = data[data['cluster'] == cluster_key].copy()
    # drop the cluster column
    data_for_given_cluster = data_for_given_cluster.drop('cluster',axis=1)
    i=0
    for column in numerical_columns:
        # Calculate Euclidean distances for each data point
        data_for_given_cluster[column] = ((data_for_given_cluster[column] - given_cluster_centroids[i]) ** 2)
        i += 1

    # data_for_given_cluster will have a column named squared_distances, it will represent each row's distance to the centroid
    data_for_given_cluster['squared_distances'] = data_for_given_cluster[numerical_columns].sum(axis=1)

    # square root of each distance
    euclidean_distances = np.sqrt(data_for_given_cluster['squared_distances'])

    # sum of euclidean distances for given cluster
    intra_cluster_distance = np.max(euclidean_distances)

    # if new distance is bigger then old distance
    # new distance will assign to max_intra_cluster_distance
    max_intra_cluster_distance=max(max_intra_cluster_distance,intra_cluster_distance)


# Then calculate inter_cluster_distance
# inter_cd = Distance between cluster centroids
min_inter_cluster_distance = 10
for centroid in cluster_centers:
    distance_list = []
    for centroid_column in cluster_centers:
        distance_list.append(np.sqrt(np.sum((centroid-centroid_column)**2)))   

    # Filter out zero distances
    # Loop checks cluster1-cluster1 therefore we should filter out zero distances
    non_zero_distances = [distance for distance in distance_list if distance != 0]
    inter_cluster_distance = min(non_zero_distances)   
   
    min_inter_cluster_distance = min(min_inter_cluster_distance,inter_cluster_distance)

dunn_index = (min_inter_cluster_distance/max_intra_cluster_distance)

with open(file_path, 'a') as result_file:
    result_file.write("\n\n\n")

    for index,cluster in cluster_counts.items():
        result_file.write(f"Cluster {index}: {cluster} records\n")

    result_file.write("\n\nWCSS: {}\n".format(wcss))
    result_file.write("BCSS: {}\n".format(bcss))
    result_file.write("Dunn Index: {}\n".format(dunn_index))

import tkinter as tk
from tkinter import ttk
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

def update_plot():
    selected_x = x_options.get()
    selected_y = y_options.get()

    ax.clear() 
    # Scatter plot with different colors for each cluster
    for cluster in data['cluster'].unique():
        cluster_data = data[data['cluster'] == cluster]
        ax.scatter(cluster_data[selected_x], cluster_data[selected_y], label=f'Cluster {cluster}')

    # Set labels and legend
    ax.set_xlabel(selected_x)
    ax.set_ylabel(selected_y)
    ax.legend()
    plt.scatter(x=data[selected_x], y=data[selected_y], c=data['cluster'])
    plt.xlim(0, 1)
    plt.ylim(0, 1)
    plt.title(f'Scatter Plot ({selected_x} vs {selected_y})')
    plt.xlabel(selected_x)
    plt.ylabel(selected_y)
    
    canvas.draw()

# Create tkinter window
# data / cluster visualization
root = tk.Tk()
root.title("Scatter Plot Selector")

# Create options for x and y axes
all_keys = list(data.keys())
x_options = ttk.Combobox(root, values=all_keys[:-1])
x_options.set('a2')  # Default x-axis
x_label = ttk.Label(root, text="X-axis:")
x_label.grid(row=0, column=0)
x_options.grid(row=0, column=1)

y_options = ttk.Combobox(root, values=all_keys[:-1])
y_options.set('a5')  # Default y-axis
y_label = ttk.Label(root, text="Y-axis:")
y_label.grid(row=1, column=0)
y_options.grid(row=1, column=1)

# Create a button to update the plot
plot_button = ttk.Button(root, text="Update Plot", command=update_plot)
plot_button.grid(row=2, column=0, columnspan=2)

# Create matplotlib figure and canvas
fig, ax = plt.subplots()
canvas = FigureCanvasTkAgg(fig, master=root)
canvas_widget = canvas.get_tk_widget()
canvas_widget.grid(row=3, column=0, columnspan=2)

# Initial plot
update_plot()

root.mainloop()
