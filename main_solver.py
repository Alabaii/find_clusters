import numpy as np
import matplotlib.pyplot as plt
from cluster_solver import k_medoids, find_cluster_center, euclidean_distance

def read_points_from_file(filepath):
    points = []
    with open(filepath, 'r') as f:
        next(f)
        for line in f:
            x_str, y_str = line.strip().split()
            points.append((float(x_str.replace(',', '.')), float(y_str.replace(',', '.'))))
    return points

def identify_anomalous_points(points, num_anomalies=3):
    sum_distances = []
    for i, p1 in enumerate(points):
        current_sum = 0
        for j, p2 in enumerate(points):
            if i != j:
                current_sum += euclidean_distance(p1, p2)
        sum_distances.append((current_sum, p1))


    sum_distances.sort(key=lambda x: x[0], reverse=True)
    anomalous_points = [p for _, p in sum_distances[:num_anomalies]]
    return anomalous_points


file_a_points = read_points_from_file('27_A_23209.txt')

k_a = 2
clusters_a, medoids_a = k_medoids(file_a_points, k_a)


centers_a = [find_cluster_center(cluster) for cluster in clusters_a]


px = max(center[0] for center in centers_a)
py = max(center[1] for center in centers_a)


file_b_points_raw = read_points_from_file('27_B_23209.txt')


anomalous_points_b = identify_anomalous_points(file_b_points_raw, num_anomalies=3)

file_b_points = [p for p in file_b_points_raw if p not in anomalous_points_b]

k_b = 3
clusters_b, medoids_b = k_medoids(file_b_points, k_b)


centers_b = [find_cluster_center(cluster) for cluster in clusters_b]


cluster_sizes_b = [(len(cluster), i) for i, cluster in enumerate(clusters_b)]
cluster_sizes_b.sort()


center_min_points = centers_b[cluster_sizes_b[0][1]]
center_max_points = centers_b[cluster_sizes_b[-1][1]]

qx = abs(center_min_points[0] - center_max_points[0])
qy = abs(center_min_points[1] - center_max_points[1])


output_px = int(abs(px * 10000))
output_py = int(abs(py * 10000))
output_qx = int(abs(qx * 10000))
output_qy = int(abs(qy * 10000))

print(f"{output_px} {output_py}")
print(f"{output_qx} {output_qy}")


plt.figure(figsize=(12, 6))


plt.subplot(1, 2, 1)
for i, cluster in enumerate(clusters_a):
    x = [p[0] for p in cluster]
    y = [p[1] for p in cluster]
    plt.scatter(x, y, label=f'Cluster {i+1}')
x_centers_a = [c[0] for c in centers_a]
y_centers_a = [c[1] for c in centers_a]
plt.scatter(x_centers_a, y_centers_a, marker='X', s=200, color='black', label='Centers')
plt.title('File A Clusters')
plt.xlabel('X Coordinate')
plt.ylabel('Y Coordinate')
plt.legend()
plt.grid(True)


plt.subplot(1, 2, 2)
for i, cluster in enumerate(clusters_b):
    x = [p[0] for p in cluster]
    y = [p[1] for p in cluster]
    plt.scatter(x, y, label=f'Cluster {i+1}')
x_centers_b = [c[0] for c in centers_b]
y_centers_b = [c[1] for c in centers_b]
plt.scatter(x_centers_b, y_centers_b, marker='X', s=200, color='black', label='Centers')


x_anomalous = [p[0] for p in anomalous_points_b]
y_anomalous = [p[1] for p in anomalous_points_b]
plt.scatter(x_anomalous, y_anomalous, marker='o', s=100, color='red', label='Anomalous Points')

plt.title('File B Clusters (Anomalous points excluded)')
plt.xlabel('X Coordinate')
plt.ylabel('Y Coordinate')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.savefig('clustering_visualization.png')


