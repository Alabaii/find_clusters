import math
import random

def euclidean_distance(p1, p2):
    return math.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)

def calculate_sum_distances(point, cluster_points):
    total_distance = 0
    for cp in cluster_points:
        total_distance += euclidean_distance(point, cp)
    return total_distance

def find_cluster_center(cluster_points):
    min_sum_distances = float("inf")
    cluster_center = None
    for p in cluster_points:
        current_sum_distances = calculate_sum_distances(p, cluster_points)
        if current_sum_distances < min_sum_distances:
            min_sum_distances = current_sum_distances
            cluster_center = p
    return cluster_center

def k_medoids_single_run(points, k, max_iterations=100):
    
    medoids = random.sample(points, k)

    for _ in range(max_iterations):
        clusters = [[] for _ in range(k)]
        for p in points:
            min_distance = float("inf")
            closest_medoid_index = -1
            for i, m in enumerate(medoids):
                dist = euclidean_distance(p, m)
                if dist < min_distance:
                    min_distance = dist
                    closest_medoid_index = i
            clusters[closest_medoid_index].append(p)

        new_medoids = []
        for cluster in clusters:
            if cluster:
                new_medoids.append(find_cluster_center(cluster))
            else:
                new_medoids.append(random.choice(points))

        if all(euclidean_distance(medoids[i], new_medoids[i]) < 1e-6 for i in range(k)):
            break
        medoids = new_medoids

    total_within_cluster_distance = 0
    for i, cluster in enumerate(clusters):
        medoid = medoids[i]
        for p in cluster:
            total_within_cluster_distance += euclidean_distance(p, medoid)

    return clusters, medoids, total_within_cluster_distance

def k_medoids(points, k, num_runs=500, max_iterations=100):
    best_clusters = None
    best_medoids = None
    min_total_within_cluster_distance = float("inf")

    for _ in range(num_runs):
        clusters, medoids, total_within_cluster_distance = k_medoids_single_run(points, k, max_iterations)
        if total_within_cluster_distance < min_total_within_cluster_distance:
            min_total_within_cluster_distance = total_within_cluster_distance
            best_clusters = clusters
            best_medoids = medoids
    return best_clusters, best_medoids


