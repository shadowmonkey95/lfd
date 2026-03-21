# https://github.com/lifelonglab/lifelong-anomaly-detection-scenarios/blob/main/clustering.py
import random
from typing import List, Tuple, Union

import numpy as np
from numpy.random import shuffle
from sklearn.cluster import KMeans


def _create_kmeans(n_clusters):
    return KMeans(
        n_clusters=n_clusters,
        init='k-means++',
        n_init=10,
        max_iter=300,
        tol=0.0001,
        verbose=0,
        random_state=42,
        copy_x=True,
        algorithm='lloyd'
    )

def cluster_kmeans(data: np.ndarray, clusters_no: int):
    """ Data should not contain labels """
    kmeans_model = _create_kmeans(clusters_no)
    clustering_results = kmeans_model.fit_predict(data)
    distances = kmeans_model.transform(data)
    return clustering_results, distances


def create_concepts(data: np.ndarray, concepts_no: int, size_per_concept: int) -> List[np.ndarray]:
    clusters_id, distances = cluster_kmeans(data, concepts_no)

    concepts = []

    for c in range(concepts_no):
        # select data for cluster by sorting it by distance
        cluster_data_with_distances = [(d, dists[c_id]) for c_id, d, dists in
                                       zip(clusters_id, data, distances) if c_id == c]
        sorted_cluster_data = [d for d, _ in sorted(cluster_data_with_distances, key=lambda t: t[1])]
        selected_data = np.array(sorted_cluster_data[:min(size_per_concept, len(sorted_cluster_data))])
        shuffle(selected_data)  # we don't want data to stay sorted
        concepts.append(selected_data)

    return concepts


# def create_concepts_with_order(
#     data: np.ndarray,
#     concepts_no: int,
#     size_per_concept: int,
#     sort_column: int
# ) -> List[np.ndarray]:
#     data_sorted = data[data[:, sort_column].argsort()]

#     total_len = len(data_sorted)
#     concept_size = min(size_per_concept, total_len // concepts_no)

#     concepts = []
#     for i in range(concepts_no):
#         start_idx = i * concept_size
#         # end_idx = min(start_idx + concept_size, total_len)
#         if i == concepts_no - 1:
#             end_idx = total_len
#         else:
#             end_idx = start_idx + concept_size
#         concept = data_sorted[start_idx:end_idx]
#         if len(concept) > 0:
#             concepts.append(concept)

#     return concepts

def create_concepts_with_order(
    data: np.ndarray,
    concepts_no: int,
    size_per_concept: int,
    sort_column: int,
    tail_strategy: str = "drop"
) -> List[np.ndarray]:
    """
    Generalized concept creator with tail-end handling.
    
    Strategies:
    - 'drop': Each concept is exactly size_per_concept. Remaining data is discarded.
             (Best for perfect Heatmap symmetry).
    - 'include': Last concept takes its share + all remaining data.
                 (Best for ensuring 100% data coverage).
    - 'overflow': If remaining data is > 50% of a concept size, create an extra concept.
                  (Best balance for lifelong learning).
    """
    # 1. Sort by time
    data_sorted = data[data[:, sort_column].argsort()]
    total_len = len(data_sorted)
    
    # 2. Determine base constraints
    step = min(size_per_concept, total_len // concepts_no)
    concepts = []

    for i in range(concepts_no):
        start_idx = i * step
        
        # Determine the end point based on strategy
        if i == concepts_no - 1:
            if tail_strategy == "include":
                end_idx = total_len
            else:
                end_idx = start_idx + step
        else:
            end_idx = start_idx + step
            
        concept = data_sorted[start_idx:end_idx]
        if len(concept) > 0:
            concepts.append(concept)

    # 3. Handle 'overflow' strategy
    if tail_strategy == "overflow":
        last_end = concepts_no * step
        remaining_count = total_len - last_end
        if remaining_count > (step / 2): # If more than half a concept is left
            concepts.append(data_sorted[last_end:])

    print(f"Strategy [{tail_strategy}]: Created {len(concepts)} concepts.")
    print(f"Sizes: {[len(c) for c in concepts]}")
    
    return concepts



def _find_closest_cluster(base_cluster_centroid: np.ndarray, clusters: List[Tuple[np.ndarray, np.ndarray]]):
    distances = [(i, np.linalg.norm(base_cluster_centroid - c_centroid)) for i, (c_data, c_centroid) in
                 enumerate(clusters)]
    # distances = [(i, wassertein_distance(np.array(base_cluster.data), np.array(c.data))) for i, c in enumerate(clusters)]
    distances.sort(key=lambda x: x[1])
    c_id = distances[0][0]
    return c_id, clusters[c_id][0]


def _calculate_centroid(c):
    return np.mean(c, axis=0)


def _reassign_clusters(anomaly_clusters, normal_clusters):
    normal_clusters_with_centroids = [(c, _calculate_centroid(c)) for c in normal_clusters]
    for j, (_, c) in enumerate(normal_clusters_with_centroids):
        for k, (_, c1) in enumerate(normal_clusters_with_centroids):
            print(j, k, np.linalg.norm(c - c1))

    anomaly_clusters_with_centroids = [(c, _calculate_centroid(c)) for c in anomaly_clusters]
    left_anomaly_clusters = anomaly_clusters_with_centroids
    sorted_anomaly_clusters = []
    for i, (c_data, centroid) in enumerate(normal_clusters_with_centroids):
        anomaly_cluster_id, anomalies = _find_closest_cluster(centroid, left_anomaly_clusters)
        sorted_anomaly_clusters.append(anomalies)
        left_anomaly_clusters = [c for i, c in enumerate(left_anomaly_clusters) if i != anomaly_cluster_id]
    return sorted_anomaly_clusters


def create_random_anomaly_clusters(anomaly_data, clusters_no, size_per_cluster):
    random.shuffle(anomaly_data)
    return [anomaly_data[size_per_cluster * i: size_per_cluster * (i + 1)] for i in range(clusters_no)]


def create_anomaly_clusters_randomly_assigned(anomaly_data, clusters_no, size_per_cluster):
    anomaly_clusters = create_concepts(anomaly_data, concepts_no=clusters_no, size_per_concept=size_per_cluster)
    shuffle(anomaly_clusters)
    return anomaly_clusters


def create_anomaly_clusters_closest_to_normal(anomaly_data, normal_clusters, clusters_no, size_per_cluster):
    anomaly_clusters = create_concepts(anomaly_data, concepts_no=clusters_no, size_per_concept=size_per_cluster)
    return _reassign_clusters(anomaly_clusters=anomaly_clusters, normal_clusters=normal_clusters)

def create_anomaly_clusters_time_based(normal_clusters, anomaly_data, sort_column, max_anomalies, config):
    anomaly_data_sorted = anomaly_data[anomaly_data[:, sort_column].argsort()]
    
    anomaly_clusters = []
    cluster_info = []
    
    for idx, normal_cluster in enumerate(normal_clusters):
        start_time = normal_cluster[:, sort_column].min()
        end_time = normal_cluster[:, sort_column].max()
        
        # Test strict matching
        mask = (anomaly_data_sorted[:, sort_column] >= start_time) & \
               (anomaly_data_sorted[:, sort_column] <= end_time)
        anomalies_in_range = anomaly_data_sorted[mask]
        
        # DEBUG: 0.0 ROC-AUC heatmap error
        # If empty, grab the 'nearest' anomalies in time
        if len(anomalies_in_range) == 0:
            # Find index of first anomaly that happens after start_time
            pos = np.searchsorted(anomaly_data_sorted[:, sort_column], start_time)
            # Take a small window around that time point
            half = max_anomalies // 2
            start_idx = max(0, pos - half)
            end_idx = min(len(anomaly_data_sorted), pos + half)
            anomalies_in_range = anomaly_data_sorted[start_idx:end_idx]
            print(f"INFO: Concept {idx} used nearest-neighbor time matching.")

        # Limit to max_anomalies
        original_count = len(anomalies_in_range)
        anomalies_final = anomalies_in_range[:max_anomalies]
        
        # anomaly_clusters.append(anomalies_in_range)
        
        cluster_info.append({
            'cluster_id': idx,
            'time_range': (float(start_time), float(end_time)),
            'normal_size': len(normal_cluster),
            'anomaly_count': len(anomalies_final),
            'original_anomaly_count': original_count,
            'is_empty': len(anomalies_final) == 0
        })
        anomaly_clusters.append(anomalies_final)
    
    # Existing helper
    write_cluster_info(normal_clusters, anomaly_clusters, cluster_info, sort_column, 'tmp_clusters.txt')
    
    return anomaly_clusters

def create_all_anomaly_clusters(anomaly_data, clusters_no):
    return [anomaly_data.copy() for _ in range(clusters_no)]

# REGULAR CLUSTER HELPER
# def write_cluster_info(normal_clusters, anomaly_clusters, cluster_info, sort_column, filename='tmp_clusters.txt'):
#     """Write detailed cluster information to a file"""
#     with open(filename, 'w') as f:
#         f.write("=" * 80 + "\n")
#         f.write("CLUSTER ANALYSIS REPORT\n")
#         f.write("=" * 80 + "\n\n")
        
#         # Summary statistics
#         total_clusters = len(normal_clusters)
#         empty_clusters = sum(1 for info in cluster_info if info['is_empty'])
#         total_normal = sum(len(nc) for nc in normal_clusters)
#         total_anomalies = sum(info['anomaly_count'] for info in cluster_info)
        
#         f.write(f"SUMMARY:\n")
#         f.write(f"  Total Clusters: {total_clusters}\n")
#         f.write(f"  Empty Anomaly Clusters: {empty_clusters} ({empty_clusters/total_clusters*100:.1f}%)\n")
#         f.write(f"  Total Normal Samples: {total_normal}\n")
#         f.write(f"  Total Anomaly Samples: {total_anomalies}\n")
#         f.write(f"  Sort Column Index: {sort_column}\n")
#         f.write("\n" + "=" * 80 + "\n\n")
        
#         # Detailed cluster information
#         for idx, (normal_cluster, anomaly_cluster, info) in enumerate(
#             zip(normal_clusters, anomaly_clusters, cluster_info)
#         ):
#             f.write(f"CLUSTER {idx}\n")
#             f.write("-" * 80 + "\n")
            
#             # Time range
#             f.write(f"  Time Range: [{info['time_range'][0]:.2f}, {info['time_range'][1]:.2f}]\n")
            
#             # Normal cluster info
#             f.write(f"\n  Normal Cluster:\n")
#             f.write(f"    Size: {info['normal_size']}\n")
#             f.write(f"    Time Min: {normal_cluster[:, sort_column].min():.2f}\n")
#             f.write(f"    Time Max: {normal_cluster[:, sort_column].max():.2f}\n")
#             f.write(f"    Shape: {normal_cluster.shape}\n")
            
#             # Anomaly cluster info
#             f.write(f"\n  Anomaly Cluster:\n")
#             f.write(f"    Size: {info['anomaly_count']}\n")
#             f.write(f"    Original Count (before limit): {info['original_anomaly_count']}\n")
#             f.write(f"    Empty: {info['is_empty']}\n")
#             f.write(f"    Shape: {anomaly_cluster.shape}\n")
            
#             if not info['is_empty']:
#                 f.write(f"    Time Min: {anomaly_cluster[:, sort_column].min():.2f}\n")
#                 f.write(f"    Time Max: {anomaly_cluster[:, sort_column].max():.2f}\n")
                
#                 # Show first few anomaly samples
#                 f.write(f"\n    First 3 anomaly samples:\n")
#                 for i, sample in enumerate(anomaly_cluster[:3]):
#                     f.write(f"      [{i}] {sample}\n")
#             else:
#                 f.write(f"WARNING: No anomalies found in this time window\n")
            
#             # Ratio
#             if info['normal_size'] > 0:
#                 ratio = info['anomaly_count'] / info['normal_size'] * 100
#                 f.write(f"\n  Anomaly/Normal Ratio: {ratio:.2f}%\n")
            
#             f.write("\n" + "=" * 80 + "\n\n")
        
#         f.write(f"\nReport generated successfully!\n")
    
#     print(f"Cluster information written to '{filename}'")

def write_cluster_info(normal_clusters, anomaly_clusters, cluster_info, sort_column, filename='tmp_clusters.txt'):
    with open(filename, 'w') as f:
        f.write("=" * 80 + "\n")
        f.write("CLUSTER ANALYSIS REPORT\n")
        f.write("=" * 80 + "\n\n")
        
        # Summary statistics
        total_clusters = len(normal_clusters)
        empty_clusters = sum(1 for info in cluster_info if info.get('is_empty', False))
        total_normal = sum(len(nc) for nc in normal_clusters)
        total_anomalies = sum(info.get('anomaly_count', 0) for info in cluster_info)
        
        f.write(f"SUMMARY:\n")
        f.write(f"  Total Clusters: {total_clusters}\n")
        f.write(f"  Empty Anomaly Clusters: {empty_clusters} ({empty_clusters/total_clusters*100:.1f}%)\n")
        f.write(f"  Total Normal Samples: {total_normal}\n")
        f.write(f"  Total Anomaly Samples: {total_anomalies}\n")
        f.write(f"  Sort Column Index: {sort_column}\n")
        f.write("\n" + "=" * 80 + "\n\n")
        
        for idx, (normal_cluster, anomaly_cluster, info) in enumerate(
            zip(normal_clusters, anomaly_clusters, cluster_info)
        ):
            f.write(f"CLUSTER {idx}\n")
            f.write("-" * 80 + "\n")
            
            t_range = info.get('time_range', (0, 0))
            f.write(f"  Time Range: [{t_range[0]:.2f}, {t_range[1]:.2f}]\n")
            
            f.write(f"\n  Normal Cluster:\n")
            f.write(f"    Size: {info.get('normal_size', len(normal_cluster))}\n")
            f.write(f"    Time Min: {normal_cluster[:, sort_column].min():.2f}\n")
            f.write(f"    Time Max: {normal_cluster[:, sort_column].max():.2f}\n")
            f.write(f"    Shape: {normal_cluster.shape}\n")
            
            f.write(f"\n  Anomaly Cluster:\n")
            a_count = info.get('anomaly_count', len(anomaly_cluster))
            orig_count = info.get('original_anomaly_count', a_count)
            is_empty = info.get('is_empty', a_count == 0)
            
            f.write(f"    Size: {a_count}\n")
            f.write(f"    Original Count: {orig_count}\n")
            f.write(f"    Empty: {is_empty}\n")
            f.write(f"    Shape: {anomaly_cluster.shape}\n")
            
            if not is_empty and len(anomaly_cluster) > 0:
                f.write(f"    Time Min: {anomaly_cluster[:, sort_column].min():.2f}\n")
                f.write(f"    Time Max: {anomaly_cluster[:, sort_column].max():.2f}\n")
                f.write(f"\n    First 3 samples:\n")
                for i, sample in enumerate(anomaly_cluster[:3]):
                    f.write(f"      [{i}] {sample}\n")
            else:
                f.write(f"WARNING: No anomalies found in this time window\n")
            
            n_size = info.get('normal_size', len(normal_cluster))
            if n_size > 0:
                ratio = (a_count / n_size) * 100
                f.write(f"\n  Anomaly/Normal Ratio: {ratio:.2f}%\n")
            
            f.write("\n" + "=" * 80 + "\n\n")
        
        f.write(f"\nReport generated successfully!\n")
    
    print(f"Cluster information written to '{filename}'")
