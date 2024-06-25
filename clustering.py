import numpy as np
from sklearn.cluster import KMeans
from kneed import KneeLocator
from sklearn.neighbors import NearestNeighbors as NN
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score
from sklearn.cluster import DBSCAN
import hdbscan.validity as dbcv_hdbscan



# Function to calculate hopkins statistic
def calculate_hopkins(data_points):
    max_vals = data_points.max(axis=0)
    min_vals = data_points.min(axis=0)
    np.random.seed(42)
    rand_x = np.random.uniform(low=min_vals[0], high=max_vals[0], size=(data_points.shape[0], 1))
    rand_y = np.random.uniform(low=min_vals[1], high=max_vals[1], size=(data_points.shape[0], 1))
    random_data = np.concatenate((rand_x, rand_y), axis=1)
    nn_distances = []
    for points in [data_points, random_data]:
        knn_model = NN(n_neighbors=1).fit(points)
        distances = []
        for i in range(points.shape[0]):
            distances_to_nearest, _ = knn_model.kneighbors(points[i, :].reshape(1, -1), 2, return_distance=True)
            distances.append(distances_to_nearest[0, -1])
        nn_distances.append(distances)  # 0 - data_points, 1 - random_data
    hopkins_statistic = sum(nn_distances[0]) / (sum(nn_distances[1]) + sum(nn_distances[0]))
    hopkins_statistic = round(hopkins_statistic, 3)
    return hopkins_statistic


# Function to find kmeans clustering result for one umap result
def kmeans_clustering(umap_result):
    embedding = umap_result['embedding']
    umap_n_neighbors = umap_result['n_neighbors']
    umap_min_dist = umap_result['min_dist']
    umap_metric = umap_result['metric']
    umap_n_components = umap_result['n_components']
    trustworthiness = umap_result['trustworthiness']

    # KMeans clustering iterating over number of clusters 5 - 11
    min_cluster = 5
    max_cluster = 12

    inertia_values = []
    silhouette_scores = []
    davies_bouldin_scores = []
    calinski_harabasz_scores = []

    for num_clusters in range(min_cluster, max_cluster):
        kmeans_model = KMeans(n_clusters=num_clusters, random_state=42, n_init='auto').fit(embedding)

        inertia_values.append(kmeans_model.inertia_)
        silhouette_scores.append(silhouette_score(embedding, kmeans_model.labels_))
        davies_bouldin_scores.append(davies_bouldin_score(embedding, kmeans_model.labels_))
        calinski_harabasz_scores.append(calinski_harabasz_score(embedding, kmeans_model.labels_))

    # Finding the number of clusters using the elbow method
    kn = KneeLocator(range(min_cluster, max_cluster), inertia_values, curve='convex', direction='decreasing')
    found_clusters = kn.elbow

    max_silhouette_score = max(silhouette_scores)
    # Adding 5 to the index to get the number of clusters (since we start from 5)
    clusters_for_max_silhouette_score = 5 + silhouette_scores.index(max_silhouette_score)
    clusters_for_min_db_score = 5 + davies_bouldin_scores.index(min(davies_bouldin_scores))
    clusters_for_max_ch_score = 5 + calinski_harabasz_scores.index(max(calinski_harabasz_scores))
    hopkins_statistic = calculate_hopkins(embedding)

    kmap_result = {
        'algo': 'k-means',
        'n_clusters_found': found_clusters,
        'n_clusters_db_score_is_min': clusters_for_min_db_score,
        'n_clusters_ch_score_is_max': clusters_for_max_ch_score,
        'n_clusters_silhouette_score_is_max': clusters_for_max_silhouette_score,
        'silhouette_score': max_silhouette_score,
        'hopkins_statistic': hopkins_statistic,
        'umap_n_neighbors': umap_n_neighbors,
        'umap_min_dist': umap_min_dist,
        'umap_metric': umap_metric,
        'umap_n_components': umap_n_components,
        'trustworthiness': trustworthiness
    }
    return kmap_result


def dbscan_clustering(umap_result):
    embedding = umap_result['embedding']
    umap_n_neighbors = umap_result['n_neighbors']
    umap_min_dist = umap_result['min_dist']
    umap_metric = umap_result['metric']
    umap_n_components = umap_result['n_components']
    trustworthiness = umap_result['trustworthiness']

    min_samples = [5, 10, 20, 30]
    eps = [0.1, 0.3, 0.7, 1.0]
    best_val_index_found = -100
    best_eps_found = None
    best_min_samples_found = None
    best_cluster_nlabels = None

    for e in eps:
        for m in min_samples:

            # DBSCAN clustering over different values of eps and min_samples
            dbscan_model = DBSCAN(eps=e, min_samples=m).fit(embedding)
            labels = dbscan_model.labels_
            try:
                validity_index = dbcv_hdbscan.validity_index(embedding.astype(np.float64), labels)
            except ValueError as er:
                print(er)
                validity_index = -1
            if np.isnan(validity_index):
                validity_index = -1
            # If the validity index is greater than the best validity index,
            # update the best validity index, best eps, min_samples, and the n_cluster labels
            if validity_index > best_val_index_found:
                best_val_index_found = validity_index
                best_eps_found = e
                best_min_samples_found = m
                best_cluster_nlabels = labels
    cap_h = calculate_hopkins(embedding)
    dbscan_result = {
        'algo': 'dbscan',
        'eps': best_eps_found,
        'dbscan_min_samples': best_min_samples_found,
        'n_clusters_found': np.max(best_cluster_nlabels) + 1,  # +1 because the labels start from 0
        'validity_index': best_val_index_found,
        'hopkins_statistic': cap_h,
        'umap_n_neighbors': umap_n_neighbors,
        'umap_min_dist': umap_min_dist,
        'umap_metric': umap_metric,
        'umap_n_components': umap_n_components,
        'trustworthiness': trustworthiness
    }
    return dbscan_result


def clustering(umap_result):
    # 1. KMeans
    kmeans_results = kmeans_clustering(umap_result)

    # Test 1
    if kmeans_results['n_clusters_found'] == kmeans_results['n_clusters_db_score_is_min'] == kmeans_results[
        'n_clusters_ch_score_is_max'] == kmeans_results['n_clusters_silhouette_score_is_max']:
        print("KMeans is the best clustering algorithm for this embedding - Test 1 passed")
        return kmeans_results
    # Test 2
    elif kmeans_results['n_clusters_db_score_is_min'] == kmeans_results['n_clusters_ch_score_is_max'] == kmeans_results[
        'n_clusters_silhouette_score_is_max']:
        print("KMeans is the best clustering algorithm for this embedding - Test 2 passed")
        return kmeans_results

    # 2. DBSCAN
    print("DBSCAN is the best clustering algorithm for this embedding - Test 1&2 failed")
    dbscan_results = dbscan_clustering(umap_result)
    return dbscan_results
