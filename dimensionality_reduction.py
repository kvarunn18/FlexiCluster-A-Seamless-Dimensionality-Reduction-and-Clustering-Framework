import umap
from sklearn.manifold import trustworthiness


# The following function is used to reduce the dimensionality of the data using UMAP
def umap_dim_red(cap_x_df, n_neighbors, min_dist, metric, n_components):
    reducer = umap.UMAP(n_neighbors=n_neighbors,
                        n_components=n_components,
                        metric=metric,
                        min_dist=min_dist,
                        spread=1.0,
                        random_state=42)
    embedding = reducer.fit_transform(cap_x_df)
    trness = trustworthiness(cap_x_df, embedding, n_neighbors=n_neighbors,metric = metric)
    result = {'embedding': embedding,
              'n_neighbors': n_neighbors,
              'min_dist': min_dist,
              'metric': metric,
              'n_components': n_components,
              'trustworthiness': trness}
    return result
