# FlexiCluster-A-Seamless-Dimensionality-Reduction-and-Clustering-Framework
A versatile framework combining UMAP dimensionality reduction with k-means and DBSCAN clustering. It dynamically adapts to data characteristics, offering robust clustering across diverse datasets. Ideal for exploratory data analysis and pattern discovery in complex, high-dimensional data.

You can view the Jupyter notebook [here](flexicluster.ipynb).

For testing purposes, I have used the MNIST digits dataset and dropped the target variable column to make it an unsupervised learning problem.
The pipeline could successfully reduce the 64 feautures into a 2 dimensional hidden latent manifold. Further it was able to succesfully cluster this into the 10 clusters (as there are 10 digits from 0-9)
