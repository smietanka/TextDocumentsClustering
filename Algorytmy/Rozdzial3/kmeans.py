from sklearn.cluster import KMeans
km = KMeans(n_clusters=num_clusters, max_iter=4)
km.fit(tfidfMatrix)
clusters = km.labels_.tolist()