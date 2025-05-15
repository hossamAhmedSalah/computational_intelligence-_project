import numpy as np
from sklearn.metrics import pairwise_distances_argmin_min
from sklearn.metrics import silhouette_score, calinski_harabasz_score


class SimulatedAnnealingClustering:
	def __init__(self, k=4, max_iter=100, initial_temp=100, cooling_rate=0.95):
		self.k = k
		self.max_iter = max_iter
		self.initial_temp = initial_temp
		self.cooling_rate = cooling_rate
		self.centroids_ = None
		self.labels_ = None
		self.cost_ = None
		#self.scaler = StandardScaler()
		self.fitted = False

	def _initialize_centroids(self, X):
		indices = np.random.choice(len(X), self.k, replace=False)
		return X[indices]

	def _assign_clusters(self, X, centroids):
		labels, _ = pairwise_distances_argmin_min(X, centroids)
		return labels

	def _calculate_cost(self, X, labels, centroids):
		return sum(np.linalg.norm(X[i] - centroids[labels[i]]) ** 2 for i in range(len(X)))

	def _perturb_centroids(self, centroids, scale=0.1):
		noise = np.random.normal(0, scale, centroids.shape)
		return centroids + noise

	def fit(self, X):
		current_centroids = self._initialize_centroids(X)
		current_labels = self._assign_clusters(X, current_centroids)
		current_cost = self._calculate_cost(X, current_labels, current_centroids)

		best_centroids = current_centroids.copy()
		best_labels = current_labels.copy()
		best_cost = current_cost

		temp = self.initial_temp

		for _ in range(self.max_iter):
			new_centroids = self._perturb_centroids(current_centroids)
			new_labels = self._assign_clusters(X, new_centroids)
			new_cost = self._calculate_cost(X, new_labels, new_centroids)

			if new_cost < current_cost or np.random.rand() < np.exp((current_cost - new_cost) / temp):
				current_centroids = new_centroids
				current_labels = new_labels
				current_cost = new_cost

				if new_cost < best_cost:
					best_centroids = new_centroids
					best_labels = new_labels
					best_cost = new_cost

			temp *= self.cooling_rate

		self.centroids_ = best_centroids
		self.labels_ = best_labels
		self.cost_ = best_cost
		self.fitted = True
		return self

	def predict(self, X_new):
		if not self.fitted:
			raise ValueError("Model has not been fitted yet. Call 'fit' with training data first.")
		labels, _ = pairwise_distances_argmin_min(X_new, self.centroids_)
		return labels

	def score(self, X):
		if not self.fitted:
			raise ValueError("Model has not been fitted yet.")
		sil = silhouette_score(X, self.labels_)
		ch = calinski_harabasz_score(X, self.labels_)
		return {"silhouette": sil, "calinski_harabasz": ch, "cost": self.cost_}
