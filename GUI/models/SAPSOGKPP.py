import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import pairwise_distances_argmin_min


class SAPSO_GKPP:
	def __init__(self, n_clusters: int = 4, n_particles: int = 50, max_iter: int = 100,
			   w_max: float = 0.9, w_min: float = 0.5, c1: float = 1.49, c2: float = 1.49,
			   initial_temperature=100, cooling_rate=0.95, random_state=41):
		self.n_clusters = n_clusters
		self.n_particles = n_particles
		self.max_iter = max_iter
		self.w_max = w_max
		self.w_min = w_min
		self.c1 = c1
		self.c2 = c2
		self.T = initial_temperature
		self.alpha = cooling_rate
		self.seed = random_state

	def _kmeans_sse(self, X, centroids):
		labels, _ = pairwise_distances_argmin_min(X, centroids)
		sse = sum(np.sum((X[labels == k] - centroids[k]) ** 2) for k in range(len(centroids)))
		return sse, labels

	def _sample_from_ged(self, particles, n_features):
		flat_particles = np.array([p.flatten() for p in particles])  # shape (N, K*n_features)
		mean = np.mean(flat_particles, axis=0)
		std = np.std(flat_particles, axis=0) + 1e-6  # avoid zero std
		new_sample_flat = np.random.normal(loc=mean, scale=std)
		return new_sample_flat.reshape(self.n_clusters, n_features)

	def fit(self, X):
		np.random.seed(self.seed)
		n_samples, n_features = X.shape

		# Initialize using KMeans++
		kmeans = KMeans(n_clusters=self.n_clusters, init='k-means++', n_init=10, random_state=self.seed)
		kmeans.fit(X)
		initial_centroids = kmeans.cluster_centers_

		# Initialize particles and velocities
		particles = [np.copy(initial_centroids) for _ in range(self.n_particles)]
		velocities = [np.zeros((self.n_clusters, n_features)) for _ in range(self.n_particles)]

		particle_best = [np.copy(p) for p in particles]
		particle_best_fitness = [self._kmeans_sse(X, p)[0] for p in particles]

		# Global best initialization
		global_best_index = np.argmin(particle_best_fitness)
		global_best = np.copy(particle_best[global_best_index])
		global_best_fitness = particle_best_fitness[global_best_index]

		T = self.T  # reset local temperature for each fit

		for iteration in range(self.max_iter):
			w = self.w_max - ((self.w_max - self.w_min) * iteration / self.max_iter)

			for i in range(self.n_particles):
				current = particles[i]
				f_current, _ = self._kmeans_sse(X, current)

				# Update personal best
				if f_current < particle_best_fitness[i]:
					particle_best[i] = np.copy(current)
					particle_best_fitness[i] = f_current

				# Update global best
				if f_current < global_best_fitness:
					global_best = np.copy(current)
					global_best_fitness = f_current

				# Update velocity
				cognitive = self.c1 * np.random.rand() * (particle_best[i] - current)
				social = self.c2 * np.random.rand() * (global_best - current)
				velocities[i] = w * velocities[i] + cognitive + social

				new_position = current + velocities[i]

				# GED sampling
				ged_position = self._sample_from_ged(particles, n_features)
				f_ged, _ = self._kmeans_sse(X, ged_position)
				delta_f = f_ged - f_current

				# Simulated Annealing Acceptance
				if delta_f < 0 or np.random.rand() < np.exp(-delta_f / T):
					particles[i] = ged_position
				else:
					particles[i] = new_position

			T *= self.alpha  # Cool down

		self.cluster_centers_ = global_best
		self.inertia_ = global_best_fitness
		_, self.labels = self._kmeans_sse(X, global_best)
		return self

	def predict(self, X):
		labels, _ = pairwise_distances_argmin_min(X, self.cluster_centers_)
		return labels

	def fit_predict(self, X):
		self.fit(X)
		return self.labels_
