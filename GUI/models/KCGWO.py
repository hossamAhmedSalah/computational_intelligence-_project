import numpy as np
from sklearn.cluster import KMeans
from tqdm import tqdm


class KCGWO:
	def __init__(self, n_clusters: int = 3, pop_size: int = 30, max_iter: int = 100):
		"""
		Initialize the K-means Clustering based Grey Wolf Optimizer

		Parameters:
		- n_clusters: number of clusters to form
		- pop_size: number of wolves in the pack
		- max_iter: maximum number of iterations
		"""
		self.n_clusters = n_clusters
		self.pop_size = pop_size
		self.max_iter = max_iter
		self.best_centroids = None
		self.best_score = float('inf')
		self.convergence_curve = np.zeros(max_iter)
		self.all_positions = []
		self.X = None
		self.n_samples = 0
		self.n_features = 2

		self.delta_score = None
		self.beta_score = None
		self.alpha_score = None
		self.delta_pos = None
		self.beta_pos = None
		self.alpha_pos = None
		self.fitness = None
		self.population = None
		self.max_val = None
		self.min_val = None
		self.dim = None

	def fit(self, X:np.ndarray):
		"""
		Fit the KCGWO algorithm to the data

		Parameters:
		- X: input data of shape (n_samples, n_features)

		Returns:
		- self: returns an instance of self
		"""
		self.X = X
		self.n_samples, self.n_features = X.shape
		self.dim = self.n_clusters * self.n_features  # dimensionality of the search space

		# Determine search domain from data
		self.min_val = np.min(X, axis=0)
		self.max_val = np.max(X, axis=0)

		# Initialize population randomly within the domain
		self.population = self._initialize_population()

		# Initialize fitness values
		self.fitness = np.zeros(self.pop_size)

		# Initialize alpha, beta, and delta positions
		self.alpha_pos = np.zeros(self.dim)
		self.beta_pos = np.zeros(self.dim)
		self.delta_pos = np.zeros(self.dim)
		self.alpha_score = float('inf')
		self.beta_score = float('inf')
		self.delta_score = float('inf')

		# Run the optimization
		self._optimize()

		return self

	def _initialize_population(self):
		"""Initialize population of wolves randomly"""
		population = np.zeros((self.pop_size, self.dim))

		for i in range(self.pop_size):
			# Initialize each wolf as a set of centroids
			for j in range(self.n_clusters):
				# Randomly select data points as initial centroids
				idx = np.random.randint(0, self.n_samples)
				for k in range(self.n_features):
					population[i, j * self.n_features + k] = self.X[idx, k]

		return population

	def _evaluate_fitness(self):
		"""Evaluate fitness for all wolves"""
		for i in range(self.pop_size):
			# Reshape wolf position to centroids
			centroids = self.population[i].reshape(self.n_clusters, self.n_features)

			# Calculate distances from all points to all centroids
			distances = np.zeros((self.n_samples, self.n_clusters))
			for j in range(self.n_clusters):
				distances[:, j] = np.sum((self.X - centroids[j]) ** 2, axis=1)

			# Assign points to nearest centroid
			labels = np.argmin(distances, axis=1)

			# Calculate inertia (sum of squared distances to closest centroid)
			inertia = 0
			for j in range(self.n_clusters):
				if np.sum(labels == j) > 0:  # Avoid empty clusters
					inertia += np.sum(np.min(distances[labels == j], axis=1))

			self.fitness[i] = inertia

			# Update alpha, beta, delta
			if self.fitness[i] < self.alpha_score:
				self.delta_score = self.beta_score
				self.delta_pos = self.beta_pos.copy()
				self.beta_score = self.alpha_score
				self.beta_pos = self.alpha_pos.copy()
				self.alpha_score = self.fitness[i]
				self.alpha_pos = self.population[i].copy()

				# Update best_centroids and best_score
				self.best_centroids = self.alpha_pos.reshape(self.n_clusters, self.n_features)
				self.best_score = self.alpha_score
			elif self.fitness[i] < self.beta_score:
				self.delta_score = self.beta_score
				self.delta_pos = self.beta_pos.copy()
				self.beta_score = self.fitness[i]
				self.beta_pos = self.population[i].copy()
			elif self.fitness[i] < self.delta_score:
				self.delta_score = self.fitness[i]
				self.delta_pos = self.population[i].copy()

	def _cluster_population(self):
		"""Cluster the population using K-means (K=3) and return centroids ordered by fitness"""
		# Restructure the population for clustering
		flat_population = self.population.reshape(self.pop_size * self.n_clusters, self.n_features)

		# Apply K-means
		kmeans = KMeans(n_clusters=3, n_init=10)
		labels = kmeans.fit_predict(flat_population)
		centroids = kmeans.cluster_centers_

		# Calculate fitness for each centroid configuration
		centroid_configs = []
		centroid_fitness = []

		# For each centroid from K-means, create a complete set of n_clusters centroids
		for i in range(3):  # K=3 for clustering the wolves
			# Get points in this cluster
			cluster_points = flat_population[labels == i]

			if len(cluster_points) >= self.n_clusters:
				# Randomly sample n_clusters points from this cluster
				indices = np.random.choice(len(cluster_points), self.n_clusters, replace=False)
				config = cluster_points[indices].flatten()

				# Evaluate fitness
				# Reshape for evaluation
				test_centroids = config.reshape(self.n_clusters, self.n_features)

				# Calculate distances and inertia
				distances = np.zeros((self.n_samples, self.n_clusters))
				for j in range(self.n_clusters):
					distances[:, j] = np.sum((self.X - test_centroids[j]) ** 2, axis=1)

				labels_test = np.argmin(distances, axis=1)
				inertia = 0
				for j in range(self.n_clusters):
					if np.sum(labels_test == j) > 0:
						inertia += np.sum(np.min(distances[labels_test == j], axis=1))

				centroid_configs.append(config)
				centroid_fitness.append(inertia)

		# Sort by fitness
		if centroid_configs:  # Check if we have any valid configurations
			centroid_configs = np.array(centroid_configs)
			centroid_fitness = np.array(centroid_fitness)
			sorted_indices = np.argsort(centroid_fitness)

			return centroid_configs[sorted_indices]
		else:
			# If we couldn't create configurations, return current alpha, beta, delta
			return np.array([self.alpha_pos, self.beta_pos, self.delta_pos])

	def _update_positions(self, a):
		"""Update positions of wolves based on alpha, beta, delta"""
		for i in range(self.pop_size):
			for d in range(self.dim):
				# For each dimension

				# Calculate distance components
				r1 = np.random.random()
				r2 = np.random.random()
				A1 = 2 * a * r1 - a
				C1 = 2 * r2

				r1 = np.random.random()
				r2 = np.random.random()
				A2 = 2 * a * r1 - a
				C2 = 2 * r2

				r1 = np.random.random()
				r2 = np.random.random()
				A3 = 2 * a * r1 - a
				C3 = 2 * r2

				# Distances to leaders
				D_alpha = abs(C1 * self.alpha_pos[d] - self.population[i, d])
				D_beta = abs(C2 * self.beta_pos[d] - self.population[i, d])
				D_delta = abs(C3 * self.delta_pos[d] - self.population[i, d])

				# Position components
				X1 = self.alpha_pos[d] - A1 * D_alpha
				X2 = self.beta_pos[d] - A2 * D_beta
				X3 = self.delta_pos[d] - A3 * D_delta

				# Weighted position update
				self.population[i, d] = (3 * X1 + 2 * X2 + X3) / 6

			# Ensure the wolf stays within bounds
			for j in range(self.n_clusters):
				for k in range(self.n_features):
					idx = j * self.n_features + k
					self.population[i, idx] = max(min(self.population[i, idx],
													  self.max_val[k]),
												  self.min_val[k])

	def _optimize(self):
		"""Run the KCGWO algorithm"""
		# Store initial positions for visualization
		self.all_positions.append(self.population.copy())

		# Initial fitness evaluation
		self._evaluate_fitness()

		# Main loop
		for t in tqdm(range(self.max_iter)):
			# Store the best score in each iteration for the convergence curve
			self.convergence_curve[t] = self.alpha_score

			# Calculate a for this iteration (linearly decreasing from 2 to 0)
			a = 2 - 2 * (t / self.max_iter)

			# With 50% probability, use K-means clustering to guide the search
			if np.random.random() > 0.5:
				# Use K-means to cluster the population
				sorted_centroids = self._cluster_population()

				if len(sorted_centroids) >= 3:  # Make sure we have enough centroids
					# Use the clustered centroids as alpha, beta, delta
					self.alpha_pos = sorted_centroids[0].copy()
					self.beta_pos = sorted_centroids[1].copy()
					self.delta_pos = sorted_centroids[2].copy()

			# Update positions of all wolves
			self._update_positions(a)

			# Reevaluate fitness and update alpha, beta, delta
			self._evaluate_fitness()

			# Store positions for visualization
			self.all_positions.append(self.population.copy())

	def predict(self, X=None):
		"""
		Predict the closest cluster for each sample in X

		Parameters:
		- X: array-like of shape (n_samples, n_features)
			  If None, uses the data passed to fit

		Returns:
		- labels: array of shape (n_samples)
		"""
		if X is None:
			X = self.X

		# Calculate distances from all points to all centroids
		distances = np.zeros((X.shape[0], self.n_clusters))
		for j in range(self.n_clusters):
			distances[:, j] = np.sum((X - self.best_centroids[j]) ** 2, axis=1)

		# Assign points to nearest centroid
		return np.argmin(distances, axis=1)

	def get_centroids(self):
		"""Return the cluster centroids"""
		return self.best_centroids
