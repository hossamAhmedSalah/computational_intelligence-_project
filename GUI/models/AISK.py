import numpy as np
class AISK:
	"""
	Artificial Immune System K-means clustering algorithm

	Parameters:
	-----------
	n_clusters : int
		Number of clusters to find (K)
	memory_size : int
		Size of memory cell (M)
	remainder_size : int
		Size of remainder cell (P_r)
	selection_size : int
		Number of top antibodies to select for cloning (n)
	clone_factor : int
		Factor for cloning (f)
	rho : float
		Parameter for mutation rate calculation
	max_iter : int
		Maximum number of iterations
	random_state : int or None
		Random seed for reproducibility
	"""

	def __init__(self, n_clusters=3, memory_size=10, remainder_size=20, selection_size=5,
				 clone_factor=3, rho=1.0, max_iter=50, random_state=None):
		self.n_clusters = n_clusters
		self.memory_size = memory_size
		self.remainder_size = remainder_size
		self.selection_size = selection_size
		self.clone_factor = clone_factor
		self.rho = rho
		self.max_iter = max_iter
		self.random_state = random_state
		self.population_size = memory_size + remainder_size
		self.rng = np.random.RandomState(random_state)

		# Results will be stored here
		self.best_antibody = None
		self.labels_ = None
		self.cluster_centers_ = None
		self.inertia_ = None
		self.affinities_history = []

	def _initialize_antibodies(self, X):
		"""Initialize population of antibodies with random centroids"""
		n_features = X.shape[1]

		# Initialize antibody population
		antibodies = []

		# Calculate min/max for each feature to constrain random initialization
		min_vals = X.min(axis=0)
		max_vals = X.max(axis=0)

		# Generate antibodies
		for _ in range(self.population_size):
			# Each antibody is K cluster centroids
			centroids = np.zeros((self.n_clusters, n_features))
			for k in range(self.n_clusters):
				# Generate random centroids within the data bounds
				centroids[k] = min_vals + self.rng.random(n_features) * (max_vals - min_vals)
			antibodies.append(centroids)

		return np.array(antibodies)

	def _calculate_affinity(self, X, antibody):
		"""Calculate affinity (fitness) for a single antibody using Equation 3"""
		# Assign each point to nearest centroid
		distances = np.zeros((X.shape[0], self.n_clusters))
		for k in range(self.n_clusters):
			# Calculate Euclidean distance from each point to centroid k
			diff = X - antibody[k]
			distances[:, k] = np.sqrt(np.sum(diff ** 2, axis=1))

		# Assign each point to the closest centroid
		labels = np.argmin(distances, axis=1)

		# Calculate SED (Sum of Euclidean Distances) - Equation 4
		sed = 0
		for k in range(self.n_clusters):
			cluster_points = X[labels == k]
			if cluster_points.shape[0] > 0:  # Skip empty clusters
				sed += np.sum(np.linalg.norm(cluster_points - antibody[k], axis=1))

		# Calculate affinity (Equation 3)
		affinity = 1 / (1 + sed)

		return affinity, labels, sed

	def _calculate_all_affinities(self, X, antibodies):
		"""Calculate affinities for all antibodies"""
		affinities = np.zeros(len(antibodies))
		all_labels = []
		all_seds = []

		for i, antibody in enumerate(antibodies):
			affinity, labels, sed = self._calculate_affinity(X, antibody)
			affinities[i] = affinity
			all_labels.append(labels)
			all_seds.append(sed)

		return affinities, all_labels, all_seds

	def _clonal_selection(self, antibodies, affinities):
		"""Select top n antibodies and clone them proportionally to affinity"""
		# Find indices of top n antibodies
		top_indices = np.argsort(affinities)[-self.selection_size:]

		# Get top n antibodies and their affinities
		top_antibodies = antibodies[top_indices]
		top_affinities = affinities[top_indices]

		# Calculate number of clones per antibody (Equation 7)
		total_clones = self.selection_size * self.clone_factor
		clone_counts = np.round(total_clones * (top_affinities / np.sum(top_affinities))).astype(int)

		# Make sure we have exactly total_clones by adjusting last count
		clone_counts[-1] = total_clones - np.sum(clone_counts[:-1])

		# Create clones
		clones = []
		clone_parents = []  # Keep track of which antibody produced each clone

		for i, (antibody, count) in enumerate(zip(top_antibodies, clone_counts)):
			for _ in range(count):
				clones.append(antibody.copy())
				clone_parents.append(top_indices[i])

		return np.array(clones), np.array(clone_parents), top_indices

	def _affinity_maturation(self, X, clones, affinities, clone_parents):
		"""Mutate clones according to their parent's affinity (Equations 8-9)"""
		# Normalize affinities (Equation 8)
		parent_affinities = affinities[clone_parents]
		max_affinity = np.max(affinities)
		normalized_affinities = parent_affinities / max_affinity

		# Calculate mutation rates (Equation 8)
		mutation_rates = np.exp(-self.rho * normalized_affinities)

		# Apply Gaussian mutation (Equation 9)
		mutated_clones = clones.copy()

		for i, (clone, mutation_rate) in enumerate(zip(clones, mutation_rates)):
			# Generate random number r ~ U(0,1)
			r = self.rng.random()

			# Apply mutation if r < mutation_rate
			if r < mutation_rate:
				# Apply Gaussian mutation to each centroid element
				noise = mutation_rate * self.rng.normal(0, 1, clone.shape)
				mutated_clones[i] = clone + noise

		return mutated_clones

	def _update_centroids(self, X, antibodies, all_labels):
		"""Update centroids for each antibody (Equation 11)"""
		updated_antibodies = antibodies.copy()

		for i, (antibody, labels) in enumerate(zip(antibodies, all_labels)):
			for k in range(self.n_clusters):
				cluster_points = X[labels == k]
				if cluster_points.shape[0] > 0:  # Skip empty clusters
					updated_antibodies[i][k] = np.mean(cluster_points, axis=0)

		return updated_antibodies

	def fit(self, X):
		"""Fit the AISK clustering model to the data"""
		X = np.asarray(X)

		# Initialize antibody population
		antibodies = self._initialize_antibodies(X)

		# Main loop (Step 12)
		for iteration in range(self.max_iter):
			# Calculate affinities (Step 3)
			affinities, all_labels, all_seds = self._calculate_all_affinities(X, antibodies)
			self.affinities_history.append(np.max(affinities))

			# Select and clone antibodies (Steps 4-5)
			clones, clone_parents, top_indices = self._clonal_selection(antibodies, affinities)

			# Apply affinity maturation (Step 6)
			mutated_clones = self._affinity_maturation(X, clones, affinities, clone_parents)

			# Calculate affinities for mutated clones
			clone_affinities, clone_labels, clone_seds = self._calculate_all_affinities(X, mutated_clones)

			# Update centroids using K-means approach (Steps 7-9)
			updated_clones = self._update_centroids(X, mutated_clones, clone_labels)

			# Recalculate affinities after centroid update
			updated_affinities, updated_labels, updated_seds = self._calculate_all_affinities(X, updated_clones)

			# Update memory (Step 10) - select best M antibodies
			all_candidates = np.vstack([antibodies, updated_clones])
			all_candidate_affinities = np.concatenate([affinities, updated_affinities])
			all_candidate_labels = all_labels + updated_labels
			all_candidate_seds = all_seds + updated_seds

			# Select the best memory_size antibodies
			best_indices = np.argsort(all_candidate_affinities)[-self.memory_size:]
			memory_antibodies = all_candidates[best_indices]
			memory_affinities = all_candidate_affinities[best_indices]
			memory_labels = [all_candidate_labels[i] for i in best_indices]
			memory_seds = [all_candidate_seds[i] for i in best_indices]

			# Generate new remainder antibodies (Step 11)
			new_remainder = self._initialize_antibodies(X)[:self.remainder_size]

			# Combine memory and remainder for new population
			antibodies = np.vstack([memory_antibodies, new_remainder])

			# Print iteration progress
			if (iteration + 1) % 10 == 0:
				print(f"Iteration {iteration + 1}, Best affinity: {np.max(memory_affinities):.6f}")

		# Get the best antibody after all iterations
		best_idx = np.argmax(memory_affinities)
		self.best_antibody = memory_antibodies[best_idx]
		self.labels_ = memory_labels[best_idx]
		self.cluster_centers_ = self.best_antibody
		self.inertia_ = memory_seds[best_idx]

		return self

	def predict(self, X):
		"""Predict cluster labels for new data points"""
		X = np.asarray(X)

		# Calculate distances to each centroid
		distances = np.zeros((X.shape[0], self.n_clusters))
		for k in range(self.n_clusters):
			diff = X - self.cluster_centers_[k]
			distances[:, k] = np.sqrt(np.sum(diff ** 2, axis=1))

		# Assign points to closest centroid
		labels = np.argmin(distances, axis=1)

		return labels
