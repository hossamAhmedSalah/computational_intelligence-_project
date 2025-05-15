import numpy as np
from scipy.spatial.distance import cdist
from sklearn.metrics import calinski_harabasz_score, silhouette_score


def euclidean_distance(x, y):
	return np.linalg.norm(x - y, axis=1)


def compute_inertia(data, labels, centers):
	inertia = 0
	for i, center in enumerate(centers):
		cluster_points = data[labels == i]
		inertia += np.sum((cluster_points - center) ** 2)
	return inertia


class Chromosome:
	def __init__(self, k_max, dim, data):
		self.k_max = k_max
		self.dim = dim
		self.data = data
		self.thresholds = np.random.rand(k_max)  # thresholds for active/inactive clusters
		self.centers = np.random.rand(k_max, dim) * (np.max(data, axis=0) - np.min(data, axis=0)) + np.min(data,
																										   axis=0)  # multiply the centers(vector of k_max * dim shape of [0,1)

	# by data range to scale and then
	# shift by the range by add min

	def get_active_centers(self):
		active_indices = np.where(self.thresholds > 0.5)[0]  # get the active clusters(cneters) that has high threshold
		return self.centers[active_indices], active_indices  # return the centers with its corresponding indeces

	def assign_clusters(self):
		active_centers, active_indices = self.get_active_centers()
		if len(active_centers) < 2:  # If fewer than 2 clusters are active, reinitialize. We need at least 2 clusters to do clustering.
			self.reinitialize_centers()
			active_centers, active_indices = self.get_active_centers()

		dists = cdist(self.data, active_centers)  # we get the distance between centers and data points
		# dicts ==> Each row represents a data point's distance to each active cluster center ;shape=[n_samples, n_active_clusters]
		labels = np.argmin(dists,
						   axis=1)  # Assign each point to the closest active center; labels ==> Returns index of the closest center for each point
		for idx in range(len(active_centers)):
			if np.sum(labels == idx) < 2:
				self.reinitialize_centers()
				return self.assign_clusters()  # recursive ==> try to assign clusters again if at least one active cluster has < 2 points
		return labels, active_centers

	def reinitialize_centers(self):
		indices = np.random.choice(len(self.data), self.k_max,
								   replace=False)  # select k_max random numbers in range len(self.data) without replacement to be indices of centers
		self.centers = self.data[indices]
		self.thresholds = np.random.rand(self.k_max)
		if np.sum(self.thresholds > 0.5) < 2:
			chosen = np.random.choice(self.k_max, 2, replace=False)
			self.thresholds[chosen] = np.random.uniform(0.5, 1.0,
														size=2)  # Force existing of ar least 2 active clusters


class ACDE:
	def __init__(self, data, k_max=5, population_size=10, t_max=50):
		self.data = data
		self.k_max = k_max
		self.dim = data.shape[1]
		self.population_size = population_size
		self.t_max = t_max
		self.population = [Chromosome(k_max, self.dim, data) for _ in range(population_size)]
		self.Cr_max = 1.0
		self.Cr_min = 0.5

	def fitness(self, labels):
		if len(np.unique(labels)) < 2:
			return -np.inf
		return calinski_harabasz_score(self.data, labels)

	def mutate(self, target_idx, F):
		indices = [i for i in range(self.population_size) if i != target_idx]
		x1, x2, x3 = np.random.choice(indices, 3, replace=False)
		base, a, b = self.population[x1], self.population[x2], self.population[x3]

		mutant_thresholds = base.thresholds + F * (a.thresholds - b.thresholds)
		mutant_centers = base.centers + F * (a.centers - b.centers)

		mutant_thresholds = np.clip(mutant_thresholds, 0, 1)
		return mutant_thresholds, mutant_centers

	def crossover(self, target, mutant_thresholds, mutant_centers, Cr):
		trial = Chromosome(self.k_max, self.dim, self.data)
		trial.thresholds = np.where(np.random.rand(self.k_max) < Cr, mutant_thresholds, target.thresholds)
		mask = np.random.rand(*target.centers.shape) < Cr
		trial.centers = np.where(mask, mutant_centers, target.centers)
		return trial

	def evolve(self):
		best_fitness = -np.inf
		best_solution = None
		best_labels = None

		for t in range(self.t_max):
			for i in range(self.population_size):
				target = self.population[i]

				# Adaptive F and CR
				F = 0.5 * (1 + np.random.rand())  # Range (0.5, 1)
				Cr = (self.Cr_max - self.Cr_min) * (self.t_max - t) / self.t_max + self.Cr_min

				mutant_thresholds, mutant_centers = self.mutate(i, F)
				trial = self.crossover(target, mutant_thresholds, mutant_centers, Cr)

				try:
					labels, _ = trial.assign_clusters()
					score = self.fitness(labels)
				except:
					score = -np.inf

				target_labels, _ = target.assign_clusters()
				target_score = self.fitness(target_labels)

				if score > target_score:
					self.population[i] = trial
					if score > best_fitness:
						best_fitness = score
						best_solution = trial.get_active_centers()[0]
						best_labels = labels

			if best_labels is not None:
				n_clusters = len(np.unique(best_labels))
			else:
				n_clusters = 0

		# print(f"Iter {t+1}/{self.t_max} | Best Fitness: {best_fitness:.4f} | F: {F:.4f} | CR: {Cr:.4f} | Clusters: {n_clusters}")

		inertia = compute_inertia(self.data, best_labels, best_solution)
		silhouette = silhouette_score(self.data, best_labels)
		return best_solution, best_labels, best_fitness, inertia, silhouette
