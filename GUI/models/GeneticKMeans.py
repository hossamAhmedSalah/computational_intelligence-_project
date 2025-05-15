import numpy as np
from typing import List, Tuple

import random


class GeneticKMeans:
	def __init__(self,
				 n_clusters: int = 3,
				 dim: int = 4,
				 pop_size: int = 50,
				 mutation_rate: float = 0.1,
				 crossover_rate: float = 0.8):
		self.n_clusters = n_clusters
		self.dim = dim
		self.chromosome_length = n_clusters * dim
		self.pop_size = pop_size
		self.mutation_rate = mutation_rate
		self.crossover_rate = crossover_rate
		self.population = []

	def initialize_population(self, data: np.ndarray) -> None:
		"""Initialize population with random centroids within data bounds"""
		min_bounds = np.min(data, axis=0)
		max_bounds = np.max(data, axis=0)

		self.population = []
		for _ in range(self.pop_size):
			chromosome = []
			for _ in range(self.n_clusters):
				centroid = [np.random.uniform(min_bounds[d], max_bounds[d])
							for d in range(self.dim)]
				chromosome.extend(centroid)
			self.population.append(np.array(chromosome))

	def fitness_function(self, chromosome: np.ndarray, data: np.ndarray) -> float:
		"""Calculate fitness using sum of squared distances"""
		centroids = self.decode_chromosome(chromosome)
		distances = self.calculate_distances(data, centroids)
		cluster_assignments = np.argmin(distances, axis=1)

		# Calculate within-cluster sum of squares
		wcss = 0
		for k in range(self.n_clusters):
			if sum(cluster_assignments == k) > 0:
				cluster_points = data[cluster_assignments == k]
				centroid = centroids[k]
				wcss += np.sum((cluster_points - centroid) ** 2)

		return 1 / (wcss + 1e-10)  # Higher fitness for lower WCSS

	def decode_chromosome(self, chromosome: np.ndarray) -> np.ndarray:
		"""Convert flat chromosome to centroids matrix"""
		return chromosome.reshape(self.n_clusters, self.dim)

	def calculate_distances(self, data: np.ndarray, centroids: np.ndarray) -> np.ndarray:
		"""Calculate distances between data points and centroids"""
		distances = np.zeros((len(data), self.n_clusters))
		for k in range(self.n_clusters):
			distances[:, k] = np.sum((data - centroids[k]) ** 2, axis=1)
		return distances

	def single_point_crossover(self, parent1: np.ndarray, parent2: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
		"""Perform single-point crossover"""
		if random.random() < self.crossover_rate:
			point = random.randint(1, self.chromosome_length - 1)
			child1 = np.concatenate([parent1[:point], parent2[point:]])
			child2 = np.concatenate([parent2[:point], parent1[point:]])
			return child1, child2
		return parent1.copy(), parent2.copy()

	def uniform_mutation(self, chromosome: np.ndarray, data: np.ndarray) -> np.ndarray:
		"""Perform uniform mutation"""
		mutated = chromosome.copy()
		min_bounds = np.min(data, axis=0)
		max_bounds = np.max(data, axis=0)

		for i in range(self.chromosome_length):
			if random.random() < self.mutation_rate:
				dim_idx = i % self.dim
				mutated[i] = np.random.uniform(min_bounds[dim_idx], max_bounds[dim_idx])

		return mutated

	def tournament_selection(self, population: List[np.ndarray],
							 fitness_values: List[float],
							 tournament_size: int = 3) -> np.ndarray:
		"""Select parent using tournament selection"""
		tournament_idx = random.sample(range(len(population)), tournament_size)
		tournament_fitness = [fitness_values[i] for i in tournament_idx]
		winner_idx = tournament_idx[np.argmax(tournament_fitness)]
		return population[winner_idx].copy()

	def run_evolution(self, data: np.ndarray, n_generations: int = 50) -> Tuple[List[float], np.ndarray]:
		"""Run the complete genetic algorithm evolution process"""
		best_fitness_history = []

		# Initialize population if not already initialized
		if not self.population:
			self.initialize_population(data)

		# Calculate initial fitness for each chromosome
		fitness_values = [self.fitness_function(chromosome, data) for chromosome in self.population]
		best_fitness = max(fitness_values)
		best_chromosome = self.population[fitness_values.index(best_fitness)]
		best_fitness_history.append(best_fitness)

		for generation in range(n_generations):
			new_population = []

			# Elitism: Keep the best chromosome
			elite_idx = fitness_values.index(max(fitness_values))
			new_population.append(self.population[elite_idx])

			# Generate new population
			while len(new_population) < self.pop_size:
				# Selection
				parent1 = self.tournament_selection(self.population, fitness_values)
				parent2 = self.tournament_selection(self.population, fitness_values)

				# Crossover
				child1, child2 = self.single_point_crossover(parent1, parent2)

				# Mutation
				child1 = self.uniform_mutation(child1, data)
				child2 = self.uniform_mutation(child2, data)

				new_population.append(child1)
				if len(new_population) < self.pop_size:
					new_population.append(child2)

			# Update population
			self.population = new_population

			# Calculate fitness for new population
			fitness_values = [self.fitness_function(chromosome, data) for chromosome in self.population]

			# Update best solution
			generation_best_fitness = max(fitness_values)
			if generation_best_fitness > best_fitness:
				best_fitness = generation_best_fitness
				best_chromosome = self.population[fitness_values.index(best_fitness)]

			best_fitness_history.append(best_fitness)

			# Optional: Print progress
			if generation % 10 == 0:
				print(f"Generation {generation}: Best fitness = {best_fitness}")

		return best_fitness_history, best_chromosome
