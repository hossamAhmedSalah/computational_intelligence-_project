import gradio as gr
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from PIL import Image

import cv2

from models.KCGWO import KCGWO
from models.AISK import AISK
from models.hACA import hACA
from models.SAPSOGKPP import SAPSO_GKPP
from models.ACDE import ACDE
from models.GeneticKMeans import GeneticKMeans
from models.SimulatedAnnealing import SimulatedAnnealingClustering

from helping_functions.preprocess_dataset import run_preprocessing


def compute_inertia(data, labels, centers):
	inertia = 0
	for i, center in enumerate(centers):
		cluster_points = data[labels == i]
		inertia += np.sum((cluster_points - center) ** 2)
	return inertia


def extract_points_from_canvas(image):
	"""
	Extracts non-white pixel coordinates from an RGBA image input (either PIL or numpy array).
	Converts to grayscale first, then finds all non-white (drawn) points.
	"""
	# If image is a NumPy array (as from Gradio Sketchpad or Canvas)
	if isinstance(image, np.ndarray):
		if image.shape[2] == 4:  # RGBA
			gray = cv2.cvtColor(image, cv2.COLOR_RGBA2GRAY)
		else:  # RGB
			gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
	else:
		# If it's a PIL Image
		gray = np.array(image.convert("L"))

	# Threshold to detect non-white pixels
	_, thresh = cv2.threshold(gray, 250, 255, cv2.THRESH_BINARY_INV)

	# Extract coordinates of the white-on-black drawing
	points = np.column_stack(np.where(thresh > 0))

	return thresh, points


def Apply_Clustering_algorithm(image: Image.Image, k: int, algorithm: str, population_size: int = 128,
							   max_iter: int = 100, memory_size: int = 10, reminder_size: int = 20,
							   selection_size: int = 5, clone_factor: int = 3, rho: float = 1.0, num_ants: int = 100,
							   grid_size: int = 128, k1: int = 3, k2: int = 0.3, no_particles: int = 50,
							   w_max: float = 0.9, w_min: float = 0.5, c1: float = 1.49, c2: float = 1.49,
							   initial_temperature: int = 100, cooling_rate: float = 0.95,
							   mutation_rate: float = 0.1, crossover_rate: float = 0.8):
	_, points = extract_points_from_canvas(image["composite"])
	if points is None or len(points) < k:
		fig = plt.figure()
		plt.text(0.5, 0.5, "Not enough points", ha='center')
		return fig

	data = np.array(points)
	model = None
	labels = []
	inertia = None
	centroids = None
	convergence_curve = None
	fig1 = None
	fig2 = None

	if algorithm == "KMeans":
		model = KMeans(n_clusters=k, n_init=10).fit(data)
		labels = model.labels_
		inertia = model.inertia_
		centroids = model.cluster_centers_
	elif algorithm == "KCGWO":
		model = KCGWO(n_clusters=k, pop_size=population_size, max_iter=max_iter)
		model.fit(data)
		labels = model.predict(data)
		inertia = model.best_score
		centroids = model.get_centroids()
		convergence_curve = model.convergence_curve
	elif algorithm == "AISK":
		model = AISK(n_clusters=k, memory_size=memory_size, remainder_size=reminder_size,
					 selection_size=selection_size, clone_factor=clone_factor, rho=rho,
					 max_iter=max_iter, random_state=42)
		model.fit(data)
		labels = model.predict(data)
		inertia = model.inertia_
		centroids = model.cluster_centers_
		convergence_curve = model.affinities_history

	elif algorithm == "hACA":
		model = hACA(num_ants=num_ants, grid_size=grid_size, num_objects=len(data), object_data=data)
		model.run(max_iter, k1=k1, k2=k2)

	elif algorithm == "SAPSOGK++":
		model = SAPSO_GKPP(n_clusters=k, n_particles=no_particles, max_iter=max_iter,
						   w_max=w_max, w_min=w_min, c1=c1, c2=c2, initial_temperature=initial_temperature,
						   cooling_rate=cooling_rate, random_state=41)
		model.fit(data)
		labels = model.predict(data)
		inertia = model.inertia_
		centroids = model.cluster_centers_

	elif algorithm == "ACDE":
		model = ACDE(data=data, k_max=k, population_size=population_size, t_max=max_iter)
		centroids, labels, _, inertia, _ = model.evolve()

	elif algorithm == "GeneticKMeans":
		model = GeneticKMeans(n_clusters=k, dim=data.shape[1], pop_size=population_size, mutation_rate=mutation_rate,
							  crossover_rate=crossover_rate)
		model.initialize_population(data)
		convergence_curve, chromosome = model.run_evolution(data, n_generations=max_iter)
		centroids = model.decode_chromosome(chromosome)
		labels = np.argmin(model.calculate_distances(data, centroids), axis=1)
		inertia = compute_inertia(data, labels, centroids)

	elif algorithm == "SimulatedAnnealing":
		model = SimulatedAnnealingClustering(k=k, max_iter=max_iter, initial_temp=initial_temperature,
											 cooling_rate=cooling_rate)
		model.fit(data)
		labels = model.predict(data)
		centroids = model.centroids_
		print(centroids)
		inertia = model.cost_

	if algorithm != "hACA":
		fig1, ax1 = plt.subplots()
		for i in range(k):
			ax1.scatter(data[labels == i, 0], data[labels == i, 1], label=f'Cluster {i + 1}', alpha=0.7)
		ax1.scatter(centroids[:, 0], centroids[:, 1], c='black', marker='X', s=200, label='Centroids')
		ax1.set_title(f"{algorithm} with k={k} with inertia= {inertia}")
		ax1.legend()
		ax1.invert_yaxis()
	if algorithm == "KCGWO":
		if convergence_curve is not None:
			fig2, ax2 = plt.subplots()
			ax2.plot(convergence_curve)
			ax2.set_title("KCGWO Convergence Curve")
			plt.xlabel('Iteration')
			plt.ylabel('Objective Function Value (Inertia)')
			plt.grid(True)
	elif algorithm == "AISK":
		if convergence_curve is not None:
			fig2, ax2 = plt.subplots()
			ax2.plot(convergence_curve)
			ax2.set_title("AISK Convergence Curve")
			plt.xlabel('Iteration')
			plt.ylabel('Best Affinity')
			plt.grid(True)
	elif algorithm == "hACA":
		fig1, ax1 = model.visualize_grid()
		_, fig2, ax2 = model.visualize_clusters()
	elif algorithm == "GeneticKMeans":
		fig2, ax2 = plt.subplots()
		ax2.plot(convergence_curve)
		ax2.set_title("GeneticKMeans Convergence Curve")
		plt.xlabel('Iteration')
		plt.ylabel('Best Fitness')
		plt.grid(True)
	return fig1, fig2


def Apply_Clustering_algorithm_csv(preprocessed_df: pd.DataFrame, idx1: int, idx2: int, k: int, algorithm: str,
								   population_size: int = 128, max_iter: int = 100, memory_size: int = 10,
								   reminder_size: int = 20, selection_size: int = 5, clone_factor: int = 3,
								   rho: float = 1.0, num_ants: int = 100, grid_size: int = 128, k1: int = 3,
								   k2: int = 0.3, no_particles: int = 50, w_max: float = 0.9, w_min: float = 0.5,
								   c1: float = 1.49, c2: float = 1.49, initial_temperature: int = 100,
								   cooling_rate: float = 0.95, mutation_rate: float = 0.1,
								   crossover_rate: float = 0.8):
	data = preprocessed_df.values
	columns_name = preprocessed_df.columns

	model = None
	labels = []
	inertia = None
	centroids = None
	convergence_curve = None
	fig1 = None
	fig2 = None

	if algorithm == "KMeans":
		model = KMeans(n_clusters=k, n_init=10).fit(data)
		labels = model.labels_
		inertia = model.inertia_
		centroids = model.cluster_centers_
	elif algorithm == "KCGWO":
		model = KCGWO(n_clusters=k, pop_size=population_size, max_iter=max_iter)
		model.fit(data)
		labels = model.predict(data)
		inertia = model.best_score
		centroids = model.get_centroids()
		convergence_curve = model.convergence_curve
	elif algorithm == "AISK":
		model = AISK(n_clusters=k, memory_size=memory_size, remainder_size=reminder_size,
					 selection_size=selection_size, clone_factor=clone_factor, rho=rho,
					 max_iter=max_iter, random_state=42)
		model.fit(data)
		labels = model.predict(data)
		inertia = model.inertia_
		centroids = model.cluster_centers_
		convergence_curve = model.affinities_history

	elif algorithm == "hACA":
		data_2D = PCA(n_components=2).fit_transform(data)
		model = hACA(num_ants=num_ants, grid_size=grid_size, num_objects=len(data_2D), object_data=data_2D)
		model.run(max_iter, k1=k1, k2=k2)

	elif algorithm == "SAPSOGK++":
		model = SAPSO_GKPP(n_clusters=k, n_particles=no_particles, max_iter=max_iter,
						   w_max=w_max, w_min=w_min, c1=c1, c2=c2, initial_temperature=initial_temperature,
						   cooling_rate=cooling_rate, random_state=41)
		model.fit(data)
		labels = model.predict(data)
		inertia = model.inertia_
		centroids = model.cluster_centers_

	elif algorithm == "ACDE":
		model = ACDE(data=data, k_max=k, population_size=population_size, t_max=max_iter)
		centroids, labels, _, inertia, _ = model.evolve()

	elif algorithm == "GeneticKMeans":
		model = GeneticKMeans(n_clusters=k, dim=data.shape[1], pop_size=population_size, mutation_rate=mutation_rate,
							  crossover_rate=crossover_rate)
		model.initialize_population(data)
		convergence_curve, chromosome = model.run_evolution(data, n_generations=max_iter)
		centroids = model.decode_chromosome(chromosome)
		labels = np.argmin(model.calculate_distances(data, centroids), axis=1)
		inertia = compute_inertia(data, labels, centroids)

	elif algorithm == "SimulatedAnnealing":
		model = SimulatedAnnealingClustering(k=k, max_iter=max_iter, initial_temp=initial_temperature,
											 cooling_rate=cooling_rate)
		model.fit(data)
		labels = model.predict(data)
		centroids = model.centroids_
		inertia = model.cost_

	if algorithm != "hACA":
		fig1, ax1 = plt.subplots()
		for i in range(k):
			ax1.scatter(data[labels == i, 0], data[labels == i, 1], label=f'Cluster {i + 1}', alpha=0.7)
		ax1.scatter(centroids[:, idx1], centroids[:, idx2], c='black', marker='X', s=200, label='Centroids')
		ax1.set_title(f"{algorithm} with k={k} with inertia= {inertia}")
		plt.xlabel(columns_name[idx1])
		plt.ylabel(columns_name[idx2])
		ax1.legend()
		ax1.invert_yaxis()
	if algorithm == "KCGWO":
		if convergence_curve is not None:
			fig2, ax2 = plt.subplots()
			ax2.plot(convergence_curve)
			ax2.set_title("KCGWO Convergence Curve")
			plt.xlabel('Iteration')
			plt.ylabel('Objective Function Value (Inertia)')
			plt.grid(True)
	elif algorithm == "AISK":
		if convergence_curve is not None:
			fig2, ax2 = plt.subplots()
			ax2.plot(convergence_curve)
			ax2.set_title("AISK Convergence Curve")
			plt.xlabel('Iteration')
			plt.ylabel('Best Affinity')
			plt.grid(True)
	if algorithm == "hACA":
		fig1, ax1 = model.visualize_grid()
		_, fig2, ax2 = model.visualize_clusters()

	elif algorithm == "GeneticKMeans":
		fig2, ax2 = plt.subplots()
		ax2.plot(convergence_curve)
		ax2.set_title("GeneticKMeans Convergence Curve")
		plt.xlabel('Iteration')
		plt.ylabel('Best Fitness')
		plt.grid(True)
	return fig1, fig2


def change_visibility(choice: str):
	if choice in ["KCGWO"]:
		return gr.update(visible=True), gr.update(visible=True), gr.update(visible=False), gr.update(
			visible=False), gr.update(visible=False), gr.update(visible=False), gr.update(visible=False), gr.update(
			visible=False), gr.update(visible=False), gr.update(visible=False), gr.update(visible=False), gr.update(
			visible=False), gr.update(visible=False), gr.update(visible=False), gr.update(visible=False), gr.update(
			visible=False), gr.update(visible=False), gr.update(visible=False), gr.update(visible=False), gr.update(
			visible=False)
	elif choice in ["KMeans"]:
		return gr.update(visible=False), gr.update(visible=False), gr.update(visible=False), gr.update(
			visible=False), gr.update(visible=False), gr.update(visible=False), gr.update(visible=False), gr.update(
			visible=False), gr.update(visible=False), gr.update(visible=False), gr.update(visible=False), gr.update(
			visible=False), gr.update(visible=False), gr.update(visible=False), gr.update(visible=False), gr.update(
			visible=False), gr.update(visible=False), gr.update(visible=False), gr.update(visible=False), gr.update(
			visible=False)
	elif choice in ["AISK"]:
		return gr.update(visible=False), gr.update(visible=True), gr.update(visible=True), gr.update(
			visible=True), gr.update(visible=True), gr.update(visible=True), gr.update(visible=True), gr.update(
			visible=False), gr.update(visible=False), gr.update(visible=False), gr.update(visible=False), gr.update(
			visible=False), gr.update(visible=False), gr.update(visible=False), gr.update(visible=False), gr.update(
			visible=False), gr.update(visible=False), gr.update(visible=False), gr.update(visible=False), gr.update(
			visible=False)
	elif choice in ["hACA"]:
		return gr.update(visible=False), gr.update(visible=True), gr.update(visible=False), gr.update(
			visible=False), gr.update(visible=False), gr.update(visible=False), gr.update(visible=False), gr.update(
			visible=True), gr.update(visible=True), gr.update(visible=True), gr.update(visible=True), gr.update(
			visible=False), gr.update(visible=False), gr.update(visible=False), gr.update(visible=False), gr.update(
			visible=False), gr.update(visible=False), gr.update(visible=False), gr.update(visible=False), gr.update(
			visible=False)

	elif choice in ["SAPSOGK++"]:
		return gr.update(visible=False), gr.update(visible=True), gr.update(visible=False), gr.update(
			visible=False), gr.update(visible=False), gr.update(visible=False), gr.update(visible=False), gr.update(
			visible=False), gr.update(visible=False), gr.update(visible=False), gr.update(visible=False), gr.update(
			visible=True), gr.update(visible=True), gr.update(visible=True), gr.update(visible=True), gr.update(
			visible=True), gr.update(visible=True), gr.update(visible=True), gr.update(visible=False), gr.update(
			visible=False)

	elif choice in ["ACDE"]:
		return gr.update(visible=True, label="Population Size"), gr.update(visible=True), gr.update(
			visible=False), gr.update(
			visible=False), gr.update(visible=False), gr.update(visible=False), gr.update(visible=False), gr.update(
			visible=False), gr.update(visible=False), gr.update(visible=False), gr.update(visible=False), gr.update(
			visible=False), gr.update(visible=False), gr.update(visible=False), gr.update(visible=False), gr.update(
			visible=False), gr.update(visible=False), gr.update(visible=False), gr.update(visible=False), gr.update(
			visible=False)

	elif choice in ["GeneticKMeans"]:
		return gr.update(visible=True, label="Population Size"), gr.update(visible=True), gr.update(
			visible=False), gr.update(visible=False), gr.update(visible=False), gr.update(visible=False), gr.update(
			visible=False), gr.update(visible=False), gr.update(visible=False), gr.update(visible=False), gr.update(
			visible=False), gr.update(visible=False), gr.update(visible=False), gr.update(visible=False), gr.update(
			visible=False), gr.update(visible=False), gr.update(visible=False), gr.update(visible=False), gr.update(
			visible=True), gr.update(visible=True)

	elif choice in ["SimulatedAnnealing"]:
		return gr.update(visible=False), gr.update(visible=True), gr.update(visible=False), gr.update(
			visible=False), gr.update(visible=False), gr.update(visible=False), gr.update(visible=False), gr.update(
			visible=False), gr.update(visible=False), gr.update(visible=False), gr.update(visible=False), gr.update(
			visible=False), gr.update(visible=False), gr.update(visible=False), gr.update(visible=False), gr.update(
			visible=False), gr.update(visible=True), gr.update(visible=True), gr.update(visible=False), gr.update(
			visible=False)


with gr.Blocks() as demo:
	with gr.Tab("Test Clustering Algorithms on User graph data"):
		gr.Markdown("## 🖊️ Draw points on the grid (tap or click), then run clustering")

		with gr.Row():
			with gr.Column():
				sketchpad = gr.Sketchpad(canvas_size=(500, 500), label="2D Grid")
				algorithm = gr.Radio(choices=["KMeans", "KCGWO", "AISK", "hACA", "SAPSOGK++", "ACDE", "GeneticKMeans",
											  "SimulatedAnnealing"],
									 value="KMeans",
									 label="Clustering Algorithm")
				k_input = gr.Slider(3, 10, step=1, label="Number of Clusters", value=3)

				# KCGWO
				pop_size = gr.Slider(3, 1000, step=1, label="Number of Wolfs (population)", value=128, visible=False)
				max_iteration = gr.Slider(10, 10000, step=1, label="Maximum number of iterations", value=1000,
										  visible=False)

				# AISK
				memory_size = gr.Slider(1, 20, step=1, label="Memory Size (Store the best solutions)", value=10,
										visible=False)
				reminder_size = gr.Slider(1, 20, step=1, label="Reminder Size (Store Random solutions)", value=20,
										  visible=False)
				selection_size = gr.Slider(1, 10, step=1, label="Selection Size n (to Clone n antibody)", value=5,
										   visible=False)
				clone_factor = gr.Slider(1, 10, step=1, label="Clone Factor f (multiply by n)", value=3,
										 visible=False)
				rho = gr.Slider(1.0, 5.0, step=0.3, label="Mutation Parameter", value=1.0,
								visible=False)

				# hACA
				ant_size = gr.Slider(1, 1000, step=1, label="Number of Ants", value=128,
									 visible=False)
				grid_size = gr.Slider(5, 500, step=1, label="The Grid Size", value=128,
									  visible=False)
				k1 = gr.Slider(0.1, 5, step=0.1, label="k1 The parameter of pickup probability", value=3,
							   visible=False)
				k2 = gr.Slider(0.1, 5, step=0.1, label="k2 The parameter of drop off probability", value=0.3,
							   visible=False)

				# SAPSOGK++
				no_particles = gr.Slider(5, 500, step=1, label="Number of Particles", value=50,
										 visible=False)
				w_max = gr.Slider(0, 3, step=0.1, label="w_max", value=0.9,
								  visible=False)
				w_min = gr.Slider(0, 3, step=0.1, label="w_min", value=0.5,
								  visible=False)
				c1 = gr.Slider(1, 5, step=0.1, label="c1", value=1.49,
							   visible=False)
				c2 = gr.Slider(1, 5, step=0.1, label="c2", value=1.49,
							   visible=False)
				initial_temperature = gr.Slider(25, 500, step=1, label="initial temperature", value=100,
												visible=False)
				cooling_rate = gr.Slider(0, 1, step=0.01, label="cooling rate", value=0.95,
										 visible=False)

				# GeneticKMeans
				dim = gr.Number(label="Dimension", value=2, visible=False)
				mutation_rate = gr.Slider(0, 1, step=0.01, label="mutation rate", value=0.1,
										  visible=False)
				crossover_rate = gr.Slider(0, 1, step=0.01, label="crossover rate", value=0.8,
										   visible=False)

			with gr.Column():
				output_plot = gr.Plot(label="Clustered Output")
				convergence_plot = gr.Plot(label="Convergence Rate")

		with gr.Row():
			run_btn = gr.Button("Run Clustering Algorithm")
			clear_btn = gr.Button("Clear")

		run_btn.click(
			fn=Apply_Clustering_algorithm,
			inputs=[sketchpad, k_input, algorithm, pop_size, max_iteration, memory_size, reminder_size, selection_size,
					clone_factor, rho, ant_size, grid_size, k1, k2, no_particles, w_max, w_min, c1, c2,
					initial_temperature, cooling_rate, mutation_rate, crossover_rate],
			outputs=[output_plot, convergence_plot]
		)
		clear_btn.click(
			fn=lambda: [None, "KMeans", 3, 128, 1000, None, None, 10, 20, 5, 3, 1.0, 128, 128, 3, 0.3, 50, 0.9, 0.5,
						1.49, 1.49, 100, 0.95, 0.1, 0.9],
			outputs=[sketchpad, algorithm, k_input, pop_size, max_iteration, output_plot, convergence_plot, memory_size,
					 reminder_size, selection_size, clone_factor, rho, ant_size, grid_size, k1, k2, no_particles, w_max,
					 w_min, c1, c2, initial_temperature, cooling_rate, mutation_rate, crossover_rate]
		)

		algorithm.change(
			fn=change_visibility,
			inputs=[algorithm],
			outputs=[pop_size, max_iteration, memory_size, reminder_size, selection_size, clone_factor, rho, ant_size,
					 grid_size, k1, k2, no_particles, w_max, w_min, c1, c2, initial_temperature, cooling_rate,
					 mutation_rate, crossover_rate]
		)

	with gr.Tab("Preprocess Uploaded Dataset"):
		gr.Markdown("## 📊 Upload CSV and Apply Preprocessing")

		with gr.Row():
			file_input = gr.File(label="Upload csv Dataset", file_types=['.csv'])

		with gr.Row():
			preprocessed_output = gr.Dataframe(label="Preprocessed Output", wrap=True, interactive=False)

		with gr.Row():
			run_preprocess = gr.Button("Preprocess Dataset")

		gr.Markdown("## Apply the Algorithm you want :)")
		with gr.Row():
			with gr.Column():
				columns_idx1 = gr.Number(label="Choose the first index to plot the result")
				columns_idx2 = gr.Number(label="Choose the second index to plot the result")

				algorithm_csv = gr.Radio(
					choices=["KMeans", "KCGWO", "AISK", "hACA", "SAPSOGK++", "ACDE", "GeneticKMeans",
							 "SimulatedAnnealing"],
					value="KMeans",
					label="Clustering Algorithm")
				k_input_csv = gr.Slider(3, 10, step=1, label="Number of Clusters", value=3)

				# KCGWO
				pop_size_csv = gr.Slider(3, 1000, step=1, label="Number of Wolfs (population)", value=128,
										 visible=False)
				max_iteration_csv = gr.Slider(10, 10000, step=1, label="Maximum number of iterations", value=1000,
											  visible=False)

				# AISK
				memory_size_csv = gr.Slider(1, 20, step=1, label="Memory Size (Store the best solutions)", value=10,
											visible=False)
				reminder_size_csv = gr.Slider(1, 20, step=1, label="Reminder Size (Store Random solutions)", value=20,
											  visible=False)
				selection_size_csv = gr.Slider(1, 10, step=1, label="Selection Size n (to Clone n antibody)", value=5,
											   visible=False)
				clone_factor_csv = gr.Slider(1, 10, step=1, label="Clone Factor f (multiply by n)", value=3,
											 visible=False)
				rho_csv = gr.Slider(1.0, 5.0, step=0.3, label="Mutation Parameter", value=1.0,
									visible=False)

				# hACA
				ant_size_csv = gr.Slider(1, 1000, step=1, label="Number of Ants", value=128,
										 visible=False)
				grid_size_csv = gr.Slider(5, 500, step=1, label="The Grid Size", value=128,
										  visible=False)
				k1_csv = gr.Slider(0.1, 5, step=0.1, label="k1 The parameter of pickup probability", value=3,
								   visible=False)
				k2_csv = gr.Slider(0.1, 5, step=0.1, label="k2 The parameter of drop off probability", value=0.3,
								   visible=False)

				# SAPSOGK++
				no_particles_csv = gr.Slider(5, 500, step=1, label="Number of Particles", value=50,
											 visible=False)
				w_max_csv = gr.Slider(0, 3, step=0.1, label="w_max", value=0.9,
									  visible=False)
				w_min_csv = gr.Slider(0, 3, step=0.1, label="w_min", value=0.5,
									  visible=False)
				c1_csv = gr.Slider(1, 5, step=0.1, label="c1", value=1.49,
								   visible=False)
				c2_csv = gr.Slider(1, 5, step=0.1, label="c2", value=1.49,
								   visible=False)
				initial_temperature_csv = gr.Slider(25, 500, step=1, label="initial temperature", value=100,
													visible=False)
				cooling_rate_csv = gr.Slider(0, 1, step=0.01, label="cooling rate", value=0.95,
											 visible=False)

				# GeneticKMeans
				mutation_rate_csv = gr.Slider(0, 1, step=0.01, label="mutation rate", value=0.1,
											  visible=False)
				crossover_rate_csv = gr.Slider(0, 1, step=0.01, label="crossover rate", value=0.8,
											   visible=False)

			with gr.Column():
				output_plot_csv = gr.Plot(label="Clustered Output")
				convergence_plot_csv = gr.Plot(label="Convergence Rate")

		with gr.Row():
			run_btn_csv = gr.Button("Run Clustering Algorithm")
			clear_btn_csv = gr.Button("Clear")

	run_preprocess.click(
		fn=run_preprocessing,
		inputs=[file_input],
		outputs=[preprocessed_output]
	)

	run_btn_csv.click(
		fn=Apply_Clustering_algorithm_csv,
		inputs=[preprocessed_output, columns_idx1, columns_idx2, k_input_csv, algorithm_csv, pop_size_csv,
				max_iteration_csv, memory_size_csv, reminder_size_csv, selection_size_csv, clone_factor_csv, rho_csv,
				ant_size_csv, grid_size_csv, k1_csv, k2_csv, no_particles_csv, w_max_csv, w_min_csv, c1_csv, c2_csv,
				initial_temperature_csv, cooling_rate_csv, mutation_rate_csv, crossover_rate_csv],
		outputs=[output_plot_csv, convergence_plot_csv]
	)

	clear_btn_csv.click(
		fn=lambda: [None, "KMeans", 3, 128, 1000, None, None, 10, 20, 5, 3, 1.0, 128, 128, 3, 0.3, 50, 0.9, 0.5,
					1.49, 1.49, 100, 0.95, 0.1, 0.9],
		outputs=[preprocessed_output, algorithm_csv, k_input_csv, pop_size_csv, max_iteration_csv, output_plot_csv,
				 convergence_plot_csv, memory_size_csv, reminder_size_csv, selection_size_csv, clone_factor_csv,
				 rho_csv, ant_size_csv, grid_size_csv, k1_csv, k2_csv, no_particles_csv, w_max_csv, w_min_csv, c1_csv,
				 c2_csv, initial_temperature_csv, cooling_rate_csv, mutation_rate_csv, crossover_rate_csv]
	)

	algorithm_csv.change(
		fn=change_visibility,
		inputs=[algorithm_csv],
		outputs=[pop_size_csv, max_iteration_csv, memory_size_csv, reminder_size_csv, selection_size_csv,
				 clone_factor_csv, rho_csv, ant_size_csv, grid_size_csv, k1_csv, k2_csv, no_particles_csv, w_max_csv,
				 w_min_csv, c1_csv, c2_csv, initial_temperature_csv, cooling_rate_csv, mutation_rate_csv,
				 crossover_rate_csv]
	)

demo.launch()
