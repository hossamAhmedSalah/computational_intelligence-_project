# computational_intelligence-_project
Clustering-Based Customer Segmentation with CI/EC Algorithms

> This is a brief overview. For a more detailed explanation, please refer to the [full documentation](docs/CI_documentation.pdf).



>This repository contains the implementation and evaluation of various Computational Intelligence (CI) and Evolutionary Computation (EC) approaches for solving customer segmentation problems using clustering techniques. Our goal was to explore and compare multiple optimization paradigms for improving clustering performance on real-world datasets.

## About the Problem
>Customer segmentation helps businesses understand and group customers based on behavior, demographics, or other attributes. In this project, we formulate clustering as an optimization problem to enhance segmentation quality using non-traditional metaheuristic methods.

* We explored:
1. Single-objective and multi-objective optimization
2. Constrained and free optimization variants

## Datasets and Preprocessing

### Mall Customer Segmentation Data ⬇️
> This dataset contains information on 200 customers, including attributes like age, gender, annual income, and spending score. It's ideal for practicing clustering algorithms such as K-Means.

https://www.kaggle.com/datasets/vjchoudhary7/customer-segmentation-tutorial-in-python

### Customer Segmentation
> An automobile company has plans to enter new markets with their existing products (P1, P2, P3, P4 and P5). After intensive market research, they’ve deduced that the behavior of new market is similar to their existing market.

https://www.kaggle.com/datasets/vetrirah/customer

![image](https://github.com/user-attachments/assets/092e2e9a-140e-4bf0-9f59-90439e09b248)


## Approaches and Algorithms

### 1. K-Means (Baseline)
> The standard clustering algorithm serves as a baseline for comparison. Benchmarked with traditional evaluation metrics.
### 2. Differential Evolution (DE)
>A population-based stochastic optimizer used to refine cluster centers by minimizing intra-cluster distance. We implemented and reviewed several DE variants and strategies.
### 3. Evolutionary Algorithms (EA)
>We studied multi-objective evolutionary clustering through algorithms like AE-IEMOKC.

### 4. Swarm Intelligence (SI)
> Inspired by social behavior in nature. Key methods:
> * K-means Clustering-based Grey Wolf Optimizer (KCGWO)
> * Hybrid Ant Clustering Algorithm (hACA)

### 5. Artificial Immune Systems (AIS)
> Inspired by the biological immune system
> AISK: Artificial Immune System K-means Clustering, This approach focused on memory cells and immune learning for adaptive clustering.


### 6. Hybrid Methods (SA-PSO-GK++ ) 
> We tried hybrid strategies combining the strengths of multiple metaheuristics to enhance convergence and cluster quality.


## Evaluation Metrics
We assessed clustering performance using:
* Silhouette Score
* Calinski-Harabasz Index
* Inertia (Within-cluster sum of squares)

## Final results 
![image](https://github.com/user-attachments/assets/c7fb0957-eea5-4c21-aa43-f2a97412dfa3)

## GUI 






