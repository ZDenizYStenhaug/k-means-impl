import math
import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

import data_generation

DATASET = data_generation.DATASET

N_SAMPLES = data_generation.N_SAMPLES
N_CENTERS = data_generation.N_CENTERS
MAX_ITERATIONS = 300

iterations = []
all_centers = []
all_labels = []

objective_func_values = []


def main():
    try:
        # read points array from file.
        points = np.loadtxt('data.csv', delimiter=',')
    except FileNotFoundError:
        print("Please first generate data using data_generation.py")
        return
    # check if figures folder exists. Create it if not.
    if not os.path.exists("./figures"):
        os.makedirs("./figures")

    plot_initial_graph(points)
    scikit_kmeans(points)
    k_means(points)


def scikit_kmeans(points):
    """
    Runs the scikit k-means algorithm, then plots the graph with centers and clusters.
    :param points: nparray. the points read from the dataset
    """
    kmeans = KMeans(n_clusters=N_CENTERS)
    kmeans.fit(points)
    plt.scatter(points[:, 0], points[:, 1], c=kmeans.labels_, cmap='Set3', s=10, alpha=0.8)
    plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], c='red', s=25)
    plt.title("Scikit's k-means")
    plt.savefig("./figures/" + DATASET.lower() + "-scikit.png")
    plt.cla()


def k_means(points):
    """
    calculates the centers of the clusters by using k-means algorithm.
    Plots the graph with the centers and the clusters on 1st, 2nd and final iteration.
    :param points:
    """
    # add random centers
    get_random_centers(points)
    # stores the info of the distances of each point to the centers and the cluster the point belongs to.
    centers = all_centers[0]
    for k in range(MAX_ITERATIONS):
        labels = []
        for i in range(len(points)):
            distances = []
            for j in range(len(centers)):
                distances.append(euclidean_distance(points[i], centers[j]))
            labels.append(distances.index(min(distances)))
        all_labels.append(labels)
        objective_func_values.append(get_objective_func_value(points, centers))
        if k == 0 or k == 1 or k == 2:
            # plot the graph
            plot_graph(points, labels, centers, str(k))
        # check if the centers are stable enough to end algorithm
        elif abs(objective_func_values[k - 1] - objective_func_values[k]) < 0.000000000000000000000000001:
            plot_graph(points, labels, centers, "final")
            plot_objective_func()
            print("k-means ended with " + str(k) + " iterations")
            break
        centers = get_new_centers(points, labels, centers)


def get_new_centers(points, labels, centers):
    """
    :param points: nparray. the points read from the dataset
    :param labels: python list. the labels of the points for that iteration.
    :param centers: 2d python list. [n][2]
    """
    new_centers = [[0] * 2 for i in range(N_CENTERS)]
    # get the sum of distances between the points and the center they belong to
    for i in range(len(points)):
        # add the x and y
        center = labels[i]
        new_centers[center][0] = new_centers[center][0] + points[i][0]
        new_centers[center][1] += points[i][1]
    # calculate new centers by dividing the sum to the population of their cluster.
    for j in range(len(new_centers)):
        if labels.count(j) == 0:
            new_centers[j] = centers[j]
            continue
        new_centers[j][0] = new_centers[j][0] / labels.count(j)
        new_centers[j][1] = new_centers[j][1] / labels.count(j)
    all_centers.append(new_centers)
    return new_centers


def get_objective_func_value(points, centers):
    """
    Returns the objective function value of the iteration we are on
    :param points: nparray. the points read from the dataset
    :param centers: 2d python list. [n][2]
    """
    value = 0
    for p in points:
        for c in centers:
            value += euclidean_distance(p, c)
    return value


def euclidean_distance(point1, point2):
    """
    Returns the euclidian distance between two points.
    :param point1: python list.
    :param point2: python list
    """
    return math.sqrt((point2[0] - point1[0]) ** 2 + (point2[1] - point1[1]) ** 2)


def get_random_centers(points):
    """
    Returns initial random centers.
    :param points: nparray. the points read from the dataset
    """
    plt.scatter(points[:, 0], points[:, 1], s=10, alpha=0.8)
    initial_centers = []
    for i in range(N_CENTERS):
        index = np.random.randint(N_SAMPLES)
        x = points[index][0]
        y = points[index][1]
        initial_centers.append([x, y])
        plt.scatter(x, y, c='red', s=25)
    all_centers.append(initial_centers)
    plt.title("Initial centers")
    plt.savefig("./figures/" + DATASET.lower() + "-initial-centers.png")
    plt.cla()


def plot_graph(points, labels, centers, iteration):
    """
    :param points: nparray. the points read from the dataset
    :param labels: python list. the labels of the points for that iteration.
    :param centers: 2d python list. [n][2]
    :param iteration: string. tells what iteration we are on.
    """
    plt.scatter(points[:, 0], points[:, 1], c=labels, cmap='Set3', s=10, alpha=0.8)
    plt.scatter([row[0] for row in centers], [row[1] for row in centers], c='red', s=25)
    plt.title("Iteration-" + iteration)
    plt.savefig("./figures/" + DATASET.lower() + "-iteration" + iteration + ".png")
    plt.cla()


def plot_objective_func():
    """
    Plots the objective function graph using the objective function values.
    """
    x = list(range(len(objective_func_values)))
    plt.plot(x, objective_func_values)
    plt.title("Objective function")
    plt.xlabel("Iteration")
    plt.ylabel("Objective Function Value")
    plt.savefig("./figures/" + DATASET.lower() + "-objective-function.png")
    plt.cla()


def plot_initial_graph(points):
    """
    plots the initial graph after the data has been read from the file.
    :param points: nparray. the points read from the dataset
    """
    plt.scatter(points[:, 0], points[:, 1], s=10, alpha=0.7)
    plt.title("Original Dataset")
    plt.savefig("./figures/" + DATASET.lower() + "-original.png")
    plt.cla()


if __name__ == '__main__':
    main()
