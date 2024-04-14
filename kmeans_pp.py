import argparse
import math

import numpy as np
import pandas as pd
import mykmeanssp



def kmeansPlus(k,eps , max_iter, file_name_1, file_name_2):
    DataPoints = ReadData(file_name_1, file_name_2)

    # convert to numpy
    DataPoints = DataPoints.values
    # convert to array
    DataPoints = np.array(DataPoints)

    rows = DataPoints.shape[0]
    dimension = DataPoints.shape[1]

    if (k >= rows) or (k <= 1):
        print("Invalid number of clusters!")
        return -1

    cetroids = np.ndarray((k, dimension), float)
    cetroids_index = np.ndarray(k, int)

    init_centroids(DataPoints, cetroids, cetroids_index, k, rows)
    # convert to list
    LDataPoints = DataPoints.tolist()
    Lcentroids = cetroids.tolist()

    centroids = mykmeanssp.fit(LDataPoints, Lcentroids, rows, dimension, k, max_iter, eps)

    centroids = np.array(centroids)
    centroids = np.round(centroids, 4)
    # print
    print_centroids(centroids)
    # the last row is empty
    print()

    return None


def print_centroids(centroids):
    for i in range(len(centroids)):
        centroid = centroids[i]
        for j in range(len(centroid)):
            if j != (len(centroid) - 1):
                print(str(centroid[j]) + ",", end="")
            else:
                if i == len(centroids) - 1:
                    print(centroid[j], end="")
                else:
                    print(centroid[j])


def ReadData(filename1, filename2):
    DataPoints1 = pd.read_csv(filename1, header=None)
    DataPoints2 = pd.read_csv(filename2, header=None)
    DataPoints = pd.merge(DataPoints1, DataPoints2, on=0)
    DataPoints.sort_values(by=[0], inplace=True, ascending=True)
    DataPoints = DataPoints.drop(columns=0)
    return DataPoints


def init_centroids(DataPoints, centroids, centroids_index, k: int, rows: int):
    total = 0
    min = float("inf")
    D = np.zeros(rows)
    np.random.seed(0)
    new = np.random.choice(rows)
    centroids_index[0] = new
    centroids[0] = DataPoints[new]
    i = 1
    while i < k:
        for j in range(0, rows):
            min = float("inf")
            for l in range(i):
                dist1 = dist(DataPoints[j], centroids[l])
                if dist1 < min:
                    min = dist1
            total = total - (D[j] - min)
            D[j] = min
        P = np.divide(D, total)
        index1 = np.random.choice(rows, p=P)
        centroids_index[i] = index1
        centroids[i] = DataPoints[index1]
        i += 1
    print(','.join(str(i) for i in centroids_index), flush=True)


def dist(point1, point2):
    total = 0
    d = len(point1)
    for i in range(d):
        total += (point1[i] - point2[i]) ** 2

    return math.sqrt(total)


def init_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("k", type=int, help="K is the number of clusters")
    parser.add_argument("Max_Iter", nargs="?", type=int, help="The maximum number of the K-means algorithm")
    parser.add_argument("eps", type=float, help="epsilon")
    parser.add_argument("file_name_1", type=str, help="The path to file 1 which contains N observations")
    parser.add_argument("file_name_2", type=str, help="The path to file 2 which contains N observations")
    args = parser.parse_args()
    k = args.k
    Max_Iter = args.Max_Iter
    eps = args.eps
    file_name_1 = args.file_name_1
    file_name_2 = args.file_name_2

    if Max_Iter:
        if (Max_Iter <= 1) or (Max_Iter >= 1000):
            print("Invalid maximum iteration!")
            return -1
    else:
        Max_Iter = 300
    if k:
        if k < 1:
            print("Invalid number of clusters!")
            return -1
    else:
        print("An Error Has Occurred")
        return -1
    if (not file_name_1) or (not file_name_2):
        print("An Error Has Occurred")
        return -1
    kmeansPlus(k,eps, Max_Iter, file_name_1, file_name_2)


init_args()
