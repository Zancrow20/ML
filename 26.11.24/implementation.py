from typing import List
from collections import deque

import numpy as np

class Point:
    def __init__(self, x, y, color):
        self.x = x
        self.y = y
        self.color = color

    def distance(self, another_point):
        return np.sqrt((another_point.x-self.x)**2 + (another_point.y-self.y)**2)

class MYDBSCAN:
    def __init__(self, eps, min_samples):
        self.eps = eps
        self.min_samples = min_samples

    def find_neighbours(self, points: List[Point], point: Point):
        return [p for p in points if p.distance(point) < self.eps]

    def expand_cluster(self, points, point, neighbours, cluster, visited, noise):
        cluster.append(point)
        queue = deque(neighbours)
        point.color = 'green'

        while queue:
            current_point: Point = queue.popleft()
            if current_point not in visited:
                visited.add(current_point)
                new_neighbors = self.find_neighbours(points, current_point)
                if len(new_neighbors) >= self.min_samples:
                    current_point.color = 'green'
                    queue.extend(new_neighbors)
                else:
                    current_point.color = 'yellow'
            if current_point not in cluster:
                cluster.append(current_point)
                if current_point in noise:
                    noise.remove(current_point)
                    current_point.color = 'yellow'

    def fit(self, points):
        clusters = []  # Кластеры
        noise = []  # Шумовые (красные) точки
        visited = set()  # Посещенные точки

        for p in points:
            if p in visited:
                continue
            visited.add(p)

            neighbours = self.find_neighbours(points, p)
            if len(neighbours) < self.min_samples:
                noise.append(p)
            else:
                new_cluster = []
                clusters.append(new_cluster)
                self.expand_cluster(points, p, neighbours, new_cluster, visited, noise)

        return clusters, noise

    @staticmethod
    def distance(p1: Point, p2: Point):
        return p1.distance(p2)
