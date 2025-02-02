#Муравьиные алгоритмы

#Генетические алгоритмы
# 1) Начальная популяция => (2)
# 2) Скрещивание и/или мутация => (3)
# 3) Селекция => (4)
# 4) Формирование нового поколения => (5)
# 5) Достигнут результат ? да => (6), нет => (2)
# 6) Готово

# Пример
# Диофантовые уравнения
# x + 2y + 3z = 25, x,y,z natural
# x = 1...20
# y = 1...10
# z = 1...7

# 1 поколение
# (1,7,4)       (3,2,5)         (14,4,3)        (12,6,2)
# 27(dist 2)    23(dist 3)      31(dist 6)       30(distance 5)

# Оставляем (оставляем 2 родителей) и скрещиваем (по составу или как придумаешь):
# (1,7,4)       (3,2,5)

# Выбор делается следующим образом:
# 1. случайный + случайный
# 2. случайный + ближайший к первому
# 3. случ. + дальний к первому
# дальность по генотипу (составу) и фенотипу (результату)

# 2 поколение
# (1,2,5) - 5      (3,7,5) - 7     (14,6,2) - 7     (3,6,2) - 6     (3,4,3) - 5     (14,2*,3) -> (14,4,5) - 12
# Проводим селекцию (оставим, только несколько потомков).
# Например, берем результаты и делаем рулетку (русскую) и "убиваем" потомков
# и тд, пока не решим

import random
import numpy as np
import pygame
from typing import List

class Point:
    def __init__(self, x, y, color):
        self.x = x
        self.y = y
        self.color = color

    def distance(self, another_point):
        return np.sqrt((another_point.x-self.x)**2 + (another_point.y-self.y)**2)

class GeneticAlgorithm:
    def __init__(self, pop_size, generations, mutation_rate, selection_rate):
        self.pop_size = pop_size
        self.generations = generations
        self.mutation_rate = mutation_rate
        self.selection_rate = selection_rate

    def _fitness_sort(self, dm, individuals):
        individuals.sort(key=lambda i: self._calculate_path_distance(i, dm))

    def _generate_distance_matrix(self, cities: List[Point]):
        num_cities = len(cities)
        matrix = np.zeros((num_cities, num_cities))

        for i in range(num_cities):
            for k in range(num_cities):
                if i != k:
                    matrix[i][k] = cities[i].distance(cities[k])

        return matrix

    def _calculate_path_distance(self, path, cities_distance_matrix):
        return sum(
            cities_distance_matrix[path[i]][path[i + 1]] for i in range(len(path) - 1)
        ) + cities_distance_matrix[path[-1]][path[0]]

    def _create_initial_population(self, num_cities):
        return [random.sample(range(num_cities), num_cities) for _ in range(self.pop_size)]

    def _select(self, population):
        del population[int(self.pop_size * self.selection_rate):]

    def _crossover(self, population):
        childs = []
        size = len(population[0])
        for _ in range(len(population), self.pop_size):
            p1, p2 = random.sample(population, 2)
            start, end = sorted(random.sample(range(size), 2))
            child = [None] * size

            child[start:end] = p1[start:end]

            pointer = 0
            for city in p2:
                if city not in child:
                    while child[pointer] is not None:
                        pointer += 1
                    child[pointer] = city

            childs.append(child)
        population += childs

    def _mutate(self, population):
        for item in population:
            if random.random() < self.mutation_rate:
                i, j = random.sample(range(len(population[0])), 2)
                item[i], item[j] = item[j], item[i]

    def fit(self, cities: List[Point]):
        distance_matrix = self._generate_distance_matrix(cities)
        num_cities = len(cities)
        population = self._create_initial_population(num_cities)


        for i in range(self.generations):
            self._fitness_sort(distance_matrix, population)
            self._select(population)
            self._crossover(population)
            self._mutate(population)
        self._fitness_sort(distance_matrix, population)
        return population[0], self._calculate_path_distance(population[0], distance_matrix)

# pygame setup
pygame.init()
screen = pygame.display.set_mode((1280, 720), flags=pygame.RESIZABLE)
screen.fill("white")
clock = pygame.time.Clock()

running = True
radius = 5
points: list[Point] = []

INIT_POINT_COLOR = 'black'
RED = 'red'
BLUE = 'blue'

pop_size = 20
generations = 100
mutation_rate = 0.1
selection_rate = 0.8

def draw_arrow(screen, colour, start, end):
    import math
    pygame.draw.line(screen,colour,start,end,2)
    rotation = math.degrees(math.atan2(start[1]-end[1], end[0]-start[0]))+90
    pygame.draw.polygon(screen, (255, 0, 0), ((end[0]+20*math.sin(math.radians(rotation)), end[1]+20*math.cos(math.radians(rotation))), (end[0]+20*math.sin(math.radians(rotation-120)), end[1]+20*math.cos(math.radians(rotation-120))), (end[0]+20*math.sin(math.radians(rotation+120)), end[1]+20*math.cos(math.radians(rotation+120)))))

GA = GeneticAlgorithm(pop_size=pop_size, generations=generations, mutation_rate=mutation_rate, selection_rate=selection_rate)

while running:
    pygame.display.flip()
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            pygame.quit()
            exit()

        if event.type == pygame.WINDOWSIZECHANGED:
            screen.fill("white")
            for p in points:
                pygame.draw.circle(screen, p.color, (p.x, p.y), radius)

        elif event.type == pygame.MOUSEBUTTONDOWN:
            if event.button == pygame.BUTTON_LEFT:
                pygame.draw.circle(screen, RED, event.pos, radius)
                points.append(Point(*event.pos, RED))

        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_SPACE:

                best_path, best_distance = GA.fit(points)
                print("Best path:", best_path)
                print("Best distance:", best_distance)
                for j in range(len(best_path)):
                    point = points[best_path[j]]
                    pygame.draw.circle(screen, 'black', (point.x, point.y), 20, 3)
                    font = pygame.font.SysFont('arial', 50)
                    text = font.render(str(best_path[j]), True, (0, 0, 0))
                    screen.blit(text, (point.x, point.y))
                    pygame.display.update()

                for j in range(len(best_path) - 1):
                    point1 = points[best_path[j]]
                    point2 = points[best_path[j + 1]]
                    draw_arrow(screen, 'black', (point1.x, point1.y), (point2.x, point2.y))
                draw_arrow(screen, 'black', (points[best_path[-1]].x, points[best_path[-1]].y), (points[best_path[0]].x, points[best_path[0]].y))

    pygame.display.update()

    clock.tick(60)

pygame.quit()
