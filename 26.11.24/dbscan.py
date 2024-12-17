import numpy as np
import pygame
from implementation import Point
from implementation import MYDBSCAN

def add_near_points(p: Point):
    points_list = []
    for i in range(np.random.randint(2, 5)):
        x = np.random.uniform(-25, 25) + p.x
        y = np.random.uniform(-25, 25) + p.y
        points_list.append(Point(x, y, INIT_POINT_COLOR))
    return points_list

# pygame setup
pygame.init()
screen = pygame.display.set_mode((1280, 720), flags=pygame.RESIZABLE)
screen.fill("white")
clock = pygame.time.Clock()

running = True
radius = 5
distance_delta = 20
is_drawing = False
points = []

INIT_POINT_COLOR = 'black'
COLORS = ['purple', 'green', 'gray', 'blue', 'cyan', 'pink', 'purple', 'brown']

clusters = []
noise = []

while running:
    pygame.display.flip()
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            pygame.quit()
            exit()

        if event.type == pygame.WINDOWSIZECHANGED:
            screen.fill("white")
            for p in points:
                # перерисовать с учетом в dbscan, в том случае, если уже кластеризовали, но переместили
                pygame.draw.circle(screen, p.color, (p.x, p.y), radius)

        if event.type == pygame.MOUSEBUTTONDOWN:
            if event.button == pygame.BUTTON_LEFT:
                is_drawing = True

        if event.type == pygame.MOUSEBUTTONUP:
            if event.button == pygame.BUTTON_LEFT:
                is_drawing = False

        if event.type == pygame.MOUSEMOTION and is_drawing:
            my_point = Point(*event.pos, INIT_POINT_COLOR)
            if len(points) == 0 or my_point.distance(points[-1]) >= distance_delta:
                pygame.draw.circle(screen, my_point.color, event.pos, radius)
                new_points = add_near_points(my_point)
                for new_point in new_points:
                    pygame.draw.circle(screen, new_point.color, (new_point.x, new_point.y), radius)
                    points.append(new_point)
                points.append(my_point)

        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_SPACE or event.key == pygame.K_RETURN:
                dbscan = MYDBSCAN(eps=30, min_samples=4)
                clusters, noise = dbscan.fit(points)

            if event.key == pygame.K_SPACE:
                for cluster_id, cluster in enumerate(clusters):
                    color = COLORS[cluster_id % len(COLORS)]
                    for point in cluster:
                        point.color = color
                        pygame.draw.circle(screen, point.color, (point.x, point.y), radius)

                for point in noise:
                    point.color = INIT_POINT_COLOR
                    pygame.draw.circle(screen, INIT_POINT_COLOR, (point.x, point.y), radius)

            if event.key == pygame.K_RETURN:
                for cluster_id, cluster in enumerate(clusters):
                    for point in cluster:
                        pygame.draw.circle(screen, point.color, (point.x, point.y), radius)

                for point in noise:
                    point.color = 'red'
                    pygame.draw.circle(screen, point.color, (point.x, point.y), radius)

    pygame.display.update()

    clock.tick(60)  # limits FPS to 60

pygame.quit()
