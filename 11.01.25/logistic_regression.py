import numpy as np
from sklearn.linear_model import LogisticRegression
import pygame

class Point:
    def __init__(self, x, y, color):
        self.x = x
        self.y = y
        self.color = color

    def distance(self, another_point):
        return np.sqrt((another_point.x-self.x)**2 + (another_point.y-self.y)**2)

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

lr: LogisticRegression = LogisticRegression()

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

            if event.button == pygame.BUTTON_RIGHT:
                pygame.draw.circle(screen, BLUE, event.pos, radius)
                points.append(Point(*event.pos, BLUE))

            #отрисовываем
            if event.button == 2:
                pos = event.pos
                color = lr.predict([pos])[0]
                # print(color)
                pygame.draw.rect(screen, color, rect=pygame.Rect(pos[0]-2, pos[1]-2, 6, 6))

        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_SPACE:
                lr.fit([[p.x, p.y] for p in points], [p.color for p in points])

                coef = lr.coef_
                a, b = coef[0][0], coef[0][1]
                c = lr.intercept_[0]

                x_start, x_end = 0, 1100
                y_start = (a * x_start + c) / (-b)
                y_end = (a * x_end + c) / (-b)

                pygame.draw.line(screen, 'black', start_pos=(x_start, y_start), end_pos=(x_end, y_end))


    pygame.display.update()

    clock.tick(60)

pygame.quit()
