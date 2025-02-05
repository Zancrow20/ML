import numpy as np
from matplotlib import pyplot as plt

import mnist_loader

training_data, validation_data, test_data = mnist_loader.load_data_wrapper()

# Создание НС
import network
net = network.Network([784, 30, 10])
net.SGD(training_data, 4, 10, 3.0, test_data=test_data)

# Предобработка тестовых данных
digits_test = {0: [], 1: [], 2: [], 3: [], 4: [], 5: [], 6: [], 7: [], 8: [], 9: []}
for x, y in test_data:
  digits_test[y].append(x.reshape(28, 28))

def combine_digits(digits):
    images = []

    for i, digit in enumerate(digits):
        index = np.random.randint(0, len(digits_test[digit]))
        pil_image = digits_test[digit][index]
        images.append(pil_image)

    final_image = np.hstack(images)

    position = 1
    number = 0
    for i in range(len(digits) - 1, -1, -1):
        number += digits[i] * position
        position *= 10

    return final_image, number

def generate_digits(min_digits=3, max_digits=5):
  low = 10 ** (min_digits - 1)
  high = (10 ** max_digits) - 1
  return np.random.randint(low, high)

# Создание картинок
DATASET_SIZE_TEST = 20

test_x, test_y = [], []

for _ in range(DATASET_SIZE_TEST):
    image, number = combine_digits(generate_digits())
    test_x.append(image)
    test_y.append(number)

from sklearn.cluster import DBSCAN
import cv2

index = np.random.randint(0, len(test_x))
img = test_x[index]

height, width = img.shape

coordinates = np.column_stack(np.where(img > 0))

db = DBSCAN(eps=5, min_samples=10).fit(coordinates)
labels = db.labels_

digits = []
for label in set(labels):
    if label == -1:
        continue

    cluster_points = coordinates[labels == label]
    x_min, y_min = cluster_points.min(axis=0)
    x_max, y_max = cluster_points.max(axis=0)

    x_min, x_max = max(x_min - 10, 0), min(x_max + 10, height)
    y_min, y_max = max(y_min - 10, 0), min(y_max + 10, width)

    digit = img[x_min:x_max, y_min:y_max]

    digit = cv2.resize(digit, dsize=(28, 28), interpolation=cv2.INTER_CUBIC)
    digits.append(digit)

predicted_digits = []
for i in range(len(digits)):
    predicted_digits.append(net.predict_number(digits[i].reshape(784,1)))

predicted_number = 0
position = 1

for i in range(len(predicted_digits) - 1, -1, -1):
  predicted_number += predicted_digits[i] * position
  position *= 10

plt.imshow(test_x[index], cmap="gray")
plt.show()
print(f'Предсказание: {predicted_number}')
print(f'Реальный результат: {test_y[index]}')