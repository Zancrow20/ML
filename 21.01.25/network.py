
import random

import numpy as np

class Network(object):

    def __init__(self, sizes):
        """sizes - массив, содержащий количество
        нейронов для соответствующего из уровней НС"""
        self.num_layers = len(sizes)
        self.sizes = sizes
        self.biases = [np.random.randn(y, 1) for y in sizes[1:]]
        self.weights = [np.random.randn(y, x)
                        for x, y in zip(sizes[:-1], sizes[1:])]

    def feedforward(self, a):
        """Возвращает выходные данные сети при входных данных 'a'"""
        for b, w in zip(self.biases, self.weights):
            a = sigmoid(np.dot(w, a) + b)
        return a

    def SGD(self, training_data, epochs, mini_batch_size, eta,
            test_data=None):
        """Обучает сеть при помощи мини-пакетов и стохастического градиентного спуска.
        training_data – список кортежей "(x, y)", обозначающих обучающие входные данные и желаемые выходные.
        Если test_data задан, тогда сеть будет оцениваться относительно проверочных данных после каждой эпохи,
        и будет выводиться текущий прогресс. """

        n = len(training_data)
        for j in range(epochs):
            random.shuffle(training_data)
            mini_batches = [
                training_data[k:k + mini_batch_size]
                for k in range(0, n, mini_batch_size)]
            for mini_batch in mini_batches:
                self.update_mini_batch(mini_batch, eta)
            if test_data:
                print(f"Epoch {j}: {self.evaluate(test_data)} / {len(test_data)}")
            else:
                print(f"Epoch {j} complete")

    def update_mini_batch(self, mini_batch, eta):
        """Обновляет веса и смещения сети, применяя градиентный спуск с использованием обратного распространения
        к одному мини-пакету.
        mini_batch – это список кортежей (x, y),
        а eta – скорость обучения."""
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]
        for x, y in mini_batch:
            delta_nabla_b, delta_nabla_w = self.backprop(x, y)
            nabla_b = [nb+dnb for nb, dnb in zip(nabla_b, delta_nabla_b)]
            nabla_w = [nw+dnw for nw, dnw in zip(nabla_w, delta_nabla_w)]
        self.weights = [w-(eta/len(mini_batch))*nw
                        for w, nw in zip(self.weights, nabla_w)]
        self.biases = [b-(eta/len(mini_batch))*nb
                       for b, nb in zip(self.biases, nabla_b)]

    def backprop(self, x, y):
        """Возвращает кортеж ``(nabla_b, nabla_w)``, представляющий градиент для функции стоимости C_x.
        ``nabla_b`` и ``nabla_w`` - послойные списки массивов numpy, похожие на ``self.biases`` and ``self.weights``."""
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]
        # прямой проход
        activation = x
        activations = [x]  # список для послойного хранения активаций
        zs = []  # список для послойного хранения z-векторов
        for b, w in zip(self.biases, self.weights):
            z = np.dot(w, activation) + b
            zs.append(z)
            activation = sigmoid(z)
            activations.append(activation)
        # обратный проход
        delta = self.cost_derivative(activations[-1], y) * \
                sigmoid_prime(zs[-1])
        nabla_b[-1] = delta
        nabla_w[-1] = np.dot(delta, activations[-2].transpose())
        for l in range(2, self.num_layers):
            z = zs[-l]
            sp = sigmoid_prime(z)
            delta = np.dot(self.weights[-l + 1].transpose(), delta) * sp
            nabla_b[-l] = delta
            nabla_w[-l] = np.dot(delta, activations[-l - 1].transpose())
        return nabla_b, nabla_w

    def evaluate(self, test_data):
        """Возвращает количество проверочных входных данных, для которых нейросеть выдаёт правильный результат.
        Выходные данные сети – это номер нейрона в последнем слое с наивысшим уровнем активации."""
        test_results = [(np.argmax(self.feedforward(x)), y)
                        for (x, y) in test_data]
        return sum(int(x == y) for (x, y) in test_results)

    def cost_derivative(self, output_activations, y):
        """Возвращает вектор частных производных (чп C_x / чп a) для выходных активаций."""
        return output_activations - y

    def predict_number(self, image):
        output = self.feedforward(image)
        return np.argmax(output)

def sigmoid(z):
    """Сигмоида"""
    return 1.0 / (1.0 + np.exp(-z))

def sigmoid_prime(z):
    """Производная сигмоиды."""
    return sigmoid(z)*(1-sigmoid(z))