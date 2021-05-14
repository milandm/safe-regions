# Reference: https://github.com/miloharper/visualise-neural-network

from matplotlib import pyplot
from math import cos, sin, atan


class Neuron:
    def __init__(self, x, y):
        self.x = x
        self.y = y

    def draw(self):
        circle = pyplot.Circle((self.x, self.y), radius=neuron_radius, fill=True, color='r')
        pyplot.gca().add_patch(circle)


class Layer:
    def __init__(self, network, number_of_neurons):
        self.previous_layer = self.__get_previous_layer(network)
        self.y = self.__calculate_layer_y_position()
        self.neurons = self.__intialise_neurons(number_of_neurons)

    def __intialise_neurons(self, number_of_neurons):
        neurons = []
        x = self.__calculate_left_margin_so_layer_is_centered(number_of_neurons)
        for iteration in range(number_of_neurons):
            neuron = Neuron(x, self.y)
            neurons.append(neuron)
            x += horizontal_distance_between_neurons
        return neurons

    def __calculate_left_margin_so_layer_is_centered(self, number_of_neurons):
        return horizontal_distance_between_neurons * (number_of_neurons_in_widest_layer - number_of_neurons) / 2

    def __calculate_layer_y_position(self):
        if self.previous_layer:
            return self.previous_layer.y + vertical_distance_between_layers
        else:
            return 0

    def __get_previous_layer(self, network):
        if len(network.layers) > 0:
            return network.layers[-1]
        else:
            return None

    def __line_between_two_neurons(self, neuron1, neuron2):
        angle = atan((neuron2.x - neuron1.x) / float(neuron2.y - neuron1.y))
        x_adjustment = neuron_radius * sin(angle)
        y_adjustment = neuron_radius * cos(angle)
        line = pyplot.Line2D((neuron1.x - x_adjustment, neuron2.x + x_adjustment), (neuron1.y - y_adjustment, neuron2.y + y_adjustment), linewidth=0.01)
        pyplot.gca().add_line(line)

    def draw(self):
        for neuron in self.neurons:
            neuron.draw()
            # if self.previous_layer:
            #     for previous_layer_neuron in self.previous_layer.neurons:
            #         self.__line_between_two_neurons(neuron, previous_layer_neuron)


class NeuralNetwork:
    def __init__(self):
        self.layers = []

    def add_layer(self, number_of_neurons):
        layer = Layer(self, number_of_neurons)
        self.layers.append(layer)

    def draw(self):
        for layer in self.layers:
            layer.draw()
        pyplot.axis('scaled')
        pyplot.show()


def visualize_in_out(model, draw_connections=False):
    # full screen
    # mng = pyplot.get_current_fig_manager()
    # mng.resize(*mng.window.maxsize())

    # set figure height and width in inches
    dpi = 100
    width = 2000
    height = 2000
    # count num_layers and w
    num_layers = 5
    w = 512
    pyplot.figure(figsize=(width / dpi, height / dpi), dpi=dpi)

    # calculated
    vertical_distance_between_layers = height / num_layers
    horizontal_distance_between_neurons = width / w
    neuron_radius = horizontal_distance_between_neurons / 3
    number_of_neurons_in_widest_layer = w
    network = NeuralNetwork()
    network.add_layer(w)
    network.add_layer(w // 2)
    network.add_layer(w // 4)
    network.add_layer(w // 8)
    network.add_layer(1)
    network.draw()


if __name__ == "__main__":
    vertical_distance_between_layers = 6
    horizontal_distance_between_neurons = 2
    neuron_radius = 0.5
    number_of_neurons_in_widest_layer = 64
    network = NeuralNetwork()
    network.add_layer(64)
    network.add_layer(64)
    network.add_layer(1)
    network.draw()
