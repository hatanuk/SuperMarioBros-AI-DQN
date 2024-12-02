from PyQt5 import QtGui, QtCore, QtWidgets
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QPen, QColor, QBrush
import sys
from typing import List
from neural_network import *
from mario import Mario
from config import Config

class NeuralNetworkViz(QtWidgets.QWidget):
    def __init__(self, parent, mario: Mario, size, config: Config, nn_params):
        super().__init__(parent)
        self.mario = mario
        self.size = size
        self.config = config
        self.nn_params = nn_params
        self.scale_factor = 1.0
        self.neuron_locations = {}
        self.set_scale_factor()
        self.show()

    def set_scale_factor(self):
        # Calculate the scale factor based on the widget's size
        base_width, base_height = 800, 600  # Define base dimensions for scaling
        self.scale_factor = min(self.width() / base_width, self.height() / base_height)

        # Update parameters
        self.horizontal_distance_between_layers = int(50 * self.scale_factor)
        self.vertical_distance_between_nodes = int(10 * self.scale_factor)
        self.neuron_radius = int(self.config.Graphics.neuron_radius * self.scale_factor)
        self.tile_size = (int(self.config.Graphics.tile_size[0] * self.scale_factor),
                          int(self.config.Graphics.tile_size[1] * self.scale_factor))
        self.x_offset = int(150 * self.scale_factor + 16 // 2 * self.tile_size[0] + 5)
        self.y_offset = int(5 + 15 * self.tile_size[1] + 5)

    def resizeEvent(self, event):
        # Recalculate scale factor and repaint
        self.set_scale_factor()
        self.update()

    def show_network(self, painter: QtGui.QPainter):
        painter.setRenderHints(QtGui.QPainter.Antialiasing)
        painter.setRenderHints(QtGui.QPainter.HighQualityAntialiasing)
        painter.setRenderHint(QtGui.QPainter.TextAntialiasing)
        painter.setRenderHint(QtGui.QPainter.SmoothPixmapTransform)
        painter.setPen(QPen(Qt.black, 1.0, Qt.SolidLine))
        horizontal_space = int(20 * self.scale_factor)  # Scaled space between nodes

        layer_nodes = self.mario.network.layer_nodes
        default_offset = self.x_offset
        h_offset = self.x_offset
        v_offset = self.y_offset + int(50 * self.scale_factor)
        inputs = self.mario.inputs_as_array

        out = self.mario.network.feed_forward(inputs)
        active_outputs = np.where(out > 0.5)[0]
        max_n = self.size[0] // (2 * self.neuron_radius + horizontal_space)

        # Draw nodes (scaled)
        for layer, num_nodes in enumerate(layer_nodes[1:], 1):
            h_offset = int((((max_n - num_nodes)) * (2 * self.neuron_radius + horizontal_space)) / 2)
            activations = None
            if layer > 0:
                activations = self.mario.network.params['A' + str(layer)]

            for node in range(num_nodes):
                x_loc = node * (self.neuron_radius * 2 + horizontal_space) + h_offset
                y_loc = v_offset
                t = (layer, node)
                if t not in self.neuron_locations:
                    self.neuron_locations[t] = (int(x_loc + self.neuron_radius), int(y_loc))

                painter.setBrush(QtGui.QBrush(Qt.white, Qt.NoBrush))
                if layer == 0:  # Input layer
                    painter.setBrush(QtGui.QBrush(Qt.green if inputs[node, 0] > 0 else Qt.white))
                elif layer > 0 and layer < len(layer_nodes) - 1:  # Hidden layers
                    saturation = max(min(activations[node, 0], 1.0), 0.0)
                    painter.setBrush(QtGui.QBrush(QtGui.QColor.fromHslF(125 / 239, saturation, 120 / 240)))
                elif layer == len(layer_nodes) - 1:  # Output layer
                    text = ('U', 'D', 'L', 'R', 'A', 'B')[node]
                    painter.drawText(
                        int(h_offset + node * (self.neuron_radius * 2 + horizontal_space)),
                        int(v_offset + 2 * self.neuron_radius + 2 * self.neuron_radius),
                        text
                    )
                    painter.setBrush(QtGui.QBrush(Qt.green if node in active_outputs else Qt.white))

                painter.drawEllipse(
                    int(x_loc), int(y_loc),
                    int(self.neuron_radius * 2), int(self.neuron_radius * 2)
                )

            v_offset += int(150 * self.scale_factor)

        # Reset horizontal offset for the weights
        h_offset = default_offset

        # Draw weights
        for l in range(2, len(layer_nodes)):
            weights = self.mario.network.params['W' + str(l)]
            prev_nodes = weights.shape[1]
            curr_nodes = weights.shape[0]
            for prev_node in range(prev_nodes):
                for curr_node in range(curr_nodes):
                    painter.setPen(QtGui.QPen(Qt.blue if weights[curr_node, prev_node] > 0 else Qt.red))
                    start = self.neuron_locations[(l-1, prev_node)]
                    end = self.neuron_locations[(l, curr_node)]
                    painter.drawLine(
                        int(start[0]), int(start[1] + self.neuron_radius * 2),
                        int(end[0]), int(end[1])
                    )
