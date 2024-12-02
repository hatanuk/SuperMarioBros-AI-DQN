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
        self.horizontal_distance_between_layers = 50
        self.vertical_distance_between_nodes = 10
        self.neuron_radius = 10  # Default radius; can be scaled dynamically
        self.neuron_locations = {}

        # Determine layer architecture
        self.layer_nodes = nn_params.hidden_layer_architecture + [6]

        self.show()

    def resizeEvent(self, event):
        """Recalculate sizes dynamically when resized."""
        self.neuron_radius = min(self.width(), self.height()) // (2 * max(self.layer_nodes))
        self.horizontal_distance_between_layers = self.width() // (len(self.layer_nodes) + 1)
        self.vertical_distance_between_nodes = self.height() // (max(self.layer_nodes) + 1)
        super().resizeEvent(event)

    def show_network(self, painter: QtGui.QPainter):
        painter.setRenderHints(QtGui.QPainter.Antialiasing | QtGui.QPainter.HighQualityAntialiasing)
        painter.setPen(QPen(Qt.black, 1.0, Qt.SolidLine))

        inputs = self.mario.inputs_as_array
        out = self.mario.network.feed_forward(inputs)
        active_outputs = np.where(out > 0.5)[0]

        # Draw nodes and connections layer by layer
        self.neuron_locations.clear()
        for layer, num_nodes in enumerate(self.layer_nodes):
            for node in range(num_nodes):
                # Calculate positions dynamically
                x = self.horizontal_distance_between_layers * (layer + 1)
                y = self.vertical_distance_between_nodes * (node + 1)
                self.neuron_locations[(layer, node)] = (x, y)

                # Determine activation/color
                if layer == 0:  # Input layer
                    color = Qt.green if inputs[node, 0] > 0 else Qt.white
                elif layer == len(self.layer_nodes) - 1:  # Output layer
                    color = Qt.green if node in active_outputs else Qt.white
                else:  # Hidden layers
                    activations = self.mario.network.params[f'A{layer}']
                    saturation = max(min(activations[node, 0], 1.0), 0.0)
                    color = QColor.fromHslF(125 / 239, saturation, 120 / 240)

                painter.setBrush(QBrush(color))
                painter.drawEllipse(int(x - self.neuron_radius), int(y - self.neuron_radius),
                                    int(2 * self.neuron_radius), int(2 * self.neuron_radius))

                # For output layer, draw labels
                if layer == len(self.layer_nodes) - 1:
                    label = ('U', 'D', 'L', 'R', 'A', 'B')[node]
                    painter.drawText(x + self.neuron_radius, y, label)

        # Draw connections
        for layer in range(1, len(self.layer_nodes)):
            prev_layer_nodes = self.layer_nodes[layer - 1]
            curr_layer_nodes = self.layer_nodes[layer]
            weights = self.mario.network.params[f'W{layer}']
            for prev_node in range(prev_layer_nodes):
                for curr_node in range(curr_layer_nodes):
                    start = self.neuron_locations[(layer - 1, prev_node)]
                    end = self.neuron_locations[(layer, curr_node)]
                    weight = weights[curr_node, prev_node]
                    color = Qt.blue if weight > 0 else Qt.red
                    painter.setPen(QPen(color, abs(weight) * 2))
                    painter.drawLine(start[0], start[1], end[0], end[1])

    def paintEvent(self, event):
        painter = QtGui.QPainter(self)
        try:
            self.show_network(painter)
        finally:
            painter.end()
