import pytest
from PyQt5.QtWidgets import QApplication
from smb_ai_dqn import MainWindow

@pytest.fixture
def app():
    app = QApplication([])
    yield app
    app.quit()

def test_main_window(app):
    window = MainWindow()
    assert window is not None

def test_make_env():
    window = MainWindow()
    env = window._make_env()

def test_update_dqn():
    window = MainWindow()
    window._update_dqn()
    assert window.mario_DQN.fitness >= 0
    assert window.max_distance_DQN >= 0

def test_update_ga():
    window = MainWindow()
    window._update_ga()
    assert window.mario_GA.fitness >= 0
    assert window.max_distance_GA >= 0

def test_next_individual_or_generation():
    window = MainWindow()
    window._next_individual_or_generation()
    assert window._current_individual == 1

def test_next_generation():
    window = MainWindow()
    window._next_generation()
    assert window._current_individual == 0
    assert window.current_generation == 1

def test_crossover():
    window = MainWindow()
    parent1_weights = parent2_weights = parent1_bias = parent2_bias = [1, 2, 3]
    child1_weights, child2_weights, child1_bias, child2_bias = window._crossover(parent1_weights, parent2_weights, parent1_bias, parent2_bias)
    assert child1_weights != parent1_weights
    assert child2_weights != parent2_weights
    assert child1_bias != parent1_bias
    assert child2_bias != parent2_bias

def test_mutation():
    window = MainWindow()
    child1_weights = child2_weights = child1_bias = child2_bias = [1, 2, 3]
    window._mutation(child1_weights, child2_weights, child1_bias, child2_bias)
    assert child1_weights != [1, 2, 3]
    assert child2_weights != [1, 2, 3]
    assert child1_bias != [1, 2, 3]
    assert child2_bias != [1, 2, 3]