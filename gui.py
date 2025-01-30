import dearpygui.dearpygui as dpg
import numpy as np

NES_SCREEN_SIZE = (224, 240, 3)


def draw_mario_network(drawing_tag, agent):
    dpg.delete_item(drawing_tag, children_only=True)
    layer_sizes = agent.layer_sizes
    max_nodes = max(layer_sizes)
    node_radius = 10
    v_spacing = 50
    x_offset, y_offset = 50, 50
    node_positions = {}

    curr_y = y_offset
    for layer_idx, n_nodes in enumerate(layer_sizes):
        start_x = x_offset + (max_nodes - n_nodes) * (node_radius + 5)
        for node_i in range(n_nodes):
            x, y = start_x + node_i * (2 * node_radius + 10), curr_y
            node_positions[(layer_idx, node_i)] = (x, y)
            activation_value = agent.activations.get(f"A{layer_idx}", [0] * n_nodes)[node_i]
            shade = int(min(max(activation_value, 0), 1) * 255)
            color = (shade, 255 - shade, 100, 200)
            dpg.draw_circle(center=[x, y], radius=node_radius, fill=color, color=(255, 255, 255), thickness=2, parent=drawing_tag)
        curr_y += (2 * node_radius + v_spacing)
    for layer_idx in range(1, len(layer_sizes)):
        w_key = f"W{layer_idx}"
        if w_key not in agent.weights:
            continue
        w_mat = agent.weights[w_key]
        for prev_node_i in range(layer_sizes[layer_idx - 1]):
            for curr_node_i in range(layer_sizes[layer_idx]):
                start_pos = node_positions[(layer_idx - 1, prev_node_i)]
                end_pos = node_positions[(layer_idx, curr_node_i)]
                weight_val = w_mat[curr_node_i, prev_node_i]
                color = (50, 100, 255, 180) if weight_val >= 0 else (255, 100, 100, 180)
                dpg.draw_line(p1=start_pos, p2=end_pos, color=color, thickness=1, parent=drawing_tag)


def update_stats_text(tag, episode, reward, distance):
    dpg.set_value(tag, f"Episode: {episode}  |  Reward: {reward:.2f}  |  Dist: {distance:.2f}")

def update_texture(tag, ram)
    dpg.set_value(tag, convert_to_rgba(ram))

# Main UI Setup

def convert_to_rgba(ram):
    return np.concatenate([ram, np.full((128, 128, 1), 255, dtype=np.uint8)], axis=-1)

class GUI:
    def __init__(self, dqn_agent, ga_agent):
        self.dqn_agent = dqn_agent
        self.ga_agent = ga_agent
        self.start_gui()
     

    def start_gui(self):

        with dpg.texture_registry(show=False):
            dpg.add_static_texture(width=NES_SCREEN_SIZE[1], height=NES_SCREEN_SIZE[0], default_value=np.zeros(shape=NES_SCREEN_SIZE[:2], tag="dqn_tex"))
            dpg.add_static_texture(width=NES_SCREEN_SIZE[1], height=NES_SCREEN_SIZE[0], default_value=np.zeros(shape=NES_SCREEN_SIZE[:2], tag="ga_tex"))

        with dpg.window(label="Main Window", tag="Primary Window", width=1200, height=800):
            with dpg.group(horizontal=True):
                with dpg.child_window(width=600, height=-1):
                    dpg.add_text("DQN Agent View:", bullet=True)
                    dpg.add_text("Episode: 0 | Reward: 0 | Dist: 0", tag="dqn_stats")
                    with dpg.group(horizontal=True):
                        dpg.add_image("dqn_tex", width=256, height=256)
                    
                    # DQN NN Viz
                    with dpg.child_window(width=-1, height=400, no_scrollbar=True):
                        dpg.add_text("Neural Network Visualization:")
                        with dpg.drawlist(width=500, height=600, tag="dqn_drawing"):
                            pass

                with dpg.child_window(width=600, height=-1):
                    dpg.add_text("Random GA Agent View:", bullet=True)
                    dpg.add_text("Episode: 0 | Reward: 0 | Dist: 0", tag="ga_stats")
                    with dpg.group(horizontal=True):
                        dpg.add_image("ga_tex", width=256, height=256)

                    # GA NN Viz
                    with dpg.child_window(width=-1, height=400, no_scrollbar=True):
                        dpg.add_text("Neural Network Visualization:")
                        with dpg.drawlist(width=500, height=600, tag="ga_drawing"):
                            pass
        dpg.create_viewport(title='Mario AI Viz Demo', width=1200, height=800)
        dpg.setup_dearpygui()
        dpg.show_viewport()
        dpg.set_primary_window("Primary Window", True)
        dpg.start_dearpygui()
        dpg.destroy_context()

    def update_gui(self):
        draw_mario_network("dqn_drawing", self.dqn_agent)
        draw_mario_network("ga_drawing", self.ga_agent)
        update_stats_text("dqn_stats" episode=1, reward=123.4, distance=25.6)
        update_stats_text("ga_stats", episode=1, reward=75.0, distance=11.2)


def start_gui():
    dpg.create_viewport(title='Mario AI Viz Demo', width=1200, height=800)
    dpg.setup_dearpygui()
    dpg.show_viewport()
    dpg.set_primary_window("Primary Window", True)
    dpg.start_dearpygui()
    dpg.destroy_context()

