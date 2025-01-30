import dearpygui.dearpygui as dpg
import numpy as np
import torch
from DQN_algorithm.DQNbaseline import dim_reduction

NES_SCREEN_SIZE = (224, 240, 3)


def draw_mario_network(drawing_tag, agent, ram):

    # gets dimensionality reduced ram:
    input_tensor = torch.tensor(dim_reduction(ram, agent.start_row, agent.viz_height, agent.viz_width, agent.encode_row),
                                dtype=torch.float32).unsqueeze(0)

    activations = {}
    x = input_tensor
    for i, layer in enumerate(agent.model.model):
        x = layer(x)
        if isinstance(layer, torch.nn.Linear):
            activations[f"A{i}"] = x.detach().numpy().flatten()

    dpg.delete_item(drawing_tag, children_only=True)

    layer_sizes = agent.model.layer_sizes
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
            
            activation_value = activations.get(f"A{layer_idx}", np.zeros(n_nodes))[node_i]
            shade = int(min(max(activation_value, 0), 1) * 255)
            color = (shade, 255 - shade, 100, 200) 

            dpg.draw_circle(center=[x, y], radius=node_radius, fill=color, color=(255, 255, 255), thickness=2, parent=drawing_tag)

        curr_y += (2 * node_radius + v_spacing)


    layer_idx = 0
    for layer in agent.model.model:
        if isinstance(layer, torch.nn.Linear):
            weight_matrix = layer.weight.data.numpy()
            if layer_idx == 0:
                prev_layer_size = agent.model.layer_sizes[layer_idx]
            else:
                prev_layer_size = agent.model.layer_sizes[layer_idx - 1]

            for prev_node_i in range(prev_layer_size):
                for curr_node_i in range(layer_sizes[layer_idx]):
                    start_pos = node_positions.get((layer_idx - 1, prev_node_i), (0, 0))
                    end_pos = node_positions.get((layer_idx, curr_node_i), (0, 0))

                    weight_val = weight_matrix[curr_node_i, prev_node_i]
                    color = (50, 100, 255, 180) if weight_val >= 0 else (255, 100, 100, 180)

                    dpg.draw_line(p1=start_pos, p2=end_pos, color=color, thickness=1, parent=drawing_tag)

            layer_idx += 1


def update_stats_text(tag, episode, reward, distance):
    dpg.set_value(tag, f"Episode: {episode}  |  Reward: {reward:.2f}  |  Dist: {distance:.2f}")

def update_texture(tag, ram):
    dpg.set_value(tag, convert_to_rgba(ram))


def convert_to_rgba(ram):
    rgba = np.concatenate((ram, np.ones((224, 240, 1), dtype=np.uint8) * 255), axis=-1)
    rgba = rgba.astype(np.float32) / 255.0
    return rgba.flatten()

class GUI:
    def __init__(self):
        self.start_gui()
     

    def start_gui(self):

        with dpg.texture_registry(show=False):
            dpg.add_static_texture(
            width=NES_SCREEN_SIZE[1], height=NES_SCREEN_SIZE[0],
            default_value=np.zeros((NES_SCREEN_SIZE[0], NES_SCREEN_SIZE[1], 4), dtype=np.float32).flatten(),
            tag="dqn_tex"
            )
            dpg.add_static_texture(
            width=NES_SCREEN_SIZE[1], height=NES_SCREEN_SIZE[0],
            default_value=np.zeros((NES_SCREEN_SIZE[0], NES_SCREEN_SIZE[1], 4), dtype=np.float32).flatten(),
            tag="ga_tex"
            )

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

    def update_gui(self, dqn_ram, ga_ram, dqn_stats, ga_stats):

        dqn_tex = convert_to_rgba(dqn_ram)
        ga_tex = convert_to_rgba(ga_ram)

        dpg.set_value("dqn_tex", dqn_tex)
        dpg.set_value("ga_tex", ga_tex)

        #draw_mario_network("dqn_drawing", dqn_agent)
        #draw_mario_network("ga_drawing", ga_agent)

        update_stats_text("dqn_stats", episode=dqn_stats['episode_num'], reward=dqn_stats['max_reward'], distance=dqn_stats['max_distance'])
        update_stats_text("ga_stats", episode=ga_stats['current_gen'], reward=ga_stats['max_fitness'], distance=ga_stats['max_distance'])


def start_gui():
    dpg.create_viewport(title='Mario AI Viz Demo', width=1200, height=800)
    dpg.setup_dearpygui()
    dpg.show_viewport()
    dpg.set_primary_window("Primary Window", True)
    dpg.start_dearpygui()
    dpg.destroy_context()

