import torch
import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Outputs the distance achieved, num. iterations, output and hidden activations, architecture, and algorithm used to train the model.")
    parser.add_argument("save_path", help="Path to the saved model")
    args = parser.parse_args()
    info = torch.load(args.save_path)

    #print(f"Algorithm: {info['algorithm']}")
    print(f"Distance achieved: {info['distance']}")
    print(f"Number of iterations: {info['iterations']}")
    print(f"Output activation: {info['output_activation']}")
    print(f"Hidden activation: {info['hidden_activation']}")
    print(f"Architecture: {info['architecture']}")

