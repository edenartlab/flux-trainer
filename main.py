import json
import argparse


if __name__ == "__main__":
    # Load config file
    with open('train_config.json', 'r') as f:
        config = json.load(f)

    # Setup argparse
    parser = argparse.ArgumentParser(description="Training configuration")
    parser.add_argument("--model_path", type=str, default=config["model_path"], help="Path to the Flux model")
    parser.add_argument("--model_path", type=str, default=config["CLIP_L_PATH"], help="Path to the clip_l model")
    parser.add_argument("--epochs", type=int, default=config["epochs"], help="Number of epochs")
    parser.add_argument("--batch_size", type=int, default=config["batch_size"], help="Batch size")
    parser.add_argument("--learning_rate", type=float, default=config["learning_rate"], help="Learning rate")
    parser.add_argument("--dataset_path", type=str, default=config["dataset_path"], help="Path to the dataset")

    args = parser.parse_args()

    # Use arguments
    print(args.model_path, args.epochs, args.batch_size, args.learning_rate, args.dataset_path)
