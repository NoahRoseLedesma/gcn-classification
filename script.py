from data import Dataset_Loader
from model import GCN
import wandb
import sys

hyperparameter_defaults = {
    "dataset": "cora",
    'num_epochs': 200,
    "learning_rate": 0.01,
    "weight_decay": 5e-4,
    "dropout": 0.2,
    "hidden_dim": 16
}

if __name__ == "__main__":
    wandb.init(config=hyperparameter_defaults, project="gcn-classification")

    print("Loading data...")
    # Load the dataset
    data = Dataset_Loader(wandb.config["dataset"]).load()

    # Create the model
    model = GCN(wandb.config, data)

    # Train the model
    print("Training...")
    model.train()
    print("Evaluating...")
    model.evaluate()