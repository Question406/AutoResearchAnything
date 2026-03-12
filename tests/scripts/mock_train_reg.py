"""Mock training script that returns deterministic metrics based on hyperparameters."""

import argparse
import json
import math


def train(lr: float, epochs: int, batch_size: int) -> dict:
    lr_factor = math.exp(-((math.log10(lr) + 3.5) ** 2) / 0.5)
    epoch_factor = 1 - math.exp(-epochs / 20)
    batch_factor = 1 - abs(batch_size - 32) / 128

    accuracy = 0.5 + 0.45 * lr_factor * epoch_factor * max(0, batch_factor)
    loss = 1.0 - accuracy + 0.05

    return {
        "accuracy": round(accuracy, 4),
        "loss": round(loss, 4),
        "lr": lr,
        "epochs": epochs,
        "batch_size": batch_size,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=32)
    args = parser.parse_args()

    result = train(args.lr, args.epochs, args.batch_size)
    print(json.dumps(result))
