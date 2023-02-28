import os
import torch
import glob
import torch.nn as nn
from pathlib import Path


class FCmodel(nn.Module):
    def __init__(self, input_size: int, output_size: int):
        super(FCmodel, self).__init__()
        self.fc1 = nn.Linear(input_size, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 32)
        self.fc4 = nn.Linear(32, output_size)
        self.embedding = torch.empty(32)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = torch.relu(self.fc3(x))
        self.embedding = x
        x = torch.sigmoid(self.fc4(x))
        return x.squeeze()

    def get_embedding(self):
        return self.embedding


def create_model(input_size: int, output_size: int) -> nn.Module:
    return FCmodel(input_size, output_size)


def save_model(checkpoint_dir: Path, model: nn.Module, checkpoint_name: str) -> None:
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    checkpoint_path = checkpoint_dir / checkpoint_name
    torch.save(model.state_dict(), checkpoint_path)


def load_model(checkpoint_dir: Path, model: nn.Module) -> nn.Module:
    if not checkpoint_dir.exists():
        return model
    # check if there are any checkpoints in the directory
    models_path = list(checkpoint_dir.glob('*.pth'))
    if not models_path:
        return model

    # get the latest checkpoint by sorting the checkpoints by the number at the end of the file name
    latest_checkpoint = sorted(models_path, key=lambda x: int(x.stem.split('_')[-1]))[-1]

    model.load_state_dict(torch.load(latest_checkpoint))

    return model


if __name__ == '__main__':
    model = create_model(10, 1)
    save_model(Path('./checkpoints'), model, 'test.pth')
    load_model(Path('./checkpoints'), model)

