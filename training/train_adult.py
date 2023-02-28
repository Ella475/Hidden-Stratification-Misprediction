from preprocess.preprocessing_adult import load_and_preprocess_adult
from training.train_model import train


def train():
    data = load_and_preprocess_adult(path="../datasets/adult")
    train()