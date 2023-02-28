from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.model_selection import train_test_split

from preprocess.preprocess_choice import choose_preprocess_func
from preprocess.preprocessing_adult import Mode, load_and_preprocess_adult
from training.model_managment import create_model, load_model, save_model


def shuffle(train_x, train_y):
    p = np.random.permutation(len(train_y))
    return train_x[p], train_y[p]


def check_accuracy(preds, labels):
    preds = preds > 0.5
    labels = labels > 0.5
    return (preds == labels).float().mean()


def test(checkpoint_dir, test_data, loss_func=torch.nn.BCELoss()):
    test_x = test_data.iloc[:, :-1].values
    test_y = test_data.iloc[:, -1].values
    model = create_model(test_x.shape[1], 1)
    checkpoints_dir = Path(checkpoint_dir)
    model = load_model(checkpoints_dir, model)

    test_x = torch.from_numpy(test_x).type(torch.FloatTensor)
    test_y = torch.from_numpy(test_y).type(torch.FloatTensor)

    test_preds = model(test_x)
    test_loss = loss_func(test_preds, test_y)
    test_acc = check_accuracy(test_preds, test_y)
    return test_loss, test_acc


def train(checkpoint_dir, dataset_name,
          df_preprocessed: pd.DataFrame = None, epochs=150, batch_size=256, print_every=10,
          loss_func=torch.nn.BCELoss(), lr_scheduler_patience=100, lr_scheduler_patience_factor=0.95):
    if df_preprocessed is None:
        raise Exception("No data was given to train on")
    # else:
    #     preprocessing_function = choose_preprocess_func(dataset_name)
    #     # Load and preprocess the dataset
    #     df_preprocessed = preprocessing_function(path=data_path, mode=Mode.TRAIN)

    # Split the dataset into training and testing sets
    train_data, val_data = train_test_split(df_preprocessed, test_size=0.2, random_state=42)

    train_x = train_data.iloc[:, :-1].values
    train_y = train_data.iloc[:, -1].values
    val_x = val_data.iloc[:, :-1].values
    val_y = val_data.iloc[:, -1].values

    history = {'train_loss': [], 'val_loss': [], 'train_acc': [], 'val_acc': [], 'iteration': []}

    num_train = train_x.shape[0]
    iterations_per_epoch = max(num_train // batch_size, 1)

    # Define the model and optimizer
    model = create_model(train_x.shape[1], 1)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer=optimizer, mode='min',
                                                              patience=lr_scheduler_patience,
                                                              factor=lr_scheduler_patience_factor,
                                                              verbose=False)

    checkpoints_dir = Path(checkpoint_dir)
    model = load_model(checkpoints_dir, model)

    best_accuracy = 0

    train_x = torch.from_numpy(train_x).type(torch.FloatTensor)
    train_y = torch.from_numpy(train_y).type(torch.FloatTensor)
    val_x = torch.from_numpy(val_x).type(torch.FloatTensor)
    val_y = torch.from_numpy(val_y).type(torch.FloatTensor)

    for i, epoch in enumerate(range(epochs)):
        train_x, train_y = shuffle(train_x, train_y)

        for j in range(iterations_per_epoch):
            # create integer Tensor from numpy array
            batch = train_x[j:(j + batch_size)]
            batch_labels = train_y[j:(j + batch_size)]
            output = model(batch)
            loss = loss_func(output, batch_labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if print_every and not j % print_every:
                model.eval()

                with torch.no_grad():
                    # create tensor from numpy array
                    output = model(train_x)
                    output = output
                    history['train_acc'].append(check_accuracy(output, train_y))
                    train_loss = loss_func(output, train_y)
                    train_loss_value = train_loss.item()
                    history['train_loss'].append(train_loss_value)

                    output = model(val_x)
                    history['val_acc'].append(check_accuracy(output, val_y))
                    val_loss = loss_func(output, val_y)
                    val_loss_value = val_loss.item()
                    history['val_loss'].append(val_loss_value)

                    if best_accuracy < history['val_acc'][-1]:
                        save_model(checkpoints_dir, model, f'checkpoint_{dataset_name}_{epoch}.pth')
                        best_accuracy = history['val_acc'][-1]

                print(f"Epoch: {i}, Iteration: {j}, Loss: {history['train_loss'][-1]:.3f}, "
                      f"Accuracy: {history['train_acc'][-1]:.3f}, Validation Accuracy: {history['val_acc'][-1]:.3f}")

                model.train()

            with torch.no_grad():
                lr_scheduler.step(loss)

    if print_every:
        num_prints = len(history['train_loss'])
        history['iteration'] = np.linspace(0, num_prints * print_every, num_prints, endpoint=False)
    return history


if __name__ == '__main__':
    df = load_and_preprocess_adult(path="../datasets/adult.data")
    train(checkpoint_dir="./checkpoints/adult", dataset_name="adult", df_preprocessed=df,
          epochs=150, batch_size=1000, print_every=10,
          loss_func=torch.nn.BCELoss(), lr_scheduler_patience=50, lr_scheduler_patience_factor=0.95)

