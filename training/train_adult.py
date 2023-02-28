import torch
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from datasets import customDataset
from pathlib import Path

from evaluation import precision, recall
from model_managment import create_model, load_model, save_model
from preprocessing_adult import load_and_preprocess_adult


def train_model(path="../datasets/credit", checkpoint_dir="./checkpoints/credit" ):
    # Load and preprocess the dataset
    df_preprocessed = load_and_preprocess_adult("../datasets/adult")
    # Split the dataset into training and testing sets
    train_data, test_data = train_test_split(df_preprocessed, test_size=0.2, random_state=42)
    # Define the dataloaders and training loop
    train_dataset = customDataset(df_preprocessed)
    train_dataloader = DataLoader(train_dataset, batch_size=32, shuffle=True)

    test_dataset = customDataset(test_data)
    test_dataloader = DataLoader(test_dataset, batch_size=32, shuffle=False)

    # Define the model and optimizer
    model = create_model(train_dataset.X.shape[1], 1)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = torch.nn.BCELoss()

    checkpoints_dir = Path('./checkpoints/adult')
    model = load_model(checkpoints_dir, model)

    for epoch in range(100):
        running_loss = 0.0
        for i, data in enumerate(train_dataloader):
            inputs, labels = data

            optimizer.zero_grad()

            outputs = model(inputs)
            loss = criterion(outputs, labels.view(labels.shape[0], -1))

            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            # Evaluate the model on the test set
            test_loss = 0.0
            test_precision = 0
            test_recall = 0
            with torch.no_grad():
                for test_data in test_dataloader:
                    inputs, test_labels = test_data
                    test_outputs = model(inputs)
                    loss = criterion(test_outputs, test_labels.view(test_labels.shape[0], -1))
                    if i != 0:
                        test_loss += loss.item()
                        # test_recall += recall(test_outputs.round().detach().numpy(), test_labels.detach().numpy())
                        test_precision += precision(test_outputs.round().detach().numpy(), test_labels.detach().numpy())

                # Calculate the average loss and accuracy
                test_loss /= len(test_dataset)
                test_precision /= len(test_dataset)
                test_recall /= len(test_dataset)

                test_loss *= 32
                test_precision *= 32
                test_recall *= 32

            # Print the loss and accuracy every 10 iterations
            if i % 10 == 0:
                print(f'Epoch{epoch + 1},Iteration{i + 1},Training Loss:{running_loss / len(train_dataset):.6f},'
                      f'Test Loss:{test_loss / len(test_dataset): .6f}, ',
                      f'precision:{precision(outputs.round().detach().numpy(), labels):.2f},'
                      f'recall:{recall(outputs.round().detach().numpy(), labels):.2f}',
                      f'precision:{test_precision:.2f},recall:{test_recall:.2f}')

        # Save the model every 10 epochs
        if epoch % 10 == 0 and epoch != 0:
            save_model(checkpoints_dir, model, f'checkpoint_adult_{epoch}.pth')


if __name__ == '__main__':
    train_model()
