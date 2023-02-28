import torch
from pathlib import Path
from training.model_managment import create_model, load_model
from training.train_model import check_accuracy


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