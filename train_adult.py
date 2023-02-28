from configs.expriment_config import DatasetNames
from training.train_model import main


if __name__ == '__main__':
    dataset_name = DatasetNames.ADULT
    main(dataset_name)

