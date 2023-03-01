# Tabular Data Science Project – Hidden Stratification Misprediction
Ella Shalom, 208288423

## Introduction
Final project at Bar-Ilan University computer science department - Tabular Data Science course.
<br />The project is about hidden stratification
misprediction and how to detect it. <br />The project is written in python.

## Installation
1. Clone the repository
2. Install the requirements
```bash
pip install -r requirements.txt
```
### Note: 
The project was written in python 3.8.5, so it is recommended to use the same version.<br />
We were asked to supply only the requirements not shown in class.<br />
To install all the requirements, run the following command:
```bash 
pip install -r requirements_full.txt
```

## Usage
There are 4 main scripts:
1. `trainer.py` - trains a model on the dataset to use as feature extractor if nedded.<br />
The relevant checkppoints are saved in the `training/checkpoints` folder, so there is no need to run again 
   unless you want to change the model.
2. `runner.py` - Experiment runner - can run in two settings: <br />
   (Both settings can run a new experiment and plot the results and save the plots, or use saved experiment results for plotting.)<br />

    1. `stop_after_clustering=True` - much faster and only performs the clustering on the data and saves the result.<br />
    2. `stop_after_clustering=False` - runs the full experiment - preform clustering and use the clustered data to preform unbalancing experiment.<br />
3. `plotter.py` - Show saved plots of experiments.<br />