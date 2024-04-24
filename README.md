# Machine Learning in Chess Move Predictions

## About the Project

This is the final project of Josh Cox for the University of Exeter. We use different machine learning models to predict the next move played in a game of chess. These results will the be interpreted in order to gain insights into the importance of different chess principles at different elo ratings.

## Table of Contents

- [Running the Project](#running-the-project)
- [Testing the Project](#testing-the-project)

## Running the Project

### Preprocessing

---

Before running the preprocessing module ensure your file of chess games is stored in the `/data` folder in pgn format, and that you have navigated to the `src` folder in a terminal.

This command should be run with the positional `TYPE` parameter {`single` | `multiple`} to signify the type of input, with each having their own required parameters:

```ps
python preprocessing.py TYPE
```

#### Options:

| Parameter  |   Type    | Description                                               |
| :--------- | :-------: | :-------------------------------------------------------- |
| --n_inputs |    int    | [Multiple Inputs] Number of inputs to use (-1 for all)    |
| --start    |    int    | [Multiple Inputs] Index of game to start at (default = 0) |
| --r_from   |    int    | [Multiple Inputs] Miniumu elo rating of either player     |
| --r_to     |    int    | [Multiple Inputs] Maximum elo rating of either player     |
| --move     |    int    | [Single Input] Specify move number for single prediction  |
| --turn     | {'w','b'} | [Single Input] Turn (White or Black)                      |
| --file     |  string   | Name of file to process (including any extensions)        |

<br>

**Example to run preprocessing on 1000 games of elo ratings between 500 & 900 from position 200:**

```ps
python preprocessing.py multiple --n_inputs 1000 --start 200 --r_from 500 --r_to 900 --file filename.pgn
```

This will preprocess the data, and save it to a CSV file.

### Feature Selecion (PCA)

---

Before running the PCA module ensure your preprocessed csv file is stored in the `/data` folder and that you have navigated to the `src` folder in a terminal.

The module can be run with the following command:

```ps
python PCA.py
```

#### Options:

| Parameter |       Type       | Description                                |
| :-------- | :--------------: | :----------------------------------------- |
| --plot    | {'elbow', 'bar'} | Type of PCA plot to show                   |
| --file    |      string      | CSV file to use (excluding any extensions) |

<br>

**Example to run PCA on file with an elbow plot:**

```ps
python PCA.py --plot elbow --file filename
```

The program will then show two plots visualising the PCA components for both the first and second model. After closing the plots, you will be asked for a number of components to use for each model. These are of type `int` and can be between 1 and the max number of components. The formatted data will the be saved to a file.

### Optimising PCA

---

Before running the optimise PCA module ensure your preprocessed csv file is stored in the `/data` folder and that you have navigated to the `src` folder in a terminal.

The module can be run with the following command:

```ps
python optimise_pca.py
```

#### Options:

| Parameter |        Type         | Description                                |
| :-------- | :-----------------: | :----------------------------------------- |
| --model   | {'dt', 'gb', 'ebm'} | Model to train                             |
| --comps_1 |         int         | Component value to test model 1 with       |
| --comps_2 |         int         | Component values to test model 2 with      |
| --file    |       string        | CSV file to use (excluding any extensions) |

<br>

**Example to run optimise_pca on a DT model with components [1, 4, 6] and [5, 7, 9]:**

```ps
python optimise_pca.py --model dt --comps_1 1 4 6 --comps_2 5 7 9 --file filename
```

The program will then train and the overall model with every combination of the different PCA value. In the example above, the model will be trained and tested 9 times. The best combination of components will be printed to the console as well as the best accuracy achieved.

### Training & Hyperparameter Tuning

---

Before running the training module ensure your PCA transformed folder is stored in the `/PCA` folder (the name of the folder will be the two component values chosen by the user, separated by an underscore). Then navigate to the `src/` folder in a terminal.

The module can be run with the following command:

```ps
python training.py
```

#### Options:

| Parameter |        Type         | Description                                |
| :-------- | :-----------------: | :----------------------------------------- |
| --model   | {'dt', 'gb', 'ebm'} | Model to train                             |
| --comps_1 |         int         | Number of components to train model 1 with |
| --comps_2 |         int         | Number of components to train model 2 with |
| --hyper   |        bool         | Flag to tune hyperparameters               |

<br>

**Example to run training on a GB model with the PCA components 4 and 5:**

```ps
python training.py --model gb --comps_1 4 -comps_2 5
```

The program will train and test the model, printing the accuracy, precision, recall and f1-score. The model and test scores will be saved to files.

<br>

**Example to run hyperparameter tuning on an EBM model with the PCA components 6 and 8:**

```ps
python training.py --model ebm --comps_1 6 -comps_2 8 --hyper
```

The program wil run the scikit-learn BayesSearch to tune the hyperparameters of the specified model. Each iteration will be printed to the console, and the best parameters as well as the cross validation score achieved will be printed and saved to a file.

### Predicting

---

Before running the prediction module ensure you have trained a model using the training module and that your input file for predictions has been formatted using the preprocessing module. Then navigate to the `src` folder in a terminal.

This command should be run with the positional `TYPE` parameter {`single` | `multiple`} to signify the type of input:

```ps
python prediction.py TYPE
```

#### Options:

| Parameter |        Type         | Description                                |
| :-------- | :-----------------: | :----------------------------------------- |
| --model   | {'dt', 'gb', 'ebm'} | Model that was trained                     |
| --comps_1 |         int         | Component value model 1 was trained with   |
| --comps_2 |         int         | Component value model 2 was trained with   |
| --input   |       string        | CSV file to use (excluding any extensions) |

<br>

**Example to run prediction on a DT model with component numbers 3 and 8:**

```ps
python prediction.py --model dt --comps_1 3 --comps_2 8 --input filename
```

If multiple inputs have been chosen, the model will make predictions and the accuracy, precision, recall and f1-score will be printed, as well as being saved to a file.
If a single prediction was chosen, the model will make predictions and the user may navigate, using the command line, the board positions, along with the actual next move and the predicted next move.

### Interpreting

---

Before running the interpet_models module ensure you have trained a model using the training module and that you have navigated to the `src` folder in a terminal.

```ps
python interpret_models.py
```

#### Options:

| Parameter   |                      Type                       | Description                                   |
| :---------- | :---------------------------------------------: | :-------------------------------------------- |
| --model     |               {'dt', 'gb', 'ebm'}               | Model that was trained                        |
| --comps_1   |                       int                       | Component value model 1 was trained with      |
| --comps_2   |                       int                       | Component value model 2 was trained with      |
| --input     |                     string                      | CSV file to use (excluding any extensions)    |
| --plot      | {'all', 'summary', 'waterfall', 'force', 'bar'} | The type of plot to show                      |
| --plot_type |               {'original', 'pca'}               | [Waterfall Plot] The features to use for plot |
| --suffix    |                     string                      | [Optional] Suffix for plot filenames          |

<br>

**Example to run interpret_models on a GB model with component values 3 and 8, for summary plot, with the suffix "gb_1":**

```ps
python interpret_models.py --model gb --comps_1 3 --comps_2 8 --input filename --plot summary --suffix gb_1
```

<br>

**Example to run interpret_models on a DT model with component values 3 and 8, for waterfall plot with original features:**

```ps
python interpret_models.py --model dt --comps_1 3 --comps_2 8 --input filename --plot waterfall --plot_type original
```

The program will run the SHAP interpretation of the model predictions on the given input. A subplot containing plots for both models will be shown, while the plots will be saved to files separately (with a suffix if given).

## Testing the Project
