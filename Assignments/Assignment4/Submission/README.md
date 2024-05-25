
# Assignment 4: ELMO

# Roll Number : 2022201060

## Downloading Pretrained Models and Data

Follow the following instructions to download the required files before running the ELMO and classifier.

1. Click on the following Google Drive link to download the `saved_models` directory:
   [[Google Drive link](https://drive.google.com/drive/folders/1DcZkODXuhTwS8YffWOmlWn2RK-bsfLkV?usp=drive_link)]

2. Once the download is complete, move the `saved_models` directory to the root folder of this repository.

3. Additionally, download the `Data` directory containing datasets and GloVe embeddings from the same link:
   [Insert Google Drive link here]

4. Place the `Data` directory in the root folder of this repository alongside the `saved_models` directory.

## Directory Structure of Pretrained Models

The `saved_models` directory has the following structure:

```
- saved_models/
  - Frozen/
    - classifier.pt
  - learning_function/
    - classifier.pt
  - Trainable/
    - classifier.pt
- bilstm.pt
```

Explanation of Each Directory and File:

1. `Frozen/`: This directory contains a pretrained classifier model named `classifier.pt`. The model in this directory is with frozen Lambda's.

2. `learning_function/`: Here, you'll find another pretrained classifier model named `classifier.pt`.The model in this directory is with  a learning function applied during training.

3. `Trainable/`: This directory includes yet another pretrained classifier model named `classifier.pt`. The model in this directory is with Trainable Lambda's.

4. `bilstm.pt`: This file is not within any subdirectory. It comprises a pretrained ELMo model in PyTorch's `.pth` format. The model asked in the assignment `bilstm.pt` is written with this name here.

## Directory Structure of Data

The `Data` directory has the following structure:

```
- Data/
  - glove.6B.100d.txt
  - test.csv
  - train.csv
```

Explanation of Each File:

1. `glove.6B.100d.txt`: This file contains GloVe word embeddings.

2. `test.csv`: This CSV file contains the testing dataset.

3. `train.csv`: This CSV file contains the training dataset.

These datasets and embeddings are essential for training and evaluating the classifier model.

## Running ELMO.py and classification.py

To run the `ELMO.py` and `classification.py` scripts with the provided arguments (Can be changed based on the requirements and resources), execute the following commands in your terminal:

For `ELMO.py`:

```bash
python3 ELMO.py --train_data_file data/train.csv --test_data_file data/test.csv --embeddings_file_location data/glove.6B.100d.txt --num_samples 1000 --num_epochs 5 --batch_size 32 --learning_rate 0.001 --embedding_dim 100 --embedding_size 100 --hidden_size 50 --dropout 0.5 --load_elmo --is_cuda
```

For `classification.py`:

```bash
python3 classification.py --cuda --train --train_data_file data/train.csv --test_data_file data/test.csv --embeddings_file_location data/glove.6B.100d.txt --classification_model_file Frozen/classifier.pt --batch_size 64 --input_size 100 --output_size 4 --num_epochs 1 --samples 120000 --learning_rate 0.001 --embedding_size 100 --hidden_size 50 --dropout 0.5
```

Ensure that the necessary data files (`data/train.csv`, `data/test.csv`, and `data/glove.6B.100d.txt`) are available in the specified locations, and the pretrained models are present in the appropriate directories as specified in the README.
