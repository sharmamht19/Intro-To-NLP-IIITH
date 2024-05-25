# ASSIGNMENT- 3 NLP

## Experiment Results and Saved Models

### 1. Experiment Results

The experiment results for different window sizes in both SVD and Skip-Gram models are available in the `experiment_results` directory.Also there are 2 ipynb files are available of some of those experiments.

### 2. Saved Models

The trained models for both SVD and Skip-Gram classifiers are stored in the `saved_models` directory. Each model is saved with a descriptive filename indicating the model type and configuration. `Also mentioning that skip-gram-word vectors are trained on 20k samples only`

You can access the train and test data, experiment results and saved models using the following Google Drive link:

[Experiment Results and Saved Models](https://drive.google.com/drive/folders/1M9vXxbF21MK0vskQkl8lIwKegzoJ8vs0?usp=sharing)

----

## How to Run python files

### 1. Running the SVD Word Vectors Model

To train the SVD Word Vectors model, you can use the provided Python script `svd.py`. The script accepts the following command-line arguments:

- `-f, --train_file`: Path to the training file in CSV format (default: `"data/train.csv"`).
- `-k, --dimension`: Dimensionality of word vectors (default: `100`).
- `-s, --save_path`: Path to save the trained model (default: `"svd_word_vectors.pt"`).
- `-c, --context_size`: Size of the context window (default: `2`).
- `-n, --num_samples`: Number of samples to use from the training data (default: `None`).

### How to Run the Script

To run the script with default settings, execute the following command:

```bash
python3 svd.py
```

You can also customize the behavior by specifying the desired arguments. For example:

```bash
python3 svd.py -f data/train.csv -k 100 -s svd_word_vectors.pt -c 2 -n 10000
```

This command will train the model using the specified training file, dimensionality of 100, save the model to `svd_word_vectors.pt`, use a context window size of 2, and train on 10,000 samples from the training data.

-----

### 2. Running the Skip-Gram Word Vectors Model

To train the Skip-Gram Word Vectors model, you can use the provided Python script `skip-gram.py`. The script accepts the following command-line arguments:

- `-d, --embedding_dim`: Dimensionality of word embeddings (default: `100`).
- `-n, --num_neg_samples`: Number of negative samples (default: `4`).
- `-e, --num_epochs`: Number of training epochs (default: `1`).
- `-f, --file_path`: Path to the training file in CSV format (default: `"data/train.csv"`).
- `-s, --num_samples`: Number of samples to use from the training data (default: `15000`).
- `-w, --window_size`: Size of the context window (default: `2`).
- `-o, --save_file`: Path to save the trained model (default: `"skip-gram-word-vectors.pt"`).

### How to Run the Script

To run the script with default settings, execute the following command:

```bash
python3 skip-gram.py
```

You can also customize the behavior by specifying the desired arguments. For example:

```bash
python3 skip-gram.py -d 100 -n 4 -e 1 -f data/train.csv -s 15000 -w 2 -o skip-gram-word-vectors.pt
```

This command will train the Skip-Gram Word Vectors model using the specified configuration.

-----

### 3. Running the SVD-based Classifier

To perform various actions with the SVD-based Classifier, you can use the provided Python script `svd-classification.py`. The script accepts different actions as input and provides the following functionalities:

- `1`: Train the classifier.
- `2`: Test the classifier.
- `3`: Evaluate the classifier.
- `4`: Predict the class of a sentence.

Additionally, you can specify various parameters to customize the behavior of the classifier:

- `--word_vector_file`: Path to the saved word vectors model (default: `'Saved models/svd_word_vectors.pt'`).
- `--train_file`: Path to the training data file (default: `"data/train.csv"`).
- `--num_samples`: Number of samples to use (default: `15000`).
- `--test_file`: Path to the testing data file (default: `"data/test.csv"`).
- `--input_size`: Input size for the classifier (default: `100`).
- `--hidden_size`: Hidden size for the classifier (default: `128`).
- `--output_size`: Output size for the classifier (default: `4`).
- `--batch_size`: Batch size for training (default: `64`).
- `--num_epochs`: Number of epochs for training (default: `2`).

### How to Run the Script

To run the script with default settings, execute the following command:

```bash
python3 svd-classification.py 1
```

This command will train the classifier using the default parameters. You can specify different actions and parameters based on your requirements.

You can also customize the behavior by specifying the desired arguments. For example:

```bash
python3 svd-classification.py 4 --word_vector_file 'Saved models/svd_word_vectors.pt' --train_file 'data/train.csv' --num_samples 15000 --test_file 'data/test.csv' --input_size 100 --hidden_size 128 --output_size 4 --batch_size 64 --num_epochs 2

```

-----

### 4. Running the Skip-Gram-based Classifier

To perform various actions with the Skip-Gram-based Classifier, you can use the provided Python script `skip-gram-classification.py`. The script accepts different actions as input and provides the following functionalities:

- `1`: Train the classifier.
- `2`: Test the classifier.
- `3`: Evaluate the classifier.
- `4`: Predict the class of a sentence.

Additionally, you can specify various parameters to customize the behavior of the classifier:

- `--word_vector_file`: Path to the saved word vectors model (default: `'Saved models/svd_word_vectors.pt'`).
- `--train_file`: Path to the training data file (default: `"data/train.csv"`).
- `--num_samples`: Number of samples to use (default: `15000`).
- `--test_file`: Path to the testing data file (default: `"data/test.csv"`).
- `--input_size`: Input size for the classifier (default: `100`).
- `--hidden_size`: Hidden size for the classifier (default: `128`).
- `--output_size`: Output size for the classifier (default: `4`).
- `--batch_size`: Batch size for training (default: `64`).
- `--num_epochs`: Number of epochs for training (default: `2`).

### How to Run the Script

To run the script with default settings and train the classifier, execute the following command:

```bash
python3 skip-gram-classification.py 1
```

This command will train the classifier using the default parameters. You can specify different actions and parameters based on your requirements.

You can also customize the behavior by specifying the desired arguments. For example:

```bash
python3 skip-gram-classification.py 4 --word_vector_file 'Saved models/svd_word_vectors.pt' --train_file 'data/train.csv' --num_samples 15000 --test_file 'data/test.csv' --input_size 100 --hidden_size 128 --output_size 4 --batch_size 64 --num_epochs 2
```
