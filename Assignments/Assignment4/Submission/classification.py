import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import numpy as np
from tqdm import tqdm
import re
import matplotlib.pyplot as plt
from torch.nn.utils.rnn import pad_sequence
import torch.optim as optim
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, ConfusionMatrixDisplay
import argparse


class CustomDataset(Dataset):
    def __init__(self, file_path=None, data=None, labels=None, vocab=None):
        if file_path:
            self.file_path = file_path
            self.load_and_preprocess_data()
        else:
            self.data = data
            self.labels = labels
            self.vocab = vocab

    def __len__(self):
        return len(self.data) if hasattr(self, 'data') else 0

    def __getitem__(self, index):
        sentence = self.data[index]
        label = self.labels[index]
        return sentence, label

    def clean_sentence(self, sent):
        sent = sent.lower()
        sent = re.sub('\n', ' ', sent)
        sent = re.sub("\s+", ' ', sent)
        sent = re.sub("http[s]?://\S+", " <URL> ", sent)
        sent = re.sub(r"[^A-Za-z0-9<>\s]", '', sent)
        return sent

    def clean_data(self, data):
        cleaned_data_with_markers = []
        for sentence in data:
            cleaned_sentence = self.clean_sentence(sentence)
            sentence_with_markers = "<sos> " + cleaned_sentence + " <eos>"
            cleaned_data_with_markers.append(sentence_with_markers)
        return cleaned_data_with_markers

    def update_corpus(self, vocab):
        self.data = [[vocab[word] if word in vocab else vocab["<unk>"]
                      for word in sent.split(" ")] for sent in self.data]

    def load_and_preprocess_data(self, num_samples=None, reduce_label=True):
        if not hasattr(self, 'file_path'):
            raise ValueError("File path not provided.")

        if num_samples:
            data = pd.read_csv(self.file_path, nrows=num_samples)
        else:
            data = pd.read_csv(self.file_path)

        if reduce_label and 'Class Index' in data.columns:
            data['Class Index'] -= 1

        self.data = self.clean_data(data['Description'])
        self.labels = data['Class Index']

    def get_data_subset(self, subset_size):
        if subset_size > len(self.data):
            subset_size = len(self.data)
        return self.data[:subset_size], self.labels[:subset_size]


class VocabBuilder:
    def __init__(self):
        self.vocab = {
            "<pad>": 0,
            "<sos>": 1,
            "<eos>": 2,
            "<unk>": 3
        }
        self.embeddings = {
            "<pad>": np.zeros(100),
            "<sos>": np.random.rand(100),
            "<eos>": np.random.rand(100),
            "<unk>": np.random.rand(100)
        }

    def build_vocab(self, cleaned_sentences):
        for sentence in cleaned_sentences:
            words_arr = sentence.split(" ")
            for word in words_arr:
                if word not in self.vocab:
                    self.vocab[word] = len(self.vocab)

    def load_glove_embeddings(self, embeddings_file_location):
        with open(embeddings_file_location, 'r', encoding="utf-8") as file:
            for line in file:
                word, *embeddings = line.split()
                embeddings = np.asarray(embeddings, dtype="float32")
                self.embeddings[word] = embeddings

    def create_embedding_matrix(self):
        embedding_matrix = []
        vocab_words = list(self.vocab.keys())
        for word in vocab_words:
            embedding_matrix.append(
                self.embeddings[word] if word in self.embeddings else self.embeddings["<unk>"])
        embedding_matrix = np.array(embedding_matrix)
        return torch.Tensor(embedding_matrix)


class ELMo(nn.Module):
    def __init__(self, vocab_size, embedding_size, hidden_size, dropout, embeddings=None):
        super(ELMo, self).__init__()
        self.embedding_size = embedding_size
        if embeddings is not None:
            self.embedding = nn.Embedding.from_pretrained(
                embeddings, freeze=True)
        else:
            self.embedding = nn.Embedding(vocab_size, embedding_size)

        self.forward_lstm1 = nn.LSTM(
            input_size=embedding_size, hidden_size=hidden_size, num_layers=1, batch_first=True)

        self.forward_lstm2 = nn.LSTM(
            input_size=hidden_size, hidden_size=hidden_size, num_layers=1, batch_first=True)

        self.backward_lstm1 = nn.LSTM(
            input_size=embedding_size, hidden_size=hidden_size, num_layers=1, batch_first=True)

        self.backward_lstm2 = nn.LSTM(
            input_size=hidden_size, hidden_size=hidden_size, num_layers=1, batch_first=True)

        self.linear = nn.Linear(hidden_size * 2, vocab_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, X):
        embeddings = self.embedding(X)

#         print("Embeddings size:", embeddings.size())

        forward_output1, _ = self.forward_lstm1(embeddings)
        forward_output1 = self.dropout(forward_output1)
        backward_output1, _ = self.backward_lstm1(embeddings.flip(dims=[1]))
        backward_output1 = self.dropout(backward_output1)
#         print("Forward LSTM 1 output size:", forward_output1.size())
#         print("Backward LSTM 1 output size:", backward_output1.size())

        forward_output2, _ = self.forward_lstm2(forward_output1)
        backward_output2, _ = self.backward_lstm2(
            backward_output1.flip(dims=[1]))

#         print("Forward LSTM 2 output size:", forward_output2.size())
#         print("Backward LSTM 2 output size:", backward_output2.size())

        layer_output1 = torch.cat([forward_output1, backward_output1], dim=-1)
        layer_output2 = torch.cat([forward_output2, backward_output2], dim=-1)

#         print("Layer output 1 size:", layer_output1.size())
#         print("Layer output 2 size:", layer_output2.size())

        output = self.linear(layer_output2)
        f_output = torch.transpose(output, 1, 2)
        return f_output


def custom_collate_fn(batch):
    sorted_batch = sorted(batch, key=lambda x: len(x[0]), reverse=True)
    sentences, labels = zip(*sorted_batch)

    # Pad sequences
    padded_sentences = pad_sequence(
        [torch.tensor(sent) for sent in sentences], batch_first=True, padding_value=0)

    # Convert labels to tensors
    label_tensors = [torch.tensor(label) for label in labels]

    return padded_sentences, torch.stack(label_tensors)


class Model(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, elmo, device, embd_size=100, use_trainable_lambdas=False, num_classes=4):
        super(Model, self).__init__()
        self.elmo = elmo
        self.lstm_layer = nn.LSTM(
            input_size, hidden_size, batch_first=True, bidirectional=True)
        self.fc_layer = nn.Linear(hidden_size * 2, output_size)
        self.hidden_size = hidden_size
        self.linear = nn.Linear(input_size, output_size)
        self.sigmoid = nn.Sigmoid()
        self.device = device
        if use_trainable_lambdas:
            self.lambda1 = nn.Parameter(torch.ones(1))
            self.lambda2 = nn.Parameter(torch.ones(1))
            self.lambda3 = nn.Parameter(torch.ones(1))
        else:
            self.lambda1 = torch.tensor([0.2])
            self.lambda2 = torch.tensor([0.4])
            self.lambda3 = torch.tensor([0.4])

        self.gamma = nn.Parameter(torch.ones(1))

    def forward(self, input_seq):
        embeddings = self.elmo.module.embedding(input_seq).to(self.device)
        forward_output1, _ = self.elmo.module.forward_lstm1(embeddings)
        backward_output1, _ = self.elmo.module.backward_lstm1(
            embeddings.flip(dims=[1]))

        forward_output2, _ = self.elmo.module.forward_lstm2(forward_output1)
        backward_output2, _ = self.elmo.module.backward_lstm2(
            backward_output1.flip(dims=[1]))

        layer_output1 = torch.cat(
            [forward_output1, backward_output1], dim=-1).to(self.device)
        layer_output2 = torch.cat(
            [forward_output2, backward_output2], dim=-1).to(self.device)

        self.lambda1 = self.lambda1.to(self.device)
        self.lambda2 = self.lambda2.to(self.device)
        self.lambda3 = self.lambda3.to(self.device)

        s_sum = self.lambda1 + self.lambda2 + self.lambda3

        final_embeddings = self.gamma * (
            self.lambda1 / s_sum * embeddings
            + self.lambda2 / s_sum * layer_output1
            + self.lambda3 / s_sum * layer_output2
        ).to(torch.float32).to(self.device)

        lstm_output, _ = self.lstm_layer(final_embeddings)
        last_hidden_state = torch.cat(
            (lstm_output[:, -1, :self.hidden_size], lstm_output[:, 0, self.hidden_size:]), dim=1)
        logits = self.fc_layer(last_hidden_state)

        return logits

    def predict(self, logits):
        predicted_class_index = torch.argmax(logits, dim=1)
        return predicted_class_index


class Model_lrnfn(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, elmo, device, embd_size=100, num_classes=4):
        super(Model_lrnfn, self).__init__()
        self.elmo = elmo
        self.lstm_layer = nn.LSTM(
            input_size, hidden_size, batch_first=True, bidirectional=True)
        self.fc_layer = nn.Linear(hidden_size * 2, output_size)
        self.hidden_size = hidden_size
        self.linear = nn.Linear(input_size, output_size)
        self.sigmoid = nn.Sigmoid()
        self.linear_concat_forward = nn.Linear(
            in_features=2 * self.hidden_size, out_features=input_size)
        self.linear_concat_backward = nn.Linear(
            in_features=2 * self.hidden_size, out_features=input_size)
        self.linear_input_embedded = nn.Linear(
            in_features=input_size, out_features=input_size)
        self.device = device

    def forward(self, input_seq):
        embeddings = self.elmo.module.embedding(input_seq).to(self.device)
        forward_output1, _ = self.elmo.module.forward_lstm1(embeddings)
        backward_output1, _ = self.elmo.module.backward_lstm1(
            embeddings.flip(dims=[1]))

        forward_output2, _ = self.elmo.module.forward_lstm2(forward_output1)
        backward_output2, _ = self.elmo.module.backward_lstm2(
            backward_output1.flip(dims=[1]))

        layer_output1 = torch.cat(
            [forward_output1, backward_output1], dim=-1).to(self.device)
        layer_output2 = torch.cat(
            [forward_output2, backward_output2], dim=-1).to(self.device)

        output_concat_forward = self.linear_concat_forward(layer_output1)
        output_concat_backward = self.linear_concat_backward(layer_output2)
        output_input_embedded = self.linear_input_embedded(embeddings)
        final_embeddings = output_concat_forward + \
            output_concat_backward + output_input_embedded

        lstm_output, _ = self.lstm_layer(final_embeddings)
        last_hidden_state = torch.cat(
            (lstm_output[:, -1, :self.hidden_size], lstm_output[:, 0, self.hidden_size:]), dim=1)
        logits = self.fc_layer(last_hidden_state)

        return logits

    def predict(self, logits):
        predicted_class_index = torch.argmax(logits, dim=1)
        return predicted_class_index


class Classifier:
    def __init__(self, input_size, hidden_size, output_size, elmo, batch_size=64, num_epochs=10, lr=0.001):
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu")
        self.num_epochs = num_epochs
        self.batch_size = batch_size
        self.elmo = elmo
        self.model = Model(input_size, hidden_size, output_size,
                           elmo, device=self.device).to(self.device)
#         self.model = Model_lrnfn(input_size, hidden_size, output_size,elmo,device = self.device ).to(self.device)
        self.model = nn.DataParallel(self.model)
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)
        # print("Input size:", input_size)
        # print("Hidden size:", hidden_size)
        # print("Output size:", output_size)
        # print("ELMo:", elmo)
        # print("Batch size:", batch_size)
        # print("Number of epochs:", num_epochs)
        # print("Learning rate:", lr)

    def train(self, train_data, model_save_path, test_data=None):
        epoch_losses = []
        epoch_accuracies = []

        for epoch in range(self.num_epochs):
            self.model.train()
            running_loss = 0.0
            epoch_predictions = []
            epoch_labels_list = []

            train_data_with_progress = tqdm(train_data, desc=f'Epoch {epoch+1}/{self.num_epochs}')

            for inputs, label in train_data_with_progress:
                inputs, label = inputs.to(self.device), label.to(self.device)

                self.optimizer.zero_grad()
                outputs = self.model(inputs)
#                 print(outputs)
                loss = self.criterion(outputs, label)
                loss.backward()
                self.optimizer.step()

                running_loss += loss.item() * inputs.size(0)
                predicted_label = self.model.module.predict(outputs)

                epoch_predictions.extend(predicted_label.tolist())
                epoch_labels_list.extend(label.cpu().tolist())
#                 print(epoch_predictions)
#                 print(epoch_labels_list)

            # Calculate loss and accuracy after each epoch
            avg_loss = running_loss / len(train_data.dataset)
            accuracy = accuracy_score(epoch_labels_list, epoch_predictions)

            epoch_losses.append(avg_loss)
            epoch_accuracies.append(accuracy)
            val_acc = self.evaluate(test_data, pr=False)
            print(f'Epoch {epoch+1}/{self.num_epochs}, Loss: {avg_loss:.4f}, Accuracy: {accuracy:.4f}, Val Acc : {val_acc:.4f}')

        self.save_model(model_save_path)
        return epoch_losses, epoch_accuracies

    def evaluate(self, test_data, pr=True):
        self.model.eval()
        predictions = []
        labels_list = []

        with torch.no_grad():
            for inputs, label in test_data:
                inputs, label = inputs.to(self.device), label.to(self.device)
                outputs = self.model(inputs)

                predicted_label = self.model.module.predict(outputs)
                predictions.extend(predicted_label.tolist())
                labels_list.extend(label.cpu().tolist())

        accuracy = accuracy_score(labels_list, predictions)
        if pr:
            print(f'Test Accuracy: {accuracy:.4f}')
        return accuracy

    def calculate_metrics(self, data_loader):
        self.model.eval()
        predictions = []
        labels_list = []

        with torch.no_grad():
            for inputs, labels in data_loader:
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                outputs = self.model(inputs)
                predicted_label = self.model.module.predict(outputs)
                predictions.extend(predicted_label.tolist())
                labels_list.extend(labels.cpu().tolist())

        accuracy = accuracy_score(labels_list, predictions)
        precision = precision_score(labels_list, predictions, average='macro')
        recall = recall_score(labels_list, predictions, average='macro')
        f1 = f1_score(labels_list, predictions, average='macro')
        confusion_mat = confusion_matrix(labels_list, predictions)

        return accuracy, precision, recall, f1, confusion_mat

    def evaluate_performance(self, train_loader, test_loader):
        print("Evaluation on Train Set:")
        train_accuracy, train_precision, train_recall, train_f1, train_confusion_mat = self.calculate_metrics(
            train_loader)
        print(f"Train Accuracy: {train_accuracy:.4f}")
        print(f"Train Precision: {train_precision:.4f}")
        print(f"Train Recall: {train_recall:.4f}")
        print(f"Train F1 Score: {train_f1:.4f}")
        print("Train Confusion Matrix:")
        disp = ConfusionMatrixDisplay(confusion_matrix=train_confusion_mat)
        disp.plot()
        plt.show()
#         print(train_confusion_mat)

        print("\nEvaluation on Test Set:")
        test_accuracy, test_precision, test_recall, test_f1, test_confusion_mat = self.calculate_metrics(
            test_loader)
        print(f"Test Accuracy: {test_accuracy:.4f}")
        print(f"Test Precision: {test_precision:.4f}")
        print(f"Test Recall: {test_recall:.4f}")
        print(f"Test F1 Score: {test_f1:.4f}")
        print("Test Confusion Matrix:")
#         print(test_confusion_mat)
        disp = ConfusionMatrixDisplay(confusion_matrix=test_confusion_mat)
        disp.plot()
        plt.show()

    def save_model(self, filepath):
        torch.save(self.model.state_dict(), filepath)
        print(f'Model saved to {filepath}')

    def predict_class(self, sentence):
        input_tensor = sentence.to(self.device)
        self.model.eval()
        with torch.no_grad():
            output = self.model(input_tensor)
            predicted_label = self.model.predict(output)

        return predicted_label + 1


def load_elmo_model(checkpoint_path, vocab_size=0, embedding_dim=100, hidden_size=128, dropout=0.5, embedding_matrix=None):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    # Instantiate the ELMo model architecture
    elmo_model = ELMo(vocab_size=vocab_size, embedding_size=embedding_dim,
                      hidden_size=hidden_size, dropout=dropout, embeddings=embedding_matrix)
    elmo_model = nn.DataParallel(elmo_model)  # If trained using DataParallel
    # Load the model checkpoint
    elmo_model.load_state_dict(torch.load(
        checkpoint_path, map_location=device))
    # Set the model to evaluation mode
    elmo_model.eval()
    print("ELMo Loaded successfully.")
    return elmo_model


def main(args):
    # Select GPU
    device = torch.device(
        "cuda" if args.cuda and torch.cuda.is_available() else "cpu")
    print("Device:", device)

    train_data_file = args.train_data_file
    test_data_file = args.test_data_file
    embeddings_file_location = args.embeddings_file_location
    classification_model_file = args.classification_model_file
    batch_size = args.batch_size
    input_size = args.input_size
    output_size = args.output_size
    num_epochs = args.num_epochs
    samples = args.samples
    learning_rate = args.learning_rate
    embedding_size = args.embedding_size
    hidden_size = args.hidden_size
    dropout = args.dropout

    # Define the datasets
    train_dataset = CustomDataset(train_data_file)
    test_dataset = CustomDataset(test_data_file)

    # Load and preprocess data for both datasets
    train_dataset.load_and_preprocess_data(
        num_samples=samples)  # Load all training data
    test_dataset.load_and_preprocess_data(
        num_samples=samples)   # Load all testing data

    # Build vocabulary and create embedding matrix
    vocab_builder = VocabBuilder()
    vocab_builder.build_vocab(train_dataset.data)
    vocab_builder.load_glove_embeddings(embeddings_file_location)
    embedding_matrix = vocab_builder.create_embedding_matrix()

    # Update corpus with vocabulary indices
    train_dataset.update_corpus(vocab_builder.vocab)
    test_dataset.update_corpus(vocab_builder.vocab)

    # Print vocab size and embedding matrix shape
    print("Vocabulary size:", len(vocab_builder.vocab))
    print("Embedding matrix shape:", embedding_matrix.shape)
    vocab_size = len(vocab_builder.vocab)

    elmo_model = load_elmo_model('saved_models/elmo_model.pth', vocab_size=vocab_size,
                                 hidden_size=hidden_size, embedding_matrix=embedding_matrix)

    # Instantiate and train classifier
    train_data_loader = DataLoader(
        train_dataset, batch_size=batch_size, collate_fn=custom_collate_fn, shuffle=True)
    test_data_loader = DataLoader(
        test_dataset, batch_size=batch_size, collate_fn=custom_collate_fn, shuffle=True)
    classifier = Classifier(input_size, hidden_size, output_size, num_epochs=num_epochs,
                            lr=learning_rate, batch_size=batch_size, elmo=elmo_model)

    if args.train:
        losses, accuracies = classifier.train(
            train_data_loader, "classifier.pt", test_data_loader)
        print("Losses:", losses)
        print("Accuracies:", accuracies)
    else:
        classifier.model.load_state_dict(torch.load(
            f'saved_models/{classification_model_file}', map_location=device))

    # Evaluate on test dataset
    print("Evaluating the Classifier....")
    test_acc = classifier.evaluate(test_data_loader)

    # Evaluate performance
    # classifier.evaluate_performance(train_data_loader, test_data_loader)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="ELMo Classifier Script")

    parser.add_argument("--cuda", action="store_true", default=True,
                        help="Use CUDA if available")
    parser.add_argument("--train", action="store_true", default=False,
                        help="Train the classifier")
    parser.add_argument("--train_data_file", type=str, default="data/train.csv",
                        help="Path to the training data file")
    parser.add_argument("--test_data_file", type=str, default="data/test.csv",
                        help="Path to the testing data file")
    parser.add_argument("--embeddings_file_location", type=str, default="data/glove.6B.100d.txt",
                        help="Path to the embeddings file")
    parser.add_argument("--classification_model_file", type=str, default="Frozen/classifier.pt",
                        help="Path to the classification model file")
    parser.add_argument("--batch_size", type=int, default=64,
                        help="Batch size for training")
    parser.add_argument("--input_size", type=int, default=100,
                        help="Input size")
    parser.add_argument("--output_size", type=int, default=4,
                        help="Output size")
    parser.add_argument("--num_epochs", type=int, default=1,
                        help="Number of epochs")
    parser.add_argument("--samples", type=int, default=None,
                        help="Number of samples")
    parser.add_argument("--learning_rate", type=float, default=0.001,
                        help="Learning rate")
    parser.add_argument("--embedding_size", type=int, default=100,
                        help="Embedding size")
    parser.add_argument("--hidden_size", type=int, default=50,
                        help="Hidden size")
    parser.add_argument("--dropout", type=float, default=0.5,
                        help="Dropout probability")

    args = parser.parse_args()
    main(args)

# how to run
# python3 classification.py --cuda --train --train_data_file data/train.csv --test_data_file data/test.csv --embeddings_file_location data/glove.6B.100d.txt --classification_model_file Frozen/classifier.pt --batch_size 64 --input_size 100 --output_size 4 --num_epochs 1 --samples 120000 --learning_rate 0.001 --embedding_size 100 --hidden_size 50 --dropout 0.5
