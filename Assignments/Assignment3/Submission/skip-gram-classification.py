import pandas as pd
import torch
import argparse
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import numpy as np
from tqdm import tqdm
import re
from torch.nn.utils.rnn import pad_sequence
import torch
import re
from torch.utils.data import DataLoader


class SkipGramTextDataProcessor:
    def __init__(self, word_vectors=None, word_to_idx=None, idx_to_word=None):
        self.word_vectors = word_vectors
        self.word_to_idx = word_to_idx

    def load_dataset(self, file_path, num_samples=None, reduce_label=True):
        if num_samples:
            data = pd.read_csv(file_path, nrows=num_samples)
        else:
            data = pd.read_csv(file_path)

        if reduce_label and 'Class Index' in data.columns:
            data['Class Index'] -= 1  # Reduce 1 from each label

        return data

    def preprocess_data(self, data, text_column='Description', label_column='Class Index'):
        corpus = data[text_column]
        labels = data[label_column] if label_column in data.columns else None
        return corpus, labels

    def load_word_vectors(self, file_path):
        saved_data = torch.load(file_path)
        self.word_vectors = saved_data['word_vectors']
        self.word_to_idx = saved_data['word_to_idx']

    def get_word_embedding(self, word):
        if self.word_to_idx is not None and self.word_vectors is not None:
            if word in self.word_to_idx:
                idx = self.word_to_idx[word]
                return self.word_vectors[idx]
            else:
                unk_index = self.word_to_idx.get('<UNK>')
                return self.word_vectors[unk_index]
        else:
            raise ValueError(
                "Word vectors and vocabulary mappings are not loaded.")

    def get_sentence_embeddings(self, sentence):
        tokens = re.findall(r'\b\w+\b', sentence.lower())
        sentence_embeddings = []
        for word in tokens:
            word_embedding = self.get_word_embedding(word)
            sentence_embeddings.append(word_embedding)
        sentence_embeddings = torch.tensor(np.array(sentence_embeddings))
        return sentence_embeddings

    def preprocess_dataset(self, data, batch_size=1):
        corpus, labels = self.preprocess_data(data)
        dataset = []

        for sentence in corpus:
            sentence_embeddings = self.get_sentence_embeddings(sentence)
            dataset.append(sentence_embeddings)

        labels_tensor = torch.tensor(labels)
        combined_data = list(zip(dataset, labels_tensor))
        train_dataloader = DataLoader(
            combined_data, batch_size=batch_size, shuffle=True, collate_fn=self.pad_collate)
        return train_dataloader

    def pad_collate(self, batch):
        batch = sorted(batch, key=lambda x: x[0].size(0), reverse=True)
        sequences = [item[0] for item in batch]
        labels = [item[1] for item in batch]
        padded_sequences = pad_sequence(sequences, batch_first=True)
        return padded_sequences, torch.stack(labels)


class Model(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(Model, self).__init__()
        self.lstm_layer = nn.LSTM(
            input_size, hidden_size, batch_first=True, bidirectional=True)
        self.fc_layer = nn.Linear(hidden_size * 2, output_size)
        self.hidden_size = hidden_size

    def forward(self, input_seq):
        lstm_output, _ = self.lstm_layer(input_seq)
        last_hidden_state = torch.cat(
            (lstm_output[:, -1, :self.hidden_size], lstm_output[:, 0, self.hidden_size:]), dim=1)
        logits = self.fc_layer(last_hidden_state)
        return logits

    def predict(self, logits):
        predicted_class_index = torch.argmax(logits, dim=1)
        return predicted_class_index


class Classifier:
    def __init__(self, input_size, hidden_size, output_size, batch_size=64, num_epochs=10, lr=0.001):
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu")
        self.num_epochs = num_epochs
        self.batch_size = batch_size

        self.model = Model(input_size, hidden_size,
                           output_size).to(self.device)

        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)

    def train(self, train_data, model_save_path):
        self.model.train()
        epoch_losses = []
        epoch_accuracies = []

        for epoch in range(self.num_epochs):
            running_loss = 0.0
            epoch_predictions = []
            epoch_labels_list = []

            train_data_with_progress = tqdm(train_data, desc=f'Epoch {epoch+1}/{self.num_epochs}')

            for inputs, label in train_data_with_progress:
                inputs, label = inputs.to(self.device), label.to(self.device)

                self.optimizer.zero_grad()
                outputs = self.model(inputs)
                loss = self.criterion(outputs, label)
                loss.backward()
                self.optimizer.step()

                running_loss += loss.item() * inputs.size(0)
                predicted_label = self.model.predict(outputs)
                epoch_predictions.extend(predicted_label.tolist())
                epoch_labels_list.extend(label.cpu().tolist())

            # Calculate loss and accuracy after each epoch
            avg_loss = running_loss / len(train_data.dataset)
            accuracy = accuracy_score(epoch_labels_list, epoch_predictions)

            epoch_losses.append(avg_loss)
            epoch_accuracies.append(accuracy)

            print(f'Epoch {epoch+1}/{self.num_epochs}, Loss: {avg_loss:.4f}, Accuracy: {accuracy:.4f}')

        self.save_model(model_save_path)
        return epoch_losses, epoch_accuracies

    def evaluate(self, test_data):
        self.model.eval()
        predictions = []
        labels_list = []

        with torch.no_grad():
            for inputs, label in test_data:
                inputs, label = inputs.to(self.device), label.to(self.device)
                outputs = self.model(inputs)
                predicted_label = self.model.predict(outputs)
                predictions.extend(predicted_label.tolist())
                labels_list.extend(label.cpu().tolist())

        accuracy = accuracy_score(labels_list, predictions)
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
                predicted_label = self.model.predict(outputs)
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
        print(train_confusion_mat)

        print("\nEvaluation on Test Set:")
        test_accuracy, test_precision, test_recall, test_f1, test_confusion_mat = self.calculate_metrics(
            test_loader)
        print(f"Test Accuracy: {test_accuracy:.4f}")
        print(f"Test Precision: {test_precision:.4f}")
        print(f"Test Recall: {test_recall:.4f}")
        print(f"Test F1 Score: {test_f1:.4f}")
        print("Test Confusion Matrix:")
        print(test_confusion_mat)

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


def main(action, savedmodel_wordvector, train_file, num_samples, test_file, input_size, hidden_size, output_size, batch_size, num_epochs=1):

    data_processor = SkipGramTextDataProcessor()
    data_processor.load_word_vectors(savedmodel_wordvector)
    if torch.cuda.is_available():
        map_location = torch.device('cuda')
    else:
        map_location = torch.device('cpu')

    classification_model_file = 'skipgram-classification-model.pt'
    if action == 1:  # Train
        if num_epochs is None:
            num_epochs = int(
                input("Enter the number of epochs for training: "))

        train_data = data_processor.load_dataset(
            train_file, num_samples=num_samples)
        train_dataset = data_processor.preprocess_dataset(
            train_data, batch_size=batch_size)

        skipgram_classifier = Classifier(
            input_size, hidden_size, output_size, num_epochs=num_epochs, lr=0.0005)
        skipgram_losses, skipgram_accuracies = skipgram_classifier.train(
            train_dataset, classification_model_file)

    elif action == 2:  # Test
        test_data = data_processor.load_dataset(test_file)
        test_dataset = data_processor.preprocess_dataset(
            test_data, batch_size=batch_size)
        skipgram_classifier = Classifier(input_size, hidden_size, output_size)
        skipgram_classifier.model.load_state_dict(torch.load(
            f'Saved models/{classification_model_file}', map_location=map_location))
        test_acc = skipgram_classifier.evaluate(test_dataset)

    elif action == 3:  # Evaluate
        test_data = data_processor.load_dataset(test_file)
        test_dataset = data_processor.preprocess_dataset(
            test_data, batch_size=batch_size)
        train_data = data_processor.load_dataset(
            train_file, num_samples=num_samples)
        train_dataset = data_processor.preprocess_dataset(
            train_data, batch_size=batch_size)
        skipgram_classifier = Classifier(input_size, hidden_size, output_size)
        skipgram_classifier.model.load_state_dict(torch.load(
            f'Saved models/{classification_model_file}', map_location=map_location))
        skipgram_classifier.evaluate_performance(train_dataset, test_dataset)
    elif action == 4:  # predict class
        sentence = input("Enter the sentence : ")
        embeddings = data_processor.get_sentence_embeddings(
            sentence).unsqueeze(0)
        skipgram_classifier = Classifier(input_size, hidden_size, output_size)
        skipgram_classifier.model.load_state_dict(torch.load(
            f'Saved models/{classification_model_file}', map_location=map_location))

        predicted_class = skipgram_classifier.predict_class(embeddings)
        predicted_class = int(predicted_class.item())
        print(f"Predicted class: {predicted_class}")
    else:
        print("Invalid action. Please choose 1 for training, 2 for testing, or 3 for evaluation.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Choose action to perform.")
    parser.add_argument(
        "action", type=int, help="1: Train, 2: Test, 3: Evaluate, 4: Predict class of a sentence")
    parser.add_argument("--word_vector_file", type=str,
                        default='Saved models/skip-gram-word-vectors.pt', help="Path to the saved word vectors model")
    parser.add_argument("--train_file", type=str,
                        default="data/train.csv", help="Path to the training data file")
    parser.add_argument("--num_samples", type=int,
                        default=15000, help="Number of samples to use")
    parser.add_argument("--test_file", type=str,
                        default="data/test.csv", help="Path to the testing data file")
    parser.add_argument("--input_size", type=int, default=100,
                        help="Input size for the classifier")
    parser.add_argument("--hidden_size", type=int, default=128,
                        help="Hidden size for the classifier")
    parser.add_argument("--output_size", type=int, default=4,
                        help="Output size for the classifier")
    parser.add_argument("--batch_size", type=int, default=64,
                        help="Batch size for training")
    parser.add_argument("--num_epochs", type=int, default=2,
                        help="Number of epoch for training")

    args = parser.parse_args()
    main(args.action, args.word_vector_file, args.train_file, args.num_samples, args.test_file,
         args.input_size, args.hidden_size, args.output_size, args.batch_size, args.num_epochs)
