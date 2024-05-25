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
from torch.nn.utils.rnn import pad_sequence
import argparse

# Define default values for variables
DEFAULT_TRAIN_DATA_FILE = "data/train.csv"
DEFAULT_TEST_DATA_FILE = "data/test.csv"
DEFAULT_EMBEDDINGS_FILE_LOCATION = "data/glove.6B.100d.txt"
DEFAULT_NUM_SAMPLES = None
DEFAULT_NUM_EPOCHS = 1
DEFAULT_BATCH_SIZE = 16
DEFAULT_LEARNING_RATE = 0.001
DEFAULT_EMBEDDING_DIM = 100
DEFAULT_EMBEDDING_SIZE = 100
DEFAULT_HIDDEN_SIZE = 50
DEFAULT_DROPOUT = 0.5
DEFAULT_LOAD_ELMO = False
DEFAULT_IS_CUDA = False

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

class ELMoTrainer:
    def __init__(self, model, train_loader, val_loader, criterion, optimizer, device='cuda' if torch.cuda.is_available() else 'cpu'):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.criterion = criterion
        self.optimizer = optimizer
        self.device = device
        self.model.to(self.device)

    def train(self, num_epochs):
        for epoch in range(num_epochs):
            train_loss = self._train_epoch()
#             val_loss = self._validate_epoch()
#             print(f'Epoch {epoch + 1}/{num_epochs}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}')
            print(
                f'Epoch {epoch + 1}/{num_epochs}, Train Loss: {train_loss:.4f}')

    def _train_epoch(self):
        self.model.train()
        total_loss = 0.0
        for sentence, label in tqdm(self.train_loader):
            inp = sentence[:, :-1].to(self.device)
            targ = sentence[:, 1:].to(self.device)
#             print("Input shape:", inp.shape)
#             print("Target shape:", targ.shape)
            self.optimizer.zero_grad()
            output = self.model(inp)
#             print("Output shape:", output.shape)
            loss = self.criterion(output, targ)
            loss.backward()
            self.optimizer.step()
            total_loss += loss.item()
        return total_loss / len(self.train_loader)

    def _validate_epoch(self):
        self.model.eval()
        total_loss = 0.0
        with torch.no_grad():
            for sent, lab in tqdm(self.val_loader):
                input_val = sent[:, :-1].to(self.device)
                target_val = sent[:, 1:].to(self.device)
                opt = self.model(input_val)

                loss_val = self.criterion(opt, target_val)
                total_loss += loss_val.item()
        return total_loss / len(self.val_loader)

    def save_model(self, filepath):
        torch.save(self.model.state_dict(), filepath)

    def load_model(self, filepath):
        self.model.load_state_dict(torch.load(
            filepath, map_location=self.device))

    def get_embeddings(self, words):
        with torch.no_grad():
            self.model.eval()
            if isinstance(words, str):  # If a single word
                words = [words]
            word_indices = [self.model.word_to_idx[word]
                            if word in self.model.word_to_idx else self.model.unk_index for word in words]
            word_tensor = torch.LongTensor(
                word_indices).unsqueeze(0).to(self.device)
            embeddings = self.model.embedding(word_tensor)
            lstm1_output, _ = self.model.layer_1(embeddings)
            lstm2_output, _ = self.model.layer_2(lstm1_output)
            elmo_representations = self.model.combine_embeddings(
                lstm1_output, lstm2_output)
            return elmo_representations.squeeze(0).cpu().numpy() if isinstance(words, str) else elmo_representations.cpu().numpy()

    def get_similar_words(self, word, top_n=5):
        with torch.no_grad():
            self.model.eval()
            word_embedding = self.get_embeddings(word)
            embeddings = self.model.embedding.weight.cpu().numpy()
            similarities = cosine_similarity(
                word_embedding.reshape(1, -1), embeddings)
            most_similar_indices = similarities.argsort()[0][-top_n:][::-1]
            most_similar_words = [self.model.idx_to_word[idx]
                                  for idx in most_similar_indices]
            return most_similar_words

def custom_collate_fn(batch):
    sorted_batch = sorted(batch, key=lambda x: len(x[0]), reverse=True)
    sentences, labels = zip(*sorted_batch)

    # Pad sequences
    padded_sentences = pad_sequence(
        [torch.tensor(sent) for sent in sentences], batch_first=True, padding_value=0)

    # Convert labels to tensors
    label_tensors = [torch.tensor(label) for label in labels]

    return padded_sentences, torch.stack(label_tensors)

def load_elmo_model(checkpoint_path,device, vocab_size=0, embedding_dim=100, hidden_size=128, dropout=0.5, embedding_matrix = None):
    # device = 'cuda' if torch.cuda.is_available() else 'cpu'
    # Instantiate the ELMo model architecture
    elmo_model = ELMo(vocab_size = vocab_size, embedding_size=embedding_dim, hidden_size=hidden_size, dropout=dropout, embeddings = embedding_matrix)
    elmo_model = nn.DataParallel(elmo_model)  # If trained using DataParallel
    # Load the model checkpoint
    elmo_model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    # Set the model to evaluation mode
    elmo_model.eval()
    print("ELMo Loaded successfully.")
    return elmo_model


def main():
    # Create ArgumentParser object
    parser = argparse.ArgumentParser(description="ELMo Model Training Script")

    # Add arguments
    parser.add_argument("--train_data_file", type=str, default=DEFAULT_TRAIN_DATA_FILE,
                        help="Path to the training data file")
    parser.add_argument("--test_data_file", type=str, default=DEFAULT_TEST_DATA_FILE,
                        help="Path to the testing data file")
    parser.add_argument("--embeddings_file_location", type=str, default=DEFAULT_EMBEDDINGS_FILE_LOCATION,
                        help="Path to the embeddings file")
    parser.add_argument("--num_samples", type=int, default=DEFAULT_NUM_SAMPLES,
                        help="Number of samples to use from the data files")
    parser.add_argument("--num_epochs", type=int, default=DEFAULT_NUM_EPOCHS,
                        help="Number of epochs for training")
    parser.add_argument("--batch_size", type=int, default=DEFAULT_BATCH_SIZE,
                        help="Batch size for training")
    parser.add_argument("--learning_rate", type=float, default=DEFAULT_LEARNING_RATE,
                        help="Learning rate for training")
    parser.add_argument("--embedding_dim", type=int, default=DEFAULT_EMBEDDING_DIM,
                        help="Dimension of word embeddings")
    parser.add_argument("--embedding_size", type=int, default=DEFAULT_EMBEDDING_SIZE,
                        help="Size of word embeddings")
    parser.add_argument("--hidden_size", type=int, default=DEFAULT_HIDDEN_SIZE,
                        help="Size of hidden layer in the model")
    parser.add_argument("--dropout", type=float, default=DEFAULT_DROPOUT,
                        help="Dropout probability")
    parser.add_argument("--load_elmo", action="store_true", default=DEFAULT_LOAD_ELMO,
                        help="Whether to load a pre-trained ELMo model")
    parser.add_argument("--is_cuda", action="store_true", default=DEFAULT_IS_CUDA,
                        help="Whether to use CUDA if available")

    # Parse command-line arguments
    args = parser.parse_args()

    # Set variables using parsed arguments
    train_data_file = args.train_data_file
    test_data_file = args.test_data_file
    embeddings_file_location = args.embeddings_file_location
    num_samples = args.num_samples
    num_epochs = args.num_epochs
    batch_size = args.batch_size
    learning_rate = args.learning_rate
    embedding_dim = args.embedding_dim
    embedding_size = args.embedding_size
    hidden_size = args.hidden_size
    dropout = args.dropout
    load_elmo = args.load_elmo
    is_cuda = args.is_cuda

    # Set device based on CUDA availability
    device = torch.device("cuda" if is_cuda and torch.cuda.is_available() else "cpu")
    print("Device : ",device)

    # Define the datasets
    train_dataset = CustomDataset(train_data_file)
    test_dataset = CustomDataset(test_data_file)

    # Load and preprocess data for both datasets
    train_dataset.load_and_preprocess_data(
        num_samples=num_samples)  # Load all training data
    test_dataset.load_and_preprocess_data(
        num_samples=num_samples)   # Load all testing data

    # Build vocabulary and create embedding matrix
    vocab_builder = VocabBuilder()
    vocab_builder.build_vocab(train_dataset.data)
    print("Vocabulary building Done.")
    vocab_builder.load_glove_embeddings(embeddings_file_location)
    embedding_matrix = vocab_builder.create_embedding_matrix()
    print("Embedding matrix building Done.")

    # Update corpus with vocabulary indices
    train_dataset.update_corpus(vocab_builder.vocab)
    test_dataset.update_corpus(vocab_builder.vocab)

    # Print vocab size and embedding matrix shape
    print("Vocabulary size:", len(vocab_builder.vocab))
    print("Embedding matrix shape:", embedding_matrix.shape)

    vocab_size = len(vocab_builder.vocab)

    train_data_loader = DataLoader(
        train_dataset, batch_size=batch_size, collate_fn=custom_collate_fn, shuffle=True)
    test_data_loader = DataLoader(
        test_dataset, batch_size=batch_size, collate_fn=custom_collate_fn, shuffle=True)

    if load_elmo:    
        elmo_model = load_elmo_model('saved_models/elmo_model.pth',device, vocab_size = vocab_size,hidden_size = hidden_size,  embedding_matrix = embedding_matrix)
    else:
        elmo_model = ELMo(vocab_size=vocab_size, embedding_size=embedding_size,
                        hidden_size=hidden_size, dropout=dropout, embeddings=embedding_matrix)
        elmo_model = nn.DataParallel(elmo_model)

        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(elmo_model.parameters(), lr=learning_rate)

        elmo_trainer = ELMoTrainer(model=elmo_model, train_loader=train_data_loader,
                                val_loader=test_data_loader, criterion=criterion, optimizer=optimizer, device=device)

        # Train the model
        elmo_trainer.train(num_epochs)
        print("model trained..")
        elmo_trainer.save_model('elmo_model.pth')
        print("model saved.")

if __name__ == "__main__":
    main()

# To run:
#   python3 ELMO.py --train_data_file data/train.csv --test_data_file data/test.csv --embeddings_file_location data/glove.6B.100d.txt --num_samples 1000 --num_epochs 5 --batch_size 32 --learning_rate 0.001 --embedding_dim 100 --embedding_size 100 --hidden_size 50 --dropout 0.5 --load_elmo --is_cuda
