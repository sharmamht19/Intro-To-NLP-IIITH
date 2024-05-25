import pandas as pd
import torch
from sklearn.feature_extraction.text import CountVectorizer
from scipy.sparse import coo_matrix as cm
import numpy as np
from sklearn.decomposition import TruncatedSVD
import argparse

class SVDWordVectors:
    def __init__(self, train_file, num_samples=10000, k=100, save_path='svd_word_vectors.pt', context_size=2):
        self.train_file = train_file
        self.num_samples = num_samples
        self.k = k
        self.save_path = save_path
        self.context_size = context_size
        self.word_vectors = None
        self.vectorizer = None
        self.vocabulary = None
        self.word_to_index = None  # Mapping from word to index
        self.unknown_token = '<UNK>'
        self.unknown_vector = None  # Initialize unknown vector

    def load_dataset(self, file_path, num_samples=None):
        if num_samples:
            data = pd.read_csv(file_path, nrows=num_samples)
        else:
            data = pd.read_csv(file_path)
        return data

    def preprocess_data(self, data):
        corpus = data['Description']
        return corpus

    def build_co_occurrence_matrix(self, corpus):
        self.vectorizer = CountVectorizer()
        X = self.vectorizer.fit_transform(corpus)
        num_words = len(self.vectorizer.vocabulary_)

        co_occurrence_matrix = np.zeros((num_words, num_words), dtype=np.int32)

        for doc in corpus:
            tokens = doc.split()
            for i, target_word in enumerate(tokens):
                target_index = self.vectorizer.vocabulary_.get(target_word)
                if target_index is None:
                    continue

                # Iterate over the context window around the target word
                for j in range(max(0, i - self.context_size), min(len(tokens), i + self.context_size + 1)):
                    if j != i:
                        context_word = tokens[j]
                        context_index = self.vectorizer.vocabulary_.get(context_word)
                        if context_index is not None:
                            co_occurrence_matrix[target_index, context_index] += 1

        coo_matrix = cm(co_occurrence_matrix)

        self.vocabulary = self.vectorizer.vocabulary_
        self.vocabulary[self.unknown_token] = len(self.vocabulary)

        self.unknown_vector = torch.zeros(self.k)

        print("Size of CountVectorizer vocabulary:", len(self.vocabulary))
        print("Shape of X matrix:", X.shape)
        print("Shape of co-occurrence matrix (coo_matrix):", coo_matrix.shape)
        print("Shape of unknown vector:", self.unknown_vector.shape)

        return coo_matrix

    def apply_svd(self, coo_matrix):
        svd = TruncatedSVD(self.k, n_iter=10)
        word_vectors_svd = svd.fit_transform(coo_matrix)
        return word_vectors_svd

    def train(self):
        data = self.load_dataset(self.train_file, self.num_samples)
        corpus = self.preprocess_data(data)
        coo_matrix = self.build_co_occurrence_matrix(corpus)
        self.word_vectors = self.apply_svd(coo_matrix)
        self.word_vectors = torch.tensor(self.word_vectors)  # Convert numpy array to PyTorch tensor
        self.word_vectors = torch.cat((self.word_vectors, self.unknown_vector.unsqueeze(0)), dim=0)
        self.word_vectors = self.word_vectors.to(torch.float32)  # Convert tensor to torch.float32

        # Create mapping from word to index
        self.word_to_index = {word: idx for word, idx in self.vocabulary.items()}

        self.save_word_vectors(self.word_vectors, self.word_to_index)  # Save both word vectors and mapping

    def save_word_vectors(self, word_vectors, word_to_index):
        # Save both word vectors and mapping
        torch.save({'word_vectors': word_vectors, 'word_to_idx': word_to_index}, self.save_path)

    def load_word_vectors(self):
        # Load both word vectors and mapping
        checkpoint = torch.load(self.save_path)
        self.word_vectors = checkpoint['word_vectors']
        self.word_to_index = checkpoint['word_to_idx']

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description="Train SVD Word Vectors model")
    parser.add_argument("-f", "--train_file", type=str, default="data/train.csv", help="Path to the training file (CSV format)")
    parser.add_argument("-k", "--dimension", type=int, default=100, help="Dimensionality of word vectors (default: 100)")
    parser.add_argument("-s", "--save_path", type=str, default="svd_word_vectors.pt", help="Path to save the trained model (default: svd_word_vectors.pt)")
    parser.add_argument("-c", "--context_size", type=int, default=2, help="Size of the context window (default: 5)")
    parser.add_argument("-n", "--num_samples", type=int, default=None, help="Number of samples to use from the training data (default: None)")

    args = parser.parse_args()
    train_file = args.train_file
    k = args.dimension
    save_path = args.save_path
    context_size = args.context_size * 2 + 1
    num_samples = args.num_samples

    svd_model = SVDWordVectors(train_file=train_file, k=k, save_path=save_path, context_size=context_size, num_samples=num_samples)
    svd_model.train()
    print("word vectors saved.")
