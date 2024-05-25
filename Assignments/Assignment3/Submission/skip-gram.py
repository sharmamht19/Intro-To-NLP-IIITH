import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
import re
import pandas as pd
import numpy as np
import argparse

class SkipGramNegativeSampling(nn.Module):
    def __init__(self, vocab_size, embedding_dim):
        super(SkipGramNegativeSampling, self).__init__()
        self.embeddings = nn.Embedding(vocab_size, embedding_dim)
        self.out_embeddings = nn.Embedding(vocab_size, embedding_dim)
        self.init_emb()

    def init_emb(self):
        initrange = 0.5 / self.embeddings.weight.size(1)
        self.embeddings.weight.data.uniform_(-initrange, initrange)
        self.out_embeddings.weight.data.uniform_(-0, 0)

    def forward(self, center_words, target_words, neg_words):
        embeds = self.embeddings(center_words)
        out_embeds = self.out_embeddings(target_words)
        neg_embeds = -self.out_embeddings(neg_words)
        
        score = torch.sum(torch.mul(embeds, out_embeds), dim=1)
        score = torch.sigmoid(score)
        
        neg_score = torch.bmm(neg_embeds, embeds.unsqueeze(2)).squeeze()
        neg_score = torch.sum(torch.log(torch.sigmoid(neg_score)), dim=1)
        
        return -torch.mean(torch.log(score) + neg_score)

class SkipGramDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

class SkipGramWord2Vec:
    def __init__(self, embedding_dim=100, num_neg_samples=4, min_count=2, batch_size=64, learning_rate=0.001, num_epochs=5):
        self.embedding_dim = embedding_dim
        self.num_neg_samples = num_neg_samples
        self.min_count = min_count
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.num_epochs = num_epochs
        self.word_to_idx = {}
        self.idx_to_word = {}
        self.word_counts = {}
        self.word_freqs = []
        self.model = None
        self.unk_token = '<UNK>'
        self.unk_index = None  
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def load_dataset(self, file_path, num_samples=10000, reduce_label=True):
        if num_samples:
            data = pd.read_csv(file_path, nrows=num_samples)
        else:
            data = pd.read_csv(file_path)

        if reduce_label and 'Class Index' in data.columns:
            data['Class Index'] -= 1

        corpus = data['Description']
        return corpus

    def preprocess_data(self, corpus):
        cleaned_sentences = []
        for sentence in corpus:
            tokens = re.findall(r'\b\w+\b', sentence.lower())  
            cleaned_sentences.append(tokens)
        return cleaned_sentences

    def build_vocab(self, sentences):
        word_counts = {}
        for sentence in sentences:
            for word in sentence:
                word_counts[word] = word_counts.get(word, 0) + 1
        word_counts = {word: count for word, count in word_counts.items() if count >= self.min_count}
        word_counts[self.unk_token] = 0
        sorted_word_counts = sorted(word_counts.items(), key=lambda x: x[1], reverse=True)
        self.word_to_idx = {word: idx + 1 for idx, (word, _) in enumerate(sorted_word_counts)}  
        self.word_to_idx[self.unk_token] = 0  
        self.idx_to_word = {idx: word for word, idx in self.word_to_idx.items()}
        self.word_counts = {word: count for word, count in sorted_word_counts}
        total_words = sum(word_counts.values())
        self.word_freqs = np.array([count / total_words for count in word_counts.values()])

        self.unk_index = 0

    def build_data(self, sentences, window_size=2):
        data = []
        for sentence in sentences:
            for i, center_word in enumerate(sentence):
                for j in range(max(0, i - window_size), min(len(sentence), i + window_size + 1)):
                    if i != j:
                        data.append((center_word, sentence[j]))
        return data

    def get_neg_sample(self, target_word_idx, num_samples):
        neg_samples = []
        while len(neg_samples) < num_samples:
            sample = np.random.choice(len(self.word_to_idx), size=num_samples, p=self.word_freqs)
            if target_word_idx not in sample:
                neg_samples.extend(sample)
        return torch.LongTensor(neg_samples)

    def save_word_vectors(self, file_path):
        word_to_idx = self.word_to_idx
        idx_to_word = self.idx_to_word
        word_vectors = self.model.embeddings.weight.cpu().detach().numpy()

        state_dict = {
            'word_to_idx': word_to_idx,
            'idx_to_word': idx_to_word,
            'word_vectors': word_vectors
        }

        torch.save(state_dict, file_path)
        print(f"Word vectors and vocabulary mappings saved to {file_path}.")

    def train(self, sentences, window_size=2, save_file='skip-gram-word-vectors.pt'):
        self.build_vocab(sentences)
        data = self.build_data(sentences, window_size)
        dataset = SkipGramDataset(data)
        dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)

        self.model = SkipGramNegativeSampling(len(self.word_to_idx), self.embedding_dim).to(self.device)
        optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)

        for epoch in range(self.num_epochs):
            total_loss = 0.0
            with tqdm(total=len(dataloader), desc=f'Epoch {epoch+1}/{self.num_epochs}', unit='batch') as pbar:
                for center_words, target_words in dataloader:
                    center_word_idxs = [self.word_to_idx.get(word, 0) for word in center_words]
                    target_word_idxs = [self.word_to_idx.get(word, 0) for word in target_words]

                    center_words_tensor = torch.LongTensor(center_word_idxs).to(self.device)
                    target_words_tensor = torch.LongTensor(target_word_idxs).to(self.device)

                    neg_words = [self.get_neg_sample(target_word_idx, self.num_neg_samples) for target_word_idx in target_word_idxs]
                    neg_words_tensor = torch.stack(neg_words).to(self.device)

                    optimizer.zero_grad()
                    loss = self.model(center_words_tensor, target_words_tensor, neg_words_tensor)
                    loss.backward()
                    optimizer.step()
                    total_loss += loss.item()

                    pbar.set_postfix({'loss': total_loss / len(dataloader)})
                    pbar.update()

            # Print the loss for the epoch
            print(f"Epoch {epoch+1}/{self.num_epochs}, Average Loss: {total_loss / len(dataloader)}")

        self.save_word_vectors(save_file)
        print("Training completed.")


    def get_word_vector(self, word):
        if word in self.word_to_idx:
            idx = torch.tensor([self.word_to_idx[word]]).to(self.device)
            return self.model.embeddings(idx).squeeze().cpu().detach().numpy()
        else:
            print(f"Word '{word}' not found in vocabulary. Using <unk> token.")
            idx = torch.tensor([self.unk_index]).to(self.device)
            return self.model.embeddings(idx).squeeze().cpu().detach().numpy()

    def most_similar(self, word, topn=5):
        if word in self.word_to_idx:
            word_vec = self.get_word_vector(word)
            if word_vec is not None:
                all_word_vecs = self.model.embeddings.weight.cpu().detach().numpy()
                similarities = np.dot(all_word_vecs, word_vec)
                top_similar_words = np.argsort(-similarities)[1:topn+1]
                return [(self.idx_to_word[idx], similarities[idx]) for idx in top_similar_words]
        return None

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Train Skip-Gram Word Vectors model")
    parser.add_argument("-d", "--embedding_dim", type=int, default=100, help="Dimensionality of word embeddings (default: 100)")
    parser.add_argument("-n", "--num_neg_samples", type=int, default=4, help="Number of negative samples (default: 4)")
    parser.add_argument("-e", "--num_epochs", type=int, default=1, help="Number of training epochs (default: 1)")
    parser.add_argument("-f", "--file_path", type=str, default="data/train.csv", help="Path to the training file (default: data/train.csv)")
    parser.add_argument("-s", "--num_samples", type=int, default=15000, help="Number of samples to use from the training data (default: 15000)")
    parser.add_argument("-w", "--window_size", type=int, default=2, help="Size of the context window (default: 2)")
    parser.add_argument("-o", "--save_file", type=str, default="skip-gram-word-vectors.pt", help="Path to save the trained model (default: skip-gram-word-vectors.pt)")

    args = parser.parse_args()
    embedding_dim = args.embedding_dim
    num_neg_samples = args.num_neg_samples
    num_epochs = args.num_epochs
    file_path = args.file_path
    num_samples = args.num_samples
    window_size = args.window_size
    save_file = args.save_file

    word2vec_model = SkipGramWord2Vec(embedding_dim=embedding_dim, num_neg_samples=num_neg_samples, num_epochs=num_epochs)
    corpus = word2vec_model.load_dataset(file_path, num_samples=num_samples)
    sentences = word2vec_model.preprocess_data(corpus)

    word2vec_model.train(sentences, window_size=window_size, save_file=save_file)
