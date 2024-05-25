import sys
import os
from collections import Counter
import json
import random
import math
from tokenizer import TextTokenizer
import numpy as np
from scipy import stats


def BuildNGram(n: int, txt: str, tokenizer):
    nGramContext = {}  # dictionaries to store N-gram context
    nGramCounter = {}  # dictionaries to store N-gram count

    # Tokenizing the input text
    sentences = tokenizer.tokenizer(txt)

    for sentence in sentences:
        sentence_lower = [word.lower() for word in sentence]
#         print(n)
        if (n - 2 > 0):
            sentence = ["<sos>"] * (n - 2) + sentence_lower
        else:
            sentence = sentence_lower

        # Tokenize the sentence
        tokens = sentence
#         print(tokens)
        # Iterate through N-grams in the sentence
        for i in range(len(tokens) - n + 1):
            context = " ".join(tokens[i:i + n - 1])
            target_word = tokens[i + n - 1]
            total_ngram = context + " " + target_word if context else target_word

            # Update N-gram counters
            if total_ngram in nGramCounter:
                nGramCounter[total_ngram] += 1
            else:
                nGramCounter[total_ngram] = 1

                # Update N-gram context
                if context in nGramContext:
                    nGramContext[context].append(target_word)
                else:
                    nGramContext[context] = [target_word]

    return nGramContext, nGramCounter

# class GoodTuringSmoothing:
#     def __init__(self, ngram_counter):
#         self.ngram_counter = ngram_counter
#         self.total_ngrams = float(sum(ngram_counter.values()))
#         self.adjusted_counts = self.calculate_adjusted_counts()

#     def count_of_counts(self):
#         count_of_counts_dict = {}
#         for count in self.ngram_counter.values():
#             count_of_counts_dict[count] = count_of_counts_dict.get(
#                 count, 0) + 1
#         return count_of_counts_dict

#     def calculate_adjusted_counts(self):
#         adjusted_counts = {}
#         count_of_counts_dict = self.count_of_counts()

#         # for count in self.ngram_counter.values():
#         #     Nc_plus_1 = count_of_counts_dict.get(count + 1, 1)
#         #     Nc = count_of_counts_dict.get(count, 1)
#         #     c_star = (count + 1) * (Nc_plus_1 / Nc)
#         #     adjusted_counts[count] = c_star
#         # print(count_of_counts_dict)

#         Nrs = dict(sorted(count_of_counts_dict.items()))

#         # print(Nrs)
#         observed_frequency, frequency_of_frequency = list(
#             Nrs.keys()), list(Nrs.values())
#         self.slope, self.intercept = self.smooth(observed_frequency, frequency_of_frequency)
#         # print(self.slope , self.intercept)
#         for count in self.ngram_counter.values():
#             # Nc = count_of_counts_dict.get(count, 1)
#             if count == 0:
#                 c_star = 1.0*(count + 1) * self.smooth_Nr(count+1)
#             else:
#                 c_star = (count + 1) * \
#                     (self.smooth_Nr(count+1) / self.smooth_Nr(count))
#             adjusted_counts[count] = c_star

#         return adjusted_counts

#     def smooth_Nr(self, count):
#         return np.exp(self.intercept + self.slope*np.log(count))

#     def smooth(self, observed_frequency, frequency_of_frequency):
#         # q is not present
#         log_r = [np.log(observed_frequency[0])]
#         log_Zr = [np.log(frequency_of_frequency[0] /
#                          (0.5 * observed_frequency[1]))]
        
#         n = len(observed_frequency)
#         # print(n)
#         for r in range(1, n - 1):
#             Nr = frequency_of_frequency[r]
#             t = observed_frequency[r + 1]
#             q = observed_frequency[r - 1]
#             y = np.log(Nr / (0.5 * (t - q)))
#             x = np.log(observed_frequency[r])
#             log_Zr.append(y)
#             log_r.append(x)

#         # t is not present
#         log_r.append(np.log(observed_frequency[-1]))
#         log_Zr.append(np.log(
#             frequency_of_frequency[-1] / (observed_frequency[-1] - observed_frequency[-2])))

#         slope, interception, _, _, _ = stats.linregress(
#             log_r, log_Zr)

#         return slope, interception

#     def smooth_ngram_probs(self):
#         self.smoothed_ngram_probs = {}

#         for ngram, count in self.ngram_counter.items():
#             c_star = self.adjusted_counts.get(count, 1)
#             probability = c_star / self.total_ngrams
#             self.smoothed_ngram_probs[ngram] = probability

#         self.smoothed_ngram_probs["<oov>"] = 0.0001

#         return self.smoothed_ngram_probs

#     def get_probability(self, trigram):
#         prob = self.smoothed_ngram_probs.get(
#             trigram, self.smoothed_ngram_probs.get("<oov>", 0.0001))
#         return prob

class GoodTuringSmoothing:
    def __init__(self, ngram_counter,vocab):
        self.ngram_counter = ngram_counter
        self.total_ngrams = float(sum(ngram_counter.values()))
        self.adjusted_counts = self.calculate_adjusted_counts()
        self.vocabulary = vocab
    def count_of_counts(self):
        count_of_counts_dict = {}
        for count in self.ngram_counter.values():
            count_of_counts_dict[count] = count_of_counts_dict.get(
                count, 0) + 1
        return count_of_counts_dict

    def calculate_adjusted_counts(self):
        adjusted_counts = {}
        count_of_counts_dict = self.count_of_counts()

        # for count in self.ngram_counter.values():
        #     Nc_plus_1 = count_of_counts_dict.get(count + 1, 1)
        #     Nc = count_of_counts_dict.get(count, 1)
        #     c_star = (count + 1) * (Nc_plus_1 / Nc)
        #     adjusted_counts[count] = c_star
        # print(count_of_counts_dict)

        Nrs = dict(sorted(count_of_counts_dict.items()))

        # print(Nrs)
        observed_frequency, frequency_of_frequency = list(
            Nrs.keys()), list(Nrs.values())
        self.slope, self.intercept = self.smooth(observed_frequency, frequency_of_frequency)
        # print(self.slope , self.intercept)
        for count in self.ngram_counter.values():
            # Nc = count_of_counts_dict.get(count, 1)
            if count == 0:
                c_star = 1.0*(count + 1) * self.smooth_Nr(count+1)
            else:
                c_star = (count + 1) * \
                    (self.smooth_Nr(count+1) / self.smooth_Nr(count))
            adjusted_counts[count] = c_star

        return adjusted_counts

    def smooth_Nr(self, count):
        return np.exp(self.intercept + self.slope*np.log(count))

    def smooth(self, observed_frequency, frequency_of_frequency):
        # q is not present
        log_r = [np.log(observed_frequency[0])]
        log_Zr = [np.log(frequency_of_frequency[0] /
                         (0.5 * observed_frequency[1]))]
        
        n = len(observed_frequency)
        # print(n)
        for r in range(1, n - 1):
            Nr = frequency_of_frequency[r]
            t = observed_frequency[r + 1]
            q = observed_frequency[r - 1]
            y = np.log(Nr / (0.5 * (t - q)))
            x = np.log(observed_frequency[r])
            log_Zr.append(y)
            log_r.append(x)

        # t is not present
        log_r.append(np.log(observed_frequency[-1]))
        log_Zr.append(np.log(
            frequency_of_frequency[-1] / (observed_frequency[-1] - observed_frequency[-2])))

        slope, interception, _, _, _ = stats.linregress(
            log_r, log_Zr)

        return slope, interception

    def smooth_ngram_probs(self):
        self.smoothed_ngram_probs = {}

        for ngram, count in self.ngram_counter.items():
            c_star = self.adjusted_counts.get(count, 1)
            c_star_sum = 0.0
            word1, word2, _ = ngram.split(maxsplit=2)

            for word3 in self.vocabulary:
                gram = word1 + " " + word2 + " " + word3
                c = self.ngram_counter.get(gram , 0)
                t_c_star = self.adjusted_counts.get(c, 1)
                c_star_sum += t_c_star
            probability = c_star / c_star_sum
            self.smoothed_ngram_probs[ngram] = probability

        self.smoothed_ngram_probs["<oov>"] = 0.0001

        return self.smoothed_ngram_probs

    def get_probability(self, trigram):
        prob = self.smoothed_ngram_probs.get(
            trigram, self.smoothed_ngram_probs.get("<oov>", 0.0001))
        return prob

class LinearInterpolationSmoothing:
    def __init__(self, trigram_counts, bigram_counts, unigram_counts, is_train=False, lambdas=None):
        self.trigram_counts = trigram_counts
        self.bigram_counts = bigram_counts
        self.unigram_counts = unigram_counts
        self.total_trigrams = 0
        self.total_bigrams = 0
        self.total_unigrams = 0
        self.is_train = is_train
        self.lambdas = lambdas

    def update_lambdas(self):
        updated_lambdas = {
            f'lambda{i}': self.lambdas[f'lambda{i}'] for i in range(1, 4)}
        total_unigram = float(sum(self.unigram_counts.values()))

        for trigram, trigram_count in self.trigram_counts.items():
            t1, t2, t3 = trigram.split()

            # Check if the frequency of the trigram is greater than 0
            if trigram_count > 0:
                # Calculate the three conditions
                condition1 = (trigram_count - 1) / max(1,
                                                       self.bigram_counts.get(f'{t1} {t2}', 0) - 1)
                condition2 = (self.bigram_counts.get(
                    f'{t2} {t3}', 0) - 1) / max(1, self.unigram_counts.get(t2, 0) - 1)
                condition3 = (self.unigram_counts.get(
                    t3, 0) - 1) / max(1, total_unigram - 1)

                # Find the index of the maximum condition
                max_condition_index = max(
                    enumerate([condition1, condition2, condition3]), key=lambda x: x[1])[0]

                # Update lambdas based on the maximum condition
                updated_lambdas[f'lambda{max_condition_index + 1}'] += trigram_count

        # Normalize the lambdas to ensure they sum to 1
        lambda_sum = sum(updated_lambdas.values())
        normalized_lambdas = {
            key: value / lambda_sum for key, value in updated_lambdas.items()}

        return normalized_lambdas

    def linear_interpolation_smoothing(self):
        self.total_trigrams = sum(self.trigram_counts.values())
        self.total_bigrams = sum(self.bigram_counts.values())
        self.total_unigrams = sum(self.unigram_counts.values())

        interpolated_probs = {}

        if self.is_train:
            self.lambdas = {f'lambda{i}': 0.0 for i in range(1, 4)}
            self.lambdas = self.update_lambdas()

        for trigram, count in self.trigram_counts.items():
            t1, t2, t3 = trigram.split()

            # Unigram probability
            P1_t3 = self.unigram_counts.get(t3, 0) / self.total_unigrams

            # Bigram probability
            P2_t3 = self.bigram_counts.get(
                f"{t2} {t3}", 0) / self.total_bigrams

            # Trigram probability
            P3_t3 = count / self.total_trigrams

            # Interpolation weights
            lambda1 = self.lambdas.get('lambda1', 1/3.0)
            lambda2 = self.lambdas.get('lambda2', 1/3.0)
            lambda3 = self.lambdas.get('lambda3', 1/3.0)

            # Linear interpolation
            interpolated_prob = lambda1 * P1_t3 + lambda2 * P2_t3 + lambda3 * P3_t3

            interpolated_probs[trigram] = interpolated_prob

        if self.is_train:
            return interpolated_probs, self.lambdas
        else:
            return interpolated_probs

    def get_probability(self, trigram):
        #         print(trigram)
        t1, t2, t3 = trigram.split()

        # Unigram probability
        P1_t3 = self.unigram_counts.get(t3, 0) / self.total_unigrams

        # Bigram probability
        P2_t3 = self.bigram_counts.get(f"{t2} {t3}", 0) / self.total_bigrams

        # Trigram probability
        P3_t3 = self.trigram_counts.get(trigram, 0) / self.total_trigrams
#         print(P1_t3,P2_t3,P3_t3)
        # Interpolation weights
        lambda1 = self.lambdas.get('lambda1', 1/3.0)
        lambda2 = self.lambdas.get('lambda2', 1/3.0)
        lambda3 = self.lambdas.get('lambda3', 1/3.0)
#         print(lambda1,lambda2, lambda3)
        # Linear interpolation
        interpolated_prob = lambda1 * P1_t3 + lambda2 * P2_t3 + lambda3 * P3_t3
        if (interpolated_prob == 0):
            interpolated_prob = 0.00001
        return interpolated_prob

def generate_ngram_model(N, corpus_path, tokenizer):

    # Read the corpus from the file
    with open(corpus_path, 'r', encoding='utf-8') as file:
        corpus = file.read()

    # Generate N-grams using the BuildNGram function
    nGramContext, nGramCounter = BuildNGram(N, corpus, tokenizer)

    return nGramContext, nGramCounter

class LanguageModel:
    def __init__(self, corpus_path, result_name="2022201060_LM1", n_gram_order=3, smoothing_type="g", lambdas=None):
        self.n_gram_order = n_gram_order
        self.smoothing_type = smoothing_type
        self.lambdas = lambdas
        self.corpus_path = corpus_path
        self.save_file_path = self._get_save_file_path(0)
        self.train_corpus_path = self._get_save_file_path(1)
        self.test_corpus_path = self._get_save_file_path(2)
        self.test_samples_count = 1000
        self.result_name = result_name
        self.nGramContext = None
        self.nGramCounter = None
        self.unigram_counter = None
        self.probs = None
        self.smoothing_instance = None  # Instance to hold the smoothing object
        self.tokenizer = TextTokenizer()  # Instance to hold the TextTokenizer object

    def _get_save_file_path(self, file_type):
        # Extract the base name of the corpus path
        corpus_name = os.path.basename(
            self.corpus_path) if self.corpus_path else "corpus_unknown.txt"

        if file_type == 0:
            return f"{self.n_gram_order}_{self.smoothing_type}_{corpus_name}.json"
        elif file_type == 1:
            return f"{self.n_gram_order}_{self.smoothing_type}_{corpus_name}_train.txt"
        elif file_type == 2:
            return f"{self.n_gram_order}_{self.smoothing_type}_{corpus_name}_test.txt"
        else:
            raise ValueError(
                "Invalid file_type. Use 0 for save_file, 1 for train_corpus, and 2 for test_corpus.")

    def setup(self, corpus_path=None):
        # If corpus_path is not provided, use the path stored in self.corpus_path
        if corpus_path is None:
            corpus_path = self.corpus_path

        # Read the corpus from the file
        with open(corpus_path, 'r', encoding='utf-8') as file:
            corpus = file.read()

        # Split the corpus into sentences
        sentences = self.tokenizer.split_sentences(corpus)
        # Exclude sentences with zero words
        sentences = [sentence for sentence in sentences if len(
            sentence.split()) > 0]
#         for s in sentences:
#             print(s)
#             print("--------------------")
        # Randomly select 1000 sentences for testing
        selected_sentences = random.sample(
            sentences, min(self.test_samples_count, len(sentences)))

        # Write the selected sentences to the test corpus file
        with open(self.test_corpus_path, 'w', encoding='utf-8') as test_file:
            test_file.write("\n".join(selected_sentences))

        # Write the remaining sentences to the train corpus file
        remaining_sentences = [
            sentence for sentence in sentences if sentence not in selected_sentences]
        with open(self.train_corpus_path, 'w', encoding='utf-8') as train_file:
            train_file.write("\n".join(remaining_sentences))

#         print(f"Setup complete. Train corpus: {self.train_corpus_path}, Test corpus: {self.test_corpus_path}")

    def train(self, corpus_path=None):
        if not corpus_path:
            corpus_path = self.corpus_path

        self.nGramContext, self.nGramCounter = generate_ngram_model(
            self.n_gram_order, corpus_path, self.tokenizer)
#         print(self.nGramContext[1])
        unigram_context, self.unigram_counter = generate_ngram_model(
            1, corpus_path, self.tokenizer)
        self.vocabulary = set(self.unigram_counter.keys())
        if self.smoothing_type == "g":
            smoothing_instance = GoodTuringSmoothing(self.nGramCounter, self.vocabulary)
            self.probs = smoothing_instance.smooth_ngram_probs()
        elif self.smoothing_type == "i":
            bigram_context, bigram_counter = generate_ngram_model(
                2, corpus_path, self.tokenizer)
            smoothing_instance = LinearInterpolationSmoothing(
                self.nGramCounter, bigram_counter, self.unigram_counter, is_train=True, lambdas=self.lambdas)
            self.probs, self.lambdas = smoothing_instance.linear_interpolation_smoothing()
        else:
            raise ValueError(
                "Invalid smoothing type. Choose 'g' for Good-Turing or 'i' for Linear Interpolation.")

        # Store the smoothing instance for future reference
        self.smoothing_instance = smoothing_instance

    def save(self):
        # Save all necessary variables to a JSON file
        model_state = {
            'n_gram_order': self.n_gram_order,
            'probs': self.probs,
            'lambdas': self.lambdas,
            'nGramContext': self.nGramContext,
            'nGramCounter': self.nGramCounter,
        }

        with open(self.save_file_path, 'w') as file:
            json.dump(model_state, file)

    def load(self):
        # Load all necessary variables from a JSON file
        with open(self.save_file_path, 'r') as file:
            model_state = json.load(file, object_hook=self.json_object_hook)

        self.n_gram_order = model_state['n_gram_order']
        self.probs = model_state['probs']
        self.lambdas = model_state['lambdas']
        self.nGramContext = model_state['nGramContext']
        self.nGramCounter = model_state['nGramCounter']

    def json_object_hook(self, dct):
        # Replace the key 'null' with None during JSON deserialization
        return {key if key != 'null' else None: value for key, value in dct.items()}

    def calculate_probability(self, sentence):
        probability = 1.0
        tokenized_sentence = self.tokenizer.sentence_tokenizer(sentence)
        if (self.n_gram_order - 2 > 0):
            tokenized_sentence = ["<sos>"] * \
                (self.n_gram_order - 2) + tokenized_sentence
        # print(tokenized_sentence)
        # total_words = len(tokenized_sentence) - 2
        for i in range(self.n_gram_order - 1, len(tokenized_sentence)):
            context = " ".join(
                tokenized_sentence[max(0, i - self.n_gram_order + 1): i])
            target_word = tokenized_sentence[i]
            n_gram = f"{context} {target_word}" if context else target_word

            # Calculate log likelihood based on the model probabilities
#             print(n_gram)
            prob = self.smoothing_instance.get_probability(n_gram)
            # print(n_gram, prob)
            probability *= prob
        # p ,_ = self.perplexity(tokenized_sentence)
        # print(p)
        return probability

    def perplexity(self, tokenized_sentence):
        total_words = len(tokenized_sentence) - 2  # Exclude <SOS> and <EOS>
        log_likelihood_sentence = 0.0
#         print(tokenized_sentence)
        for i in range(self.n_gram_order - 1, len(tokenized_sentence)):
            context = " ".join(
                tokenized_sentence[max(0, i - self.n_gram_order + 1): i])
            target_word = tokenized_sentence[i]
            n_gram = f"{context} {target_word}" if context else target_word

            # Calculate log likelihood based on the model probabilities
#             print(n_gram)
            prob = self.smoothing_instance.get_probability(n_gram)
#             print(prob)
            log_likelihood_sentence += math.log(prob)

        perplexity_sentence = 2 ** (-log_likelihood_sentence / total_words)
        return perplexity_sentence, log_likelihood_sentence

    def evaluate_helper(self, result_file_path, corpus_path):
        # Read the corpus from the file
        with open(corpus_path, 'r', encoding='utf-8') as file:
            corpus = file.read()

        sentences = self.tokenizer.split_sentences(corpus)
        # Exclude sentences with zero words
#         sentences = [sentence for sentence in sentences if len(sentence.split()) > 0]

#         print(sentences)
        total_words = 0
        log_likelihood_sum = 0.0
        perplexity_scores = []

        with open(result_file_path, 'w', encoding='utf-8') as result_file:
            # Initialize average perplexity
            average_perplexity = None

            for sentence in sentences:
                tokenized_sentence = self.tokenizer.sentence_tokenizer(
                    sentence)
                if len(tokenized_sentence) > self.n_gram_order:
                    perplexity_sentence, log_likelihood_sentence = self.perplexity(
                        tokenized_sentence)
                    perplexity_scores.append(perplexity_sentence)
    #                 print(sentence , " : perplexity  score : ",perplexity_sentence )

                    # Write results to the file
                    result_file.write(f"{sentence}\t{perplexity_sentence}\n")

                    log_likelihood_sum += log_likelihood_sentence
                    total_words += len(sentence.split())

            if total_words > 0:
                # Calculate average perplexity
                average_perplexity = 2 ** (-log_likelihood_sum / total_words)

            # Move the cursor to the beginning of the file
            result_file.seek(0)

            # Write average perplexity at the first line of the file
            result_file.write(f"avg_perplexity\t{average_perplexity}\n")
        return average_perplexity

    def evaluate(self, train=True, test=True):
        if train:
            corpus_path = self.train_corpus_path
            result_file_path = f"{self.result_name}_train-perplexity.txt"
            avg_perp = self.evaluate_helper(result_file_path, corpus_path)
            print("Average Perplexity on Train Set:", avg_perp)
        if test:
            corpus_path = self.test_corpus_path
            result_file_path = f"{self.result_name}_test-perplexity.txt"
            avg_perp = self.evaluate_helper(result_file_path, corpus_path)
            print("Average Perplexity on Test Set:", avg_perp)
        if (not train and not test):
            raise ValueError("Either train or test flag should be True.")


if __name__ == "__main__":
    # Check if the correct number of command-line arguments is provided
    if len(sys.argv) < 3:
        print("Usage: python3 language_model.py <lm_type> <corpus_path>")
        sys.exit(1)

    # Extract command-line arguments
    lm_type = sys.argv[1]
    corpus_path = sys.argv[2]

    if len(sys.argv) == 4:
        model_num = sys.argv[3]
        result_name = "2022201060_LM" + str(model_num)
        # Create an instance of the LanguageModel class
        lm_model = LanguageModel(
            corpus_path, result_name, smoothing_type=lm_type)
        lm_model.setup()
        lm_model.train()
        lm_model.evaluate()
        sys.exit(0)
    else:
        # Create an instance of the LanguageModel class
        lm_model = LanguageModel(corpus_path, smoothing_type=lm_type)
        lm_model.setup()
        lm_model.train(corpus_path)

    while True:
        # Prompt for user input
        input_sentence = input("input sentence: ")
        if (input_sentence == ""):
            break
        # Calculate the probability of the input sentence
        probability_score = lm_model.calculate_probability(input_sentence)

        # Print the probability score
        print("score:", probability_score)
