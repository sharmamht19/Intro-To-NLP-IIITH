import re
class TextTokenizer:
    def __init__(self):
        self.mailid_regex = r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'
        self.url_regex = r'https?://\S+|www\.\S+'
        self.mention_regex = r'@\w+'
        self.hashtag_regex = r'#\w+'
        self.number_regex = r'\b\d+\b'
        self.word_regex = r'\w+'
        self.punctuation_regex = r'[^\w\s]'
        self.sentence_regex = r'(?<=\.|\?|\s)\s+'

    def split_sentences(self, text):
        sentenced_text = re.split(self.sentence_regex, text)
        return sentenced_text

    def sentence_tokenizer(self, sentence):
        # Replacing mails
        sentence = re.sub(self.mailid_regex, "<MAILID>", sentence)

        # Replacing URLs
        sentence = re.sub(self.url_regex, "<URL>", sentence)

        # Replacing mentions
        sentence = re.sub(self.mention_regex, "<MENTION>", sentence)

        # Replacing hashtags
        sentence = re.sub(self.hashtag_regex, "<HASHTAG>", sentence)

        # Replacing numbers
        sentence = re.sub(self.number_regex, "<NUM>", sentence)
        # print(sentence)
#         Word tokenizer
        words = re.findall(self.word_regex, sentence)
        # print(words)
        # Append End of sentence and start of sentence to the list of words
        tokenized_sentence = ["<sos>"] + words + ["<eos>"]


#         # Word tokenizer including punctuation
#         words_with_punctuation = re.findall(r'\S+|\s+', sentence)


#         # Remove leading and trailing whitespaces
#         words_with_punctuation = [word.strip() for word in words_with_punctuation]

#         # Append End of sentence and start of sentence to the list of words
#         tokenized_sentence = ["<sos>"] + words_with_punctuation + ["<eos>"]

        return tokenized_sentence

    def tokenizer(self, text):
        # Split the given text into sentences.
        sentenced_text = self.split_sentences(text)
#         print(sentenced_text)
        # Final tokenized tokens.
        tokenized_sentences = []

        for sentence in sentenced_text:
            tokenized_sentences.append(self.sentence_tokenizer(sentence))

        return tokenized_sentences

if __name__ == "__main__":
    tokenizer = TextTokenizer()
    user_input = input("Your text: ")
    tokenized_text = tokenizer.tokenizer(user_input)
    print("Tokenized text:", tokenized_text[-1])