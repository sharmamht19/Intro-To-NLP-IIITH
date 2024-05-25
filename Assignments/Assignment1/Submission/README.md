# Assignment1 - 2022201060

## Tokenizer

This script tokenizes input sentences using regular expressions.It came also be used by other python files by importing.

### Usage

```bash
python3 tokenizer.py
```

When prompted, enter a sentence, and the script will output the tokenized sentences.

## Language Model

This script calculates the probability of a given sentence using language models. It is also use to analysis of different Language models with different corpuses.

### Usage

```bash
python3 language_model.py <lm_type> <corpus_path>
```

- `<lm_type>`: Choose 'g' for Good-Turing Smoothing Model or 'i' for Interpolation Model.
- `<corpus_path>`: Path to the corpus file.

When prompted, enter a sentence, and the script will output the probability score.

## Generator

This script generates k number of candidates for the next word in a sentence.

#### Usage

```bash
python3 generator.py <lm_type> <corpus_path> <k>
```

- `<lm_type>`: Choose 'g' for Good-Turing Smoothing Model or 'i' for Interpolation Model.
- `<corpus_path>`: Path to the corpus file.
- `<k>`: Number of candidates for the next word.

When prompted, enter a sentence, and the script will output the most probable next words along with their probability scores.

### Experiments

Implemented language models with the following configurations:

**Pride and Prejudice Corpus:**

1. **LM 1:** Tokenization + 3-gram LM + Good-Turing Smoothing

   ```bash
   python3 language_model.py g '../corpus/Pride and Prejudice - Jane Austen.txt' 1
   ```

2. **LM 2:** Tokenization + 3-gram LM + Linear Interpolation

   ```bash
   python3 language_model.py i '../corpus/Pride and Prejudice - Jane Austen.txt' 2
   ```

**Ulysses Corpus:**

3. **LM 3:** Tokenization + 3-gram LM + Good-Turing Smoothing

   ```bash
   python3 language_model.py g '../corpus/Ulysses James Joyce.txt' 3
   ```

4. **LM 4:** Tokenization + 3-gram LM + Linear Interpolation

   ```bash
   python3 language_model.py i '../corpus/Ulysses James Joyce.txt' 4
   ```
