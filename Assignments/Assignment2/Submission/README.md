
# NLP Assignment - 2

---

## Description

This assignment implements a part-of-speech (POS) tagger using a feedforward neural network (FFN) and a recurrent neural network (RNN). The POS tagger takes input sentences and predicts the POS tags for each word in the sentence.

---

## Execution Instructions

### Running the Code

1. Ensure you have Python installed on your system.
2. Navigate to the directory containing the code files.
3. Run the following command to execute the POS tagger:

   ```
   python pos_tagger.py <m_type> [<value>]
   ```

   Replace `<m_type>` with `'f'` for FFN or `'r'` for RNN. `<value>` is optional and can be `0` for testing or `1` for training.

### Loading the Pretrained Model

- Pretrained Models

   Pretrained models for the POS tagger are available for download from the following link:

   [Download Pretrained Models](https://drive.google.com/drive/folders/1beKwDhxwC_icCcREi8PmXsk6xgQfBq45?usp=sharing)

   To access the models, follow these steps:

   1. Click on the link provided above to open the Google Drive folder.
   2. From the folder, download the files containing the pretrained models.

- If using the RNN model (`<m_type> = 'r'`), the pretrained model file `model_epoch.pt` should be present in the directory.
- Ensure the model file is loaded automatically during execution.

### Input Format

- When prompted, enter a sentence to be processed by the POS tagger.

### Output Format

- The POS tags for each word in the input sentence will be printed in the format:

  ```
  word POS_tag
  ```

- POS tags are represented using the Universal POS tag set.

### Exiting the Program

- To exit the program, enter `'quit'` when prompted for a sentence.

---

## Implementation Assumptions

1. The POS tagger uses the Universal POS tag set for tagging.
2. The `conllu` library is used to parse CoNLL-U formatted data.
3. The paths for the training, testing, and development datasets are assumed to be as follows:
   - Training data: `en_atis-ud-train.conllu`
   - Testing data: `en_atis-ud-dev.conllu`
   - Development data: `en_atis-ud-test.conllu`

---

## Dependencies

- Python 3.x
- PyTorch
- NumPy
- tqdm
- pyconll
- torchtext
- pandas
