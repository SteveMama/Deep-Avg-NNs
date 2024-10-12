# Sentiment Classifier using CNN and DAN

## Overview

This project implements a sentiment classification model using two primary architectures:
- **Convolutional Neural Network (CNN)** for sentiment analysis with multiple filter sizes.
- **Deep Averaging Network (DAN)**, a simpler neural network that averages word embeddings and uses a fully connected neural network for classification.

The models can predict the sentiment of text data based on pre-trained word embeddings such as GloVe.

## Files
- `models.py`: Contains the model architectures (CNN, DAN) and helper functions for training and evaluation.
- `sentiment_data.py`: Provides utility functions for reading data and word embeddings.
- `neural_sentiment_classifier.py`: The main script that handles training, evaluation, and prediction for both CNN and DAN models.

## Dependencies

- Python 3.x
- PyTorch
- NumPy
- Matplotlib
- tqdm

Install required libraries using:

```bash
pip install torch numpy matplotlib tqdm
```

## Model Architectures

### CNN (Convolutional Neural Network)
A one-layer CNN with multiple configurations (different filter sizes and numbers of filters) is trained to classify sentiment. The model uses a convolutional layer followed by max-pooling and a fully connected layer.

### DAN (Deep Averaging Network)
DAN is a simpler model that averages the word embeddings in the sentence and passes them through a fully connected network with ReLU activation. It is used for binary classification.

## Usage

### Training and Evaluation

To train the model, use `neural_sentiment_classifier.py` with the following command:

```bash
python neural_sentiment_classifier.py --model <MODEL_TYPE> --train_path <TRAIN_FILE> --dev_path <DEV_FILE> --word_vecs_path <EMBEDDING_FILE> --num_epochs 10 --batch_size 32 --lr 0.001
```

Arguments:
- `--model`: Choose between `CNN`, `DAN`, or `TRIVIAL`.
- `--train_path`: Path to the training data file.
- `--dev_path`: Path to the development data file.
- `--word_vecs_path`: Path to the word embeddings file (e.g., GloVe).
- `--num_epochs`: Number of epochs for training.
- `--batch_size`: Batch size for training.
- `--lr`: Learning rate for optimization.

### Prediction
To predict sentiments on the test set, use the `--test_output_path` flag:

```bash
python neural_sentiment_classifier.py --model CNN --test_output_path test-blind.output.txt
```

The predictions will be saved in the specified output file.

## Example Commands

### Train and evaluate DAN
```bash
python neural_sentiment_classifier.py --model DAN --train_path data/train.txt --dev_path data/dev.txt --word_vecs_path data/glove.6B.300d.txt --num_epochs 10 --batch_size 32 --lr 0.001
```

### Train and evaluate CNN with multiple configurations
```bash
python neural_sentiment_classifier.py --model CNN --train_path data/train.txt --dev_path data/dev.txt --word_vecs_path data/glove.6B.300d.txt --num_epochs 10 --batch_size 32 --lr 0.001
```

## Visualization
After training, the system will plot the following for each CNN configuration:
- Development loss over epochs
- Development accuracy over epochs
- Training loss over epochs

These plots will help you track the model's performance across different configurations.

## Dataset
The dataset should be in a tab-separated format with each line containing:
- A label (`0` for negative sentiment, `1` for positive sentiment)
- A sentence.

Example:
```
1    I love this movie!
0    I hated the ending.
```

## Word Embeddings
Pre-trained word embeddings (e.g., GloVe) are used for input representation. You can provide a path to the embeddings file using the `--word_vecs_path` argument.

## Contact
For questions or issues, feel free to reach out!