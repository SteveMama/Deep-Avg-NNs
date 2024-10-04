# models.py

import torch
import torch.nn as nn
import numpy as np
from sentiment_data import *
from utils import *
import random

class DeepAveragingNetworkClassifier(nn.Module):
    def __init__(self, word_embeddings, hidden_size):
        super(DeepAveragingNetworkClassifier, self).__init__()
        self.embedding = word_embeddings.get_initialized_embedding_layer(frozen=False)
        embedding_dim = word_embeddings.get_embedding_length()
        self.fc1 = nn.Linear(embedding_dim, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, 2)  # 2 classes: negative and positive
        self.log_softmax = nn.LogSoftmax(dim=1)

    def forward(self, inputs):
        # inputs: tensor of shape (batch_size, seq_len)
        embeddings = self.embedding(inputs)  # Shape: (batch_size, seq_len, embedding_dim)
        avg_embeddings = embeddings.mean(dim=1)  # Shape: (batch_size, embedding_dim)
        hidden = self.relu(self.fc1(avg_embeddings))  # Shape: (batch_size, hidden_size)
        output = self.fc2(hidden)  # Shape: (batch_size, 2)
        log_probs = self.log_softmax(output)  # Shape: (batch_size, 2)
        return log_probs

class NeuralSentimentClassifier:
    def __init__(self, model, word_embeddings):
        self.model = model
        self.word_embeddings = word_embeddings
        self.model.eval()

    def predict(self, words):
        word_indices = [self.word_embeddings.word_indexer.index_of(word) if self.word_embeddings.word_indexer.contains(word)
                        else self.word_embeddings.word_indexer.index_of("UNK") for word in words]
        inputs = torch.tensor([word_indices], dtype=torch.long)
        with torch.no_grad():
            log_probs = self.model(inputs)
            predicted = torch.argmax(log_probs, dim=1)
            return predicted.item()

    def predict_all(self, sentences):
        all_predictions = []
        for words in sentences:
            prediction = self.predict(words)
            all_predictions.append(prediction)
        return all_predictions

def train_deep_averaging_network(args, train_exs, dev_exs, word_embeddings):
    model = DeepAveragingNetworkClassifier(word_embeddings, args.hidden_size)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    loss_function = nn.NLLLoss()
    batch_size = args.batch_size  # Use batch_size from args
    for epoch in range(args.num_epochs):
        total_loss = 0.0
        random.shuffle(train_exs)
        model.train()
        for i in range(0, len(train_exs), batch_size):
            batch_exs = train_exs[i:i+batch_size]
            inputs, targets = prepare_batch(batch_exs, word_embeddings)
            model.zero_grad()
            log_probs = model(inputs)
            loss = loss_function(log_probs, targets)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"Epoch {epoch+1}, Loss: {total_loss}")
    return NeuralSentimentClassifier(model, word_embeddings)

def prepare_batch(batch_exs, word_embeddings):
    # Prepare inputs and targets for a batch
    max_len = max(len(ex.words) for ex in batch_exs)
    batch_size = len(batch_exs)
    padded_inputs = []
    targets = []
    pad_index = word_embeddings.word_indexer.index_of("PAD")
    unk_index = word_embeddings.word_indexer.index_of("UNK")
    for ex in batch_exs:
        word_indices = [word_embeddings.word_indexer.index_of(word) if word_embeddings.word_indexer.contains(word)
                        else unk_index for word in ex.words]
        padded_word_indices = word_indices + [pad_index] * (max_len - len(word_indices))
        padded_inputs.append(padded_word_indices)
        targets.append(ex.label)
    inputs = torch.tensor(padded_inputs, dtype=torch.long)
    targets = torch.tensor(targets, dtype=torch.long)
    return inputs, targets

class TrivialSentimentClassifier:
    def predict(self, words):
        return 1  # Always predicts positive

    def predict_all(self, sentences):
        return [1 for _ in sentences]
