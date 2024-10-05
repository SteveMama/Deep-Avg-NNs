import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from typing import List
from sentiment_data import *
import numpy as np

class SentimentClassifier(object):
    """
    Base class for sentiment classifiers.
    """
    def predict(self, words: List[str]) -> int:
        """
        Predicts the sentiment label for a given sentence.

        :param words: List of words in the sentence.
        :return: 1 if positive sentiment, 0 if negative sentiment.
        """
        raise Exception("Don't call me, call my subclasses")

    def predict_all(self, sentences: List[List[str]]) -> List[int]:
        """
        Predicts sentiment labels for a list of sentences.

        :param sentences: List of sentences, each a list of words.
        :return: List of predictions.
        """
        return [self.predict(words) for words in sentences]

class TrivialSentimentClassifier(SentimentClassifier):
    """
    A trivial classifier that always predicts positive sentiment.
    """
    def predict(self, words: List[str]) -> int:
        return 1


class customDataset(torch.utils.data.Dataset):
    def __init__(self, examples: List[SentimentExample], word_embeddings : WordEmbeddings):
        self.examples = examples
        self.word_embeddings = word_embeddings
        self.word_indexer = word_embeddings.word_indexer
        self.embedding_dim = word_embeddings.get_embedding_length()

        if not self.word_indexer.contains("UNK"):
            unk_vector = np.zeroes(self.embedding_dim)
            self.word_indexer.add_and_get_index("UNK")
            self.word_embeddings.add_embedding("UNK", unk_vector)


    def __len__(self):
        return len(self.examples)

    def __getitem__(self, item_index):
        ex = self.examples[item_index]
        words = ex.words
        label = ex.label

        embeddings = []
        for word in words:
            embedding = self.word_embeddings.get_embedding(word)
            if embedding is None:
                embedding = self.word_embeddings.get_embedding("UNK")
            embeddings.append(embedding)
        embeddings = torch.tensor(embeddings, dtype=torch.float)

        avg_embedding = embeddings.mean(dim=0)

        return avg_embedding, torch.tensor(label, dtype=torch.long)


class DAN(nn.Module):
    def __init__(self, embedding_dim, args):
        super(DAN, self).__init__()

        hidden_size = args.hidden_size #hidden layer size
        self.fc1 = nn.Linear(embedding_dim, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, 2) #binary classification


    def forward(self, avg_embedding):

        output = self.fc1(avg_embedding)
        output = self.relu(output)
        logits = self.fc2(output)

        return logits

class DANClassifier(SentimentClassifier):

    def __init__(self, model, word_embeddings):
        self.model = model
        self.model.eval()
        self.word_embeddings = word_embeddings
        self.word_indexer = word_embeddings.word_indexer
        self.embedding_dim = word_embeddings.get_embedding_length()

    def predict(self, words: List[str]) -> int:

        embeddings = []
        for word in words:
            embedding = self.word_embeddings.get_embedding(word)
            if embedding is None:
                embedding = self.word_embeddings.get_embedding("UNK")
            embeddings.append(embedding)
        embeddings = torch.tensor(embeddings, dtype=torch.float)

        avg_embeddings = embeddings.mean(dim=0).unsqueeze(0)

        with torch.no_grad():
            logits = self.model(avg_embeddings)
            cls_prob = F.softmax(logits, dim=1)
            pred = torch.argmax(cls_prob, dim=1).item()

        return pred


def train_deep_averaging_network(args, train_exs: List[SentimentExample],
                                 dev_exs: List[SentimentExample],
                                 word_embeddings: WordEmbeddings) -> SentimentClassifier:

    train_dataset = customDataset(train_exs, word_embeddings)
    dev_dataset = customDataset(dev_exs, word_embeddings)

    train_dataset = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle = True
    )

    dev_dataset = torch.utils.data.DataLoader(
        dev_dataset, batch_size = args.batch_size, shuffle = False
    )

    embedding_dim = word_embeddings.get_embedding_length()
    model = DAN(embedding_dim, args)
    model.train()


    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr = args.lr)

    num_epochs = args.num_epochs
    for epoch in range(num_epochs):
        total_loss = 0.0

        for avg_embeddings, labels in train_dataset:
            optimizer.zero_grad()
            logits = model(avg_embeddings)
            loss = criterion(logits, labels)
            total_loss += loss.item()
            loss.backward()
            optimizer.step()

        avg_loss = total_loss/ len(train_dataset)
        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {avg_loss:.4f}")

        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for avg_embeddings, labels in dev_dataset:
                logits = model(avg_embeddings)
                predictions = torch.argmax(logits, dim=1)
                correct += (predictions == labels).sum().item()
                total += labels.size(0)
        accuracy = correct / total
        print(f"Dev Accuracy: {accuracy:.4f}")
        model.train()

    return DANClassifier(model, word_embeddings)

