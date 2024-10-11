# models.py

import torch.nn.functional as F
import torch
import torch.nn as nn
import torch.optim as optim
from typing import List, Tuple, Dict
from sentiment_data import *
import numpy as np

from tqdm import tqdm
from matplotlib import pyplot as plt

import math
import random
np.random.seed(0)
torch.manual_seed(0)
random.seed(0)


class SentimentClassifier(object):
    """
    Sentiment classifier base type
    """

    def predict(self, ex_words: List[str]) -> int:
        """
        Makes a prediction on the given sentence
        :param ex_words: words to predict on
        :return: 0 or 1 with the label
        """
        raise Exception("Don't call me, call my subclasses")

    def predict_all(self, all_ex_words: List[List[str]]) -> List[int]:
        """
        You can leave this method with its default implementation, or you can override it to a batched version of
        prediction if you'd like. Since testing only happens once, this is less critical to optimize than training
        for the purposes of this assignment.
        :param all_ex_words: A list of all examples to do prediction on
        :return:
        """
        return [self.predict(ex_words) for ex_words in all_ex_words]


class TrivialSentimentClassifier(SentimentClassifier):
    def predict(self, ex_words: List[str]) -> int:
        """
        Always predicts positive sentiment.
        :param ex_words: List of words in the sentence
        :return: 1
        """
        return 1


class NeuralSentimentClassifier(SentimentClassifier):
    """
    Neural sentiment classifier that wraps the CNN model.
    """

    def __init__(self, model, max_length, vocab):
        self.model = model
        self.model.eval()  # Set the model to evaluation mode
        self.sentence_length = max_length
        self.vocab = vocab

    def predict(self, ex_words: List[str]) -> int:
        """
        Predicts the sentiment label for a single sentence.

        :param ex_words: List of words in the sentence.
        :return: Predicted label (0 or 1).
        """
        indices = [self.vocab.get(word, self.vocab['UNK']) for word in ex_words]

        if len(indices) < self.sentence_length:
            indices += [self.vocab['PAD']] * (self.sentence_length - len(indices))
        else:
            indices = indices[:self.sentence_length]
        indices_tensor = torch.tensor(indices, dtype=torch.long).unsqueeze(0)  # Shape: (1, max_length)

        with torch.no_grad():
            outputs = self.model(indices_tensor)  # Shape: (1, output_dim)
            probabilities = F.softmax(outputs, dim=1)
            prediction = torch.argmax(probabilities, dim=1).item()

        return prediction

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


def train_deep_averaging_network(args, train_exs: List[SentimentExample],
                                 dev_exs: List[SentimentExample],
                                 word_embeddings: WordEmbeddings) -> SentimentClassifier:
    """
    Trains a Deep Averaging Network (DAN) for sentiment analysis.

    :param args: Command-line arguments.
    :param train_exs: Training examples.
    :param dev_exs: Development examples.
    :param word_embeddings: Pre-trained word embeddings.
    :return: A trained SentimentClassifier model.
    """

    if args.random_init:
        vocab_size = len(word_embeddings.word_indexer)
        embedding_dim = args.embedding_dim
        model = DeepAveragingNetworkRandom(vocab_size, embedding_dim, args)
    else:
        model = DeepAveragingNetwork(word_embeddings, args)

    model.train()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)


    word_indexer = word_embeddings.word_indexer
    train_sentences = [ex.words for ex in train_exs]
    train_labels = [ex.label for ex in train_exs]
    train_indices = [
        torch.tensor([word_indexer.index_of(word) if word_indexer.contains(word)
                      else word_indexer.index_of("UNK") for word in sentence], dtype=torch.long)
        for sentence in train_sentences
    ]

    train_labels = torch.tensor(train_labels, dtype=torch.long)


    batch_size = args.batch_size

    num_epochs = args.num_epochs
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0.0


        permutation = torch.randperm(len(train_indices))

        for i in range(0, len(train_indices), batch_size):
            optimizer.zero_grad()

            indices = permutation[i:i + batch_size]
            batch_sentences = [train_indices[idx] for idx in indices]
            batch_labels = train_labels[indices]
            batch_sentences_padded = nn.utils.rnn.pad_sequence(
                batch_sentences, batch_first=True, padding_value=0
            )  # Shape: (batch_size, max_seq_length)


            outputs = model(batch_sentences_padded)  # Shape: (batch_size, num_classes)
            loss = criterion(outputs, batch_labels)
            total_loss += loss.item() * len(batch_sentences)
            loss.backward()
            optimizer.step()

        avg_loss = total_loss / len(train_indices)
        print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {avg_loss:.4f}")


    return DANClassifier(model, word_indexer)


class DeepAveragingNetwork(nn.Module):
    def __init__(self, word_embeddings: WordEmbeddings, args):
        super(DeepAveragingNetwork, self).__init__()
        embedding_dim = word_embeddings.get_embedding_length()
        self.embeddings = word_embeddings.get_initialized_embedding_layer(frozen=False)
        self.embeddings.padding_idx = 0

        hidden_size = args.hidden_size


        self.fc1 = nn.Linear(embedding_dim, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, 2)  # Output layer for binary classification

    def forward(self, x):
        """
        Forward pass of the network.

        :param x: Input tensor of shape (batch_size, sequence_length).
        :return: Output tensor of shape (batch_size, num_classes).
        """

        embeds = self.embeddings(x)
        mask = (x != 0).unsqueeze(-1).float()
        embeds = embeds * mask
        sum_embeds = torch.sum(embeds, dim=1)
        lengths = torch.sum(mask, dim=1)
        lengths[lengths == 0] = 1
        avg_embeds = sum_embeds / lengths  # Shape: (batch_size, embedding_dim)
        out = self.fc1(avg_embeds)
        out = self.relu(out)
        out = self.fc2(out)

        return out


class DeepAveragingNetworkRandom(nn.Module):
    def __init__(self, vocab_size, embedding_dim, args):
        super(DeepAveragingNetworkRandom, self).__init__()
        self.embeddings = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        hidden_size = args.hidden_size
        self.fc1 = nn.Linear(embedding_dim, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, 2)  # Output layer for binary classification

    def forward(self, x):

        embeds = self.embeddings(x)

        mask = (x != 0).unsqueeze(-1).float()
        embeds = embeds * mask

        sum_embeds = torch.sum(embeds, dim=1)

        lengths = torch.sum(mask, dim=1)

        lengths[lengths == 0] = 1

        avg_embeds = sum_embeds / lengths
        out = self.fc1(avg_embeds)
        out = self.relu(out)
        out = self.fc2(out)

        return out


class DANClassifier(SentimentClassifier):
    def __init__(self, model, word_indexer):
        self.model = model
        self.model.eval()  # Set the model to evaluation mode
        self.word_indexer = word_indexer

    def predict(self, words: List[str]) -> int:
        """
        Predicts the sentiment label for a single sentence.

        :param words: List of words in the sentence.
        :return: Predicted label (0 or 1).
        """
        indices = [self.word_indexer.index_of(word) if self.word_indexer.contains(word)
                   else self.word_indexer.index_of("UNK") for word in words]

        indices_tensor = torch.tensor(indices, dtype=torch.long).unsqueeze(0)  # Shape: (1, seq_length)

        with torch.no_grad():
            outputs = self.model(indices_tensor)  # Shape: (1, num_classes)
            probabilities = nn.functional.softmax(outputs, dim=1)
            prediction = torch.argmax(probabilities, dim=1).item()

        return prediction

class CNN(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        embed_dim: int,
        num_filters: int,
        filter_size: int,
        output_dim: int,
        dropout: float,
        pretrained_embeddings: torch.FloatTensor = None,
        freeze_embeddings: bool = True,
    ):
        super(CNN, self).__init__()

        if pretrained_embeddings is not None:
            self.embedding = nn.Embedding.from_pretrained(
                pretrained_embeddings, freeze=freeze_embeddings, padding_idx=0
            )
        else:
            self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)


        self.conv = nn.Conv1d(
            in_channels=embed_dim,
            out_channels=num_filters,
            kernel_size=filter_size,
            padding=0,  # No padding to keep it simple
        )
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(num_filters, output_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the CNN.

        :param x: Input tensor of shape (batch_size, sequence_length)
        :return: Output tensor of shape (batch_size, output_dim)
        """
        embedded = self.embedding(x)  # Shape: (batch_size, seq_length, embed_dim)
        embedded = embedded.permute(0, 2, 1)  # Shape: (batch_size, embed_dim, seq_length)

        conv_out = self.conv(embedded)  # Shape: (batch_size, num_filters, seq_length - filter_size + 1)
        activated = F.relu(conv_out)  # Apply ReLU

        # Max-over-time pooling
        pooled = F.max_pool1d(
            activated, kernel_size=activated.shape[2]
        )  # Shape: (batch_size, num_filters, 1)
        pooled = pooled.squeeze(2)  # Shape: (batch_size, num_filters)

        dropped = self.dropout(pooled)  # Apply dropout

        logits = self.fc(dropped)  # Shape: (batch_size, output_dim)

        return logits


def train_CNN(
    args,
    train_exs: List[SentimentExample],
    dev_exs: List[SentimentExample],
    word_embeddings: WordEmbeddings,
    configurations: List[Tuple[int, int]] = [(3, 100), (4, 150), (5, 200)],
    dropout: float = 0.5,
) -> Dict[str, NeuralSentimentClassifier]:
    """
    Trains multiple one-layer CNNs with varying filter sizes and numbers of filters for sentiment analysis.

    :param args: Command-line arguments
    :param train_exs: Training examples
    :param dev_exs: Development examples
    :param word_embeddings: Pre-trained word embeddings
    :param configurations: List of tuples, each containing (filter_size, num_filters)
    :param dropout: Dropout rate
    :return: Dictionary mapping configuration identifiers to their trained NeuralSentimentClassifier models
    """
    print("\nTraining CNN models with multiple configurations...")

    # Parameters
    vocab = word_embeddings.word_indexer.objs_to_ints
    vocab_size = len(vocab)
    embed_dim = word_embeddings.get_embedding_length()
    output_dim = 2  # Binary classification
    max_length = max(len(ex.words) for ex in train_exs + dev_exs)
    pad_token_index = vocab['PAD']
    unk_token_index = vocab['UNK']

    # Device configuration
    device = torch.device(args.device if hasattr(args, 'device') else 'cpu')
    print(f"Using device: {device}")


    def prepare_batch(examples: List[SentimentExample], max_length: int) -> Tuple[torch.Tensor, torch.Tensor]:
        batch_indices = []
        labels = []
        for ex in examples:
            indices = [vocab.get(word, unk_token_index) for word in ex.words]
            if len(indices) < max_length:
                indices += [pad_token_index] * (max_length - len(indices))
            else:
                indices = indices[:max_length]
            batch_indices.append(indices)
            labels.append(ex.label)
        batch_tensor = torch.tensor(batch_indices, dtype=torch.long).to(device)
        labels_tensor = torch.tensor(labels, dtype=torch.long).to(device)
        return batch_tensor, labels_tensor


    trained_models = {}
    performance_logs = {}


    for idx, (filter_size, num_filters) in enumerate(configurations, start=1):
        config_id = f"Config {idx}"  # Naming as Config 1, Config 2, Config 3
        print(f"\n--- Training {config_id} (Filter Size: {filter_size}, Number of Filters: {num_filters}) ---")

        # Initialize the model
        model = CNN(
            vocab_size=vocab_size,
            embed_dim=embed_dim,
            num_filters=num_filters,
            filter_size=filter_size,
            output_dim=output_dim,
            dropout=dropout,
            pretrained_embeddings=torch.FloatTensor(word_embeddings.vectors),
            freeze_embeddings=False,  # Allow fine-tuning of embeddings
        ).to(device)
        print(model)


        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=args.lr)


        num_epochs = args.num_epochs
        batch_size = args.batch_size

        # Lists to store loss and accuracy for plotting
        train_losses = []
        dev_losses = []
        dev_accuracies = []

        for epoch in range(num_epochs):
            model.train()
            total_loss = 0.0


            random.shuffle(train_exs)


            for i in range(0, len(train_exs), batch_size):
                batch_exs = train_exs[i : i + batch_size]
                batch_inputs, batch_labels = prepare_batch(batch_exs, max_length)


                outputs = model(batch_inputs)  # Shape: (batch_size, output_dim)
                loss = criterion(outputs, batch_labels)


                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                total_loss += loss.item() * len(batch_exs)

            avg_train_loss = total_loss / len(train_exs)
            train_losses.append(avg_train_loss)
            print(f"Epoch [{epoch + 1}/{num_epochs}], Training Loss: {avg_train_loss:.4f}")


            model.eval()
            with torch.no_grad():
                dev_inputs, dev_labels = prepare_batch(dev_exs, max_length)
                dev_outputs = model(dev_inputs)  # Shape: (dev_size, output_dim)
                dev_loss = criterion(dev_outputs, dev_labels).item()
                dev_losses.append(dev_loss)

                _, predicted = torch.max(dev_outputs, 1)
                correct = (predicted == dev_labels).sum().item()
                accuracy = correct / len(dev_exs)
                dev_accuracies.append(accuracy)

                print(
                    f"Epoch [{epoch + 1}/{num_epochs}], Dev Loss: {dev_loss:.4f}, Dev Accuracy: {accuracy:.4f}"
                )


        trained_models[config_id] = NeuralSentimentClassifier(model, max_length, vocab)


        performance_logs[config_id] = {
            "train_losses": train_losses,
            "dev_losses": dev_losses,
            "dev_accuracies": dev_accuracies,
        }


    plt.figure(figsize=(12, 6))
    for config_id, logs in performance_logs.items():
        plt.plot(
            range(1, num_epochs + 1),
            logs["dev_losses"],
            label=f"{config_id} Dev Loss",
            marker='o'
        )
    plt.legend()
    plt.title("Development Loss over Epochs for Different CNN Configurations")
    plt.xlabel("Epoch")
    plt.ylabel("Dev Loss")
    plt.grid(True)
    plt.show()

    # Plot Dev Accuracy
    plt.figure(figsize=(12, 6))
    for config_id, logs in performance_logs.items():
        plt.plot(
            range(1, num_epochs + 1),
            logs["dev_accuracies"],
            label=f"{config_id} Dev Accuracy",
            marker='o'
        )
    plt.legend()
    plt.title("Development Accuracy over Epochs for Different CNN Configurations")
    plt.xlabel("Epoch")
    plt.ylabel("Dev Accuracy")
    plt.grid(True)
    plt.show()

    # Optionally, plot Training Loss as well
    plt.figure(figsize=(12, 6))
    for config_id, logs in performance_logs.items():
        plt.plot(
            range(1, num_epochs + 1),
            logs["train_losses"],
            label=f"{config_id} Train Loss",
            marker='o'
        )
    plt.legend()
    plt.title("Training Loss over Epochs for Different CNN Configurations")
    plt.xlabel("Epoch")
    plt.ylabel("Training Loss")
    plt.grid(True)
    plt.show()

    return trained_models
