#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import torch
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence

# Dataset class
class QADataset(Dataset):
    def __init__(self, questions, answers):
        self.questions = questions
        self.answers = answers

    def __len__(self):
        return len(self.questions)

    def __getitem__(self, idx):
        return self.questions[idx], self.answers[idx]

# Model definition
class QAModel(torch.nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, output_dim, dropout_rate, num_layers=2):
        super(QAModel, self).__init__()
        self.embedding = torch.nn.Embedding(vocab_size, embedding_dim)
        self.lstm = torch.nn.LSTM(embedding_dim, hidden_dim, batch_first=True, dropout=dropout_rate, num_layers=num_layers)
        self.fc = torch.nn.Linear(hidden_dim, output_dim)

    def forward(self, text):
        embedded = self.embedding(text)
        lstm_out, (hidden, _) = self.lstm(embedded)
        hidden = hidden[-1]  # Using the last layer's hidden state
        output = self.fc(hidden)
        return output

# Helper function to convert text to sequence
def text_to_sequence(text_list):
    sequences = [torch.tensor([ord(char) % 256 for char in text]) for text in text_list]
    return pad_sequence(sequences, batch_first=True, padding_value=0)

