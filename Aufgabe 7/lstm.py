"""
P7 Tools - Aufgabe 8
Sentiment Prediction using LSTM

Gruppe:
Marcel Braasch
Nadja Seeberg
Sinem Kühlewind (geb. Demiraslan)
"""

import torch
import torchtext.legacy.data as data
import torch.nn as nn


class LSTM:

    def build_datasets(self, data_path):
        labels = data.Field()
        text = data.Field()

        train, val, test = data.TabularDataset.splits(
            path=data_path,
            train="sentiment.train.tsv",
            validation="sentiment.dev.tsv",
            test="sentiment.test.tsv",
            format='tsv',
            fields=[('labels', labels),
                    ('text', text)])

        labels.build_vocab(train, test, val)
        text.build_vocab(train, test, val)

        # Testweises printen der Indices
        print(text.vocab.stoi["because"])
        print(text.vocab.itos[172])

        return train, test, val

    def create_iterator(self, data_set):
        # TODO-NS: Hier ggf. noch weitere Parameter übergeben? (z.B. "device")
        bucket_iterator = data.BucketIterator(
            dataset=data_set,
            # Ohne den sort_key ist es bei mir nicht durchgelaufen - hoffe es passt wenn wir nach Textlänge sortieren
            sort_key=lambda x: len(x.text),
            batch_size=32)

        return bucket_iterator

    def lstm(self):

        no_layers = 2
        #laut VL sollen wir das Vokabular auf die 5000 häufigsten Wörter reduzieren
        vocabulary_size = 5000
        #TODO-NS: Welche Embedding/ Hidden Dimensionen?
        hidden_dim = 0
        embedding_dim = 0
        output_dim = 0
        dropout_factor = 0.3

        embedding = nn.Embedding(vocabulary_size, embedding_dim)
        lstm = nn.LSTM(input_size=embedding_dim,
                       hidden_size=hidden_dim,
                       num_layers=no_layers,
                       dropout=dropout_factor)

        #TODO-NS: Average Pooling/Max Pooling

        ll = nn.Linear(in_features=embedding_dim,
                       out_features=output_dim)


if __name__ == '__main__':
    lstm = LSTM()
    train, val, test = lstm.batch_data("./data/")
    it = lstm.create_iterator(train)
