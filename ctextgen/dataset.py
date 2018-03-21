import re
from torchtext import data, datasets
import spacy
from torchtext.vocab import GloVe
import os, csv, pdb

class MyDataset:

    def __init__(self, data_dir, filenames, emb_dim=50, mbsize=32 ):
        self.nlp = spacy.load('en')
        print("Filepath = {}".format(data_dir+filenames[0]+'.csv'))
        if not os.path.isfile(data_dir+filenames[0] + ".csv"):
            self.convertTxtToCSV(data_dir, filenames)


        train_file = data_dir + filenames[0] + ".csv"
        validation_file = data_dir +filenames[1] + ".csv"
        test_file = data_dir + filenames[2] + ".csv"

        self.TEXT = data.Field(sequential=False, init_token='<start>', eos_token='<eos>', lower=True, tokenize='spacy', fix_length=16)
        self.LABEL = data.Field(sequential=False, unk_token=None)


        # Only take sentences with length <= 15
        f = lambda ex: len(ex.text) <= 15 and ex.label != 'neutral'
        # pdb.set_trace()


        train = data.TabularDataset(path=train_file, format='csv', fields=[('text', self.TEXT)])
        val = data.TabularDataset(path=validation_file, format='csv', fields=[('text', self.TEXT)])
        test = data.TabularDataset(path=test_file, format='csv', fields=[('text', self.TEXT)])

        # train, val, test = data.TabularDataset.splits(
        #     fields=[('text', self.TEXT), ('label', self.LABEL)],
        #     format='csv', #skip_header=False,
        #     train=train_file, validation=validation_file, test=test_file,
        #     # fine_grained=False, train_subtrees=False,
        #     # filter_pred=f
        # )

        self.TEXT.build_vocab(train, vectors=GloVe('6B', dim=emb_dim))

        self.n_vocab = len(self.TEXT.vocab.itos)
        self.emb_dim = emb_dim

        self.train_iter, self.val_iter, _ = data.BucketIterator.splits(
            (train, val, test), batch_size=mbsize, device=-1, shuffle=True, sort_key=lambda x: len(x.text)
        )

    def convertTxtToCSV(self, data_dir, filenames):
        for filename in filenames:
            with open(data_dir + filename + ".txt", 'r', encoding='utf-8', errors='ignore') as f:
                file = f.read()

            tokens = self.make_tokens(file)

            i = 0
            with open(data_dir + filename + ".csv", 'w', encoding='utf-8', newline='\n') as f:
                wr = csv.writer(f, delimiter=' ',
                                quotechar=' ', escapechar=' ', quoting=csv.QUOTE_NONE)
                wr.writerow(['text'])
                while i < len(tokens):
                    line_count = min(i+20, len(tokens))
                    wr.writerow([" ".join(tokens[i:line_count])])
                    i = line_count

    def tokenizer(self, sentences):
        return [tok.text.lower() for tok in self.nlp.tokenizer(sentences)]

    def prepare_text(self, file):
        # remove excess spaces, \n characters
        file = re.sub(" {1,}", " ", file)
        file = re.sub("\n", " ", file)
        return file

    def make_tokens(self, file):
        file = self.prepare_text(file)
        tokens = self.tokenizer(file)
        for i in range(len(tokens)):
            # sub out special words such as kill'd for killed
            if tokens[i][-2:] == "'d'":
                tokens[i] = tokens[i][:-2] + "ed"
            # fix some spacy specific tokenization bugs around '-'
            if tokens[i][-2:] == '.-':
                tokens[i] = tokens[i][:-2]
                tokens.insert(i, '.')
                tokens.insert(i+1, '-')
            if tokens[i][-2:] == ',-':
                tokens[i] = tokens[i][:-2]
                tokens.insert(i, ',')
                tokens.insert(i+1, '-')
        return tokens

    def get_vocab_vectors(self):
        return self.TEXT.vocab.vectors

    def next_batch(self, gpu=False):
        batch = next(iter(self.train_iter))

        if gpu:
            return batch.text.cuda()

        return batch.text

    def next_validation_batch(self, gpu=False):
        batch = next(iter(self.val_iter))

        if gpu:
            return batch.text.cuda()

        return batch.text

    def idxs2sentence(self, idxs):
        return ' '.join([self.TEXT.vocab.itos[i] for i in idxs])

    def idx2label(self, idx):
        return self.LABEL.vocab.itos[idx]

class SST_Dataset:

    def __init__(self, emb_dim=50, mbsize=32):
        self.TEXT = data.Field(init_token='<start>', eos_token='<eos>', lower=True, tokenize='spacy', fix_length=16)
        self.LABEL = data.Field(sequential=False, unk_token=None)

        # Only take sentences with length <= 15
        f = lambda ex: len(ex.text) <= 15 and ex.label != 'neutral'


        train, val, test = datasets.SST.splits(
            self.TEXT, self.LABEL, fine_grained=False, train_subtrees=False,
            filter_pred=f
        )

        self.TEXT.build_vocab(train, vectors=GloVe('6B', dim=emb_dim))
        self.LABEL.build_vocab(train)

        self.n_vocab = len(self.TEXT.vocab.itos)
        self.emb_dim = emb_dim

        self.train_iter, self.val_iter, _ = data.BucketIterator.splits(
            (train, val, test), batch_size=mbsize, device=-1, shuffle=True
        )

    def get_vocab_vectors(self):
        return self.TEXT.vocab.vectors

    def next_batch(self, gpu=False):
        batch = next(iter(self.train_iter))

        if gpu:
            return batch.text.cuda(), batch.label.cuda()

        return batch.text, batch.label

    def next_validation_batch(self, gpu=False):
        batch = next(iter(self.val_iter))

        if gpu:
            return batch.text.cuda(), batch.label.cuda()

        return batch.text, batch.label

    def idxs2sentence(self, idxs):
        return ' '.join([self.TEXT.vocab.itos[i] for i in idxs])

    def idx2label(self, idx):
        return self.LABEL.vocab.itos[idx]


class IMDB_Dataset:

    def __init__(self, emb_dim=50, mbsize=32):
        self.TEXT = data.Field(init_token='<start>', eos_token='<eos>', lower=True, tokenize='spacy', fix_length=None)
        self.LABEL = data.Field(sequential=False, unk_token=None)

        # Only take sentences with length <= 15
        f = lambda ex: len(ex.text) <= 15

        train, test = datasets.IMDB.splits(
            self.TEXT, self.LABEL, filter_pred=f
        )

        self.TEXT.build_vocab(train, vectors=GloVe('6B', dim=emb_dim))
        self.LABEL.build_vocab(train)

        self.n_vocab = len(self.TEXT.vocab.itos)
        self.emb_dim = emb_dim

        self.train_iter, _ = data.BucketIterator.splits(
            (train, test), batch_size=mbsize, device=-1, shuffle=True
        )

    def get_vocab_vectors(self):
        return self.TEXT.vocab.vectors

    def next_batch(self, gpu=False):
        batch = next(iter(self.train_iter))

        if gpu:
            return batch.text.cuda(), batch.label.cuda()

        return batch.text, batch.label

    def idxs2sentence(self, idxs):
        return ' '.join([self.TEXT.vocab.itos[i] for i in idxs])


class WikiText_Dataset:

    def __init__(self, emb_dim=50, mbsize=32):
        self.TEXT = data.Field(init_token='<start>', eos_token='<eos>', lower=True, tokenize='spacy', fix_length=None)
        self.LABEL = data.Field(sequential=False, unk_token=None)

        train, val, test = datasets.WikiText2.splits(self.TEXT)

        self.TEXT.build_vocab(train, vectors=GloVe('6B', dim=emb_dim))
        self.LABEL.build_vocab(train)

        self.n_vocab = len(self.TEXT.vocab.itos)
        self.emb_dim = emb_dim

        self.train_iter, _, _ = data.BPTTIterator.splits(
            (train, val, test), batch_size=10, bptt_len=15, device=-1
        )

    def get_vocab_vectors(self):
        return self.TEXT.vocab.vectors

    def next_batch(self, gpu=False):
        batch = next(iter(self.train_iter))
        return batch.text.cuda() if gpu else batch.text

    def idxs2sentence(self, idxs):
        return ' '.join([self.TEXT.vocab.itos[i] for i in idxs])
