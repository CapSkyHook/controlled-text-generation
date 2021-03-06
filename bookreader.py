import os, glob, pdb, re
from torchtext import data

class BookReader(data.Dataset):

    name = ''
    dirname = ''
    urls = []

    @staticmethod
    def sort_key(ex):
        return len(ex.text)

    def __init__(self, path, text_field, label_field, **kwargs):
        """Create a dataset instance given a path to book .txts and fields.

        Arguments:
            path: Path to the dataset's highest level directory
            text_field: The field that will be used for text data.
            label_field: The field that will be used for label data.
            Remaining keyword arguments: Passed to the constructor of
                data.Dataset.
        """
        fields = [('text', text_field), ('label', label_field)]
        examples = []
        # pdb.set_trace()
        blob = glob.iglob(os.path.join(path, '*.txt'))
        for fname in glob.iglob(os.path.join(path, '*.txt')):
            with open(fname, 'r', encoding='utf-8') as f:
                file, label = f.read(), None
                # remove excess spaces, \n characters
                file = re.sub(" {1,}", " ", file)
                file = re.sub("\n{1,}", "\n ", file)
                for text in file.split("\n"):
                    examples.append(data.Example.fromlist([text, label], fields))

        super(BookReader, self).__init__(examples, fields, **kwargs)

    @classmethod
    def splits(cls, text_field, label_field, root='data',
               train='train', test='test', **kwargs):
        """Create dataset objects for splits of the provided books.
        Arguments:
           text_field: The field that will be used for the sentence.
           label_field: The field that will be used for label data.
           root: Root dataset storage directory. Default is '.data'.
           train: The directory that contains the training examples
           test: The directory that contains the test examples
           Remaining keyword arguments: Passed to the splits method of
               Dataset.
        """
        return super(BookReader, cls).splits(
            root=root, text_field=text_field, label_field=label_field,
            train=train, validation=None, test=test, **kwargs)

    @classmethod
    def iters(cls, batch_size=32, device=1, root='.data', vectors=None, **kwargs):
        """Creater iterator objects for splits of the books dataset.
        Arguments:
           batch_size: Batch_size
           device: Device to create batches on. Use - 1 for CPU and None for
               the currently active GPU device.
           root: The root directory that contains the books dataset subdirectory
           vectors: one of the available pretrained vectors or a list with each
               element one of the available pretrained vectors (see Vocab.load_vectors)
           Remaining keyword arguments: Passed to the splits method.
        """
        TEXT = data.Field()
        LABEL = data.Field(sequential=False)

        train, test = cls.splits(TEXT, LABEL, root=root, **kwargs)

        TEXT.build_vocab(train, vectors=vectors)
        LABEL.build_vocab(train)

        return data.BucketIterator.splits(
            (train, test), batch_size=batch_size, device=device)

