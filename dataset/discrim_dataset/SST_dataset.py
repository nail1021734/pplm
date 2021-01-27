import torch
from torchtext import data as torchtext_data
from torchtext import datasets
from tqdm import tqdm, trange
from nltk.tokenize.treebank import TreebankWordDetokenizer
from transformers import GPT2Tokenizer, GPT2LMHeadModel


class SSTdataset(torch.utils.data.Dataset):
    def __init__(self, typ='train'):
        super().__init__()
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        self.tokenizer = GPT2Tokenizer.from_pretrained("gpt2-medium")
        self.encoder = GPT2LMHeadModel.from_pretrained("gpt2-medium")
        self.embed_size = self.encoder.transformer.config.hidden_size
        self.idx2class = [
            "positive",
            "negative",
            "very positive",
            "very negative",
            "neutral"
        ]
        self.class2idx = {c: i for i, c in enumerate(self.idx2class)}
        self.x = []
        self.y = []
        self.getData(typ=typ)
        self.clsnum = len(self.idx2class)

    def __len__(self):
        return len(self.x)

    def getData(self, typ='train'):
        text = torchtext_data.Field()
        label = torchtext_data.Field(sequential=False)
        train_data, val_data, test_data = datasets.SST.splits(
            text,
            label,
            fine_grained=True,
            train_subtrees=True
        )

        if typ == 'train':
            for i in trange(len(train_data), ascii=True):
                seq = TreebankWordDetokenizer().detokenize(
                    vars(train_data[i])['text']
                )
                self.x.append(seq)
                self.y.append(vars(train_data[i])['label'])
        elif typ == 'test':
            for i in trange(len(test_data), ascii=True):
                seq = TreebankWordDetokenizer().detokenize(
                    vars(train_data[i])['text']
                )
                self.x.append(seq)
                self.y.append(vars(test_data[i])['label'])

    def __getitem__(self, index):
        return self.x[index], self.y[index]

    def cls2idx(self, _class):
        return self.class2idx[_class]

    def idx2cls(self, idx):
        return self.idx2class[idx]

    def batch_cls2idx(self, cls_list):
        return [self.cls2idx(i) for i in cls_list]

    def batch_idx2cls(self, idx_list):
        return [self.idx2cls(i) for i in idx_list]

    def collate_fn(self, batch):
        x = list(map(lambda arg: arg[0], batch))
        x = list(map(lambda arg: [50256] + self.tokenizer.encode(arg), x))
        x = list(map(lambda arg: torch.tensor(arg), x))

        y = list(map(lambda arg: arg[1], batch))
        y = self.batch_cls2idx(y)
        y = torch.tensor(y)

        lengths = [len(i) for i in x]
        padded_seq = torch.zeros(
            len(x),
            max(lengths)
        )
        for i, seq in enumerate(x):
            end = lengths[i]
            padded_seq[i, :end] = seq[:end]
        x = padded_seq

        return x, y


if __name__ == "__main__":
    a = SSTdataset(typ='train')
    d = torch.utils.data.DataLoader(
        a,
        batch_size=8,
        collate_fn=a.collate_fn
    )
    for x, y in d:
        print(x)
        print(y)
        break
