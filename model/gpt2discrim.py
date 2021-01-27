import os
import torch


class Discrim(torch.nn.Module):
    def __init__(self, embed_size, cls_num):
        super().__init__()
        self.embed_size = embed_size
        self.cls_num = cls_num
        self.discrim = torch.nn.Linear(
            in_features=embed_size,
            out_features=cls_num
        )
        self.idx2class = [
            "positive",
            "negative",
            "very positive",
            "very negative",
            "neutral"
        ]
        self.class2idx = {c: i for i, c in enumerate(self.idx2class)}

    def avg_repr(self, x, mask):
        masked_hidden = x * mask
        avg_repr = torch.sum(masked_hidden, dim=1) / (
            torch.sum(mask, dim=1).detach() + 1e-10
        )
        return avg_repr

    def train_forward(self, x, mask):
        x = self.avg_repr(x, mask)
        x = self.discrim(x)
        x = torch.nn.functional.log_softmax(x, dim=-1)
        return x

    def forward(self, avg_repr):
        logits = self.discrim(avg_repr)
        return logits

    def save_model(self, exp_name, log):
        path = os.path.join('checkpoint', 'discrim', exp_name)
        if not os.path.exists(path):
            os.mkdir(path)
        torch.save(
            self.state_dict(),
            f=os.path.join(path, f'discrim_model-{log}.pt')
        )

    def load_model(self, exp_name, filename):
        path = os.path.join('checkpoint', 'discrim', exp_name)
        self.load_state_dict(torch.load(
            f=os.path.join(path, filename)
        ))
