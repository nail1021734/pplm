import torch
from tqdm import trange
from transformers import GPT2Tokenizer, GPT2LMHeadModel


class GPT2(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.tokenizer = GPT2Tokenizer.from_pretrained('gpt2-medium')
        self.model = GPT2LMHeadModel.from_pretrained(
            'gpt2-medium',
            output_hidden_states=True
        )
        self.embed_size = self.model.transformer.config.hidden_size
        self.model.eval()
        for param in self.model.parameters():
            param.requires_grad = False

    def forward(self, x):
        with torch.no_grad():
            x = self.model(x)
        return x

    def get_repr(self, x):
        with torch.no_grad():
            hidden = self.model.transformer(x.long())
        return hidden['last_hidden_state']

    def generate_text(self, x, max_length):
        r"""
        args:
            x: a tokenized string.ex:'Who was Jim Henson ?'
            max_length: max_length of generate sentence.
        """
        x = [50256] + self.tokenizer.encode(x)
        for _ in trange(max_length, ascii=True):
            context = torch.tensor([x])
            output = self(context)
            token = output['logits'][..., -1, :].argmax(dim=-1)
            x += [token.tolist()[-1]]

        return self.tokenizer.decode(x)


if __name__ == "__main__":
    model = GPT2()
    sentence = 'Who was Jim Henson ?'
    a = model.tokenizer.encode(sentence)
    print(vars(model(torch.tensor(a)))['hidden_states'][-1])
    print(model.get_repr(torch.tensor(a)))
