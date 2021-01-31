import torch
import os
import numpy as np
from operator import add
from model.gpt2 import GPT2
from model.gpt2discrim import Discrim
from model.pplm_classification_head import ClassificationHead

import numpy as np


class pplm():
    def __init__(
        self,
        attr_model,
        gpt_model,
        window_size,
        gamma,
        kl_scale,
        step_size,
        gm_scale,
        sample,
        device
    ):
        super().__init__()
        self.device = device
        self.attr_model = attr_model
        self.gpt_model = gpt_model
        self.kl_scale = kl_scale
        self.step_size = step_size
        self.gamma = gamma
        self.gm_scale = gm_scale
        self.window_size = window_size
        self.gpt_model.model.eval()
        self.sample = sample
        for param in self.gpt_model.model.parameters():
            param.requires_grad = False

    def topk_filter(self, distribution, k):
        if k == 0:
            return distribution
        else:
            min_values = distribution.topk(k)[0][:, -1]
            return distribution.where(
                distribution >= min_values,
                torch.zeros_like(distribution)
            )

    def unpert_generate(self, prefix, max_length, k):
        prefix = [50256] + self.gpt_model.tokenizer.encode(prefix)

        for _ in range(max_length):
            x = torch.tensor([prefix]).to(self.device)
            output = self.gpt_model.model(x)
            logits = output['logits']
            probs = torch.nn.functional.softmax(logits[:, -1, :], dim=-1)
            topk_probs = self.topk_filter(probs, k=k)
            if topk_probs.sum(dim=-1) <= 1:
                topk_probs = topk_probs / topk_probs.sum(dim=-1)
            # greedy or sample
            if self.sample:
                text = topk_probs.multinomial(num_samples=1)
            else:
                _, text = topk_probs.topk(k=1, dim=-1)

            prefix += [text.item()]

        result = self.gpt_model.tokenizer.decode(prefix)

        return result

    def calc_discrim_loss(
        self,
        sum_unpert_repr,
        new_output,
        unpert_past,
        probs,
        cls_label,
        repr_num
    ):
        # Add x_t+1 repr to discrim input
        discrim_input = sum_unpert_repr +\
            new_output['hidden_states'][-1].sum(dim=1).detach()

        # Calc distrim model loss
        ce_loss = torch.nn.CrossEntropyLoss()

        # Get gpt word embedding
        wte = self.gpt_model.model.resize_token_embeddings()
        input_embed = torch.matmul(probs, wte.weight.data)

        # Get perturb gpt output(x_t+2)
        pert_output = self.gpt_model.model(
            past_key_values=unpert_past,
            inputs_embeds=input_embed
        )

        pert_last_hidden = pert_output['hidden_states'][-1]
        discrim_input = discrim_input + pert_last_hidden.sum(dim=1)
        discrim_input = discrim_input / (repr_num)

        # Calc discrim loss
        pred = self.attr_model(discrim_input)
        label = torch.LongTensor(
            [cls_label]
        ).to(device)
        discrim_loss = ce_loss(pred, label)
        return discrim_loss

    def get_BoW_words(self, cls_list):
        word_list = []
        dir_path = 'BoW'
        for topic in cls_list:
            with open(
                os.path.join(dir_path, topic+'.txt'), 'r', encoding='utf8'
            ) as input_files:
                words = input_files.readlines()
            for word in words:
                word_list.append(word.strip())

        return word_list

    def get_one_hot_vectors(self, word_list):
        bow_index = [self.gpt_model.tokenizer.encode(
            word,
            add_prefix_space=True,
            add_special_tokens=False
        ) for word in word_list]
        bow_index = torch.tensor(bow_index).to(self.device)
        one_hot_vector = torch.zeros(
            bow_index.shape[0],
            self.gpt_model.tokenizer.vocab_size
        ).to(self.device)
        one_hot_vector.scatter_(1, bow_index, 1)
        return one_hot_vector

    def calc_BoW_loss(self, probs, bow_one_hot_vec):
        bow_logits = torch.mm(
            probs.squeeze(dim=1),
            bow_one_hot_vec.T
        )
        bow_loss = -torch.log(bow_logits.sum(dim=-1))
        return bow_loss[0]

    def get_pert_past(
        self,
        unpert_output,
        last,
        past_key_values,
        max_iter,
        cls_label,
        classifier
    ):
        SMALL_CONST = 1e-10

        # create window mask
        past_seq_length = past_key_values[0].shape[3]
        if self.window_size is not None and past_seq_length > self.window_size and self.window_size > 0:
            ones_key_val_shape = (
                tuple(past_key_values[0].shape[:-2])
                + tuple([self.window_size])
                + tuple(past_key_values[0].shape[-1:])
            )
            zeros_key_val_shape = (
                tuple(past_key_values[0].shape[:-2])
                + tuple([past_seq_length - self.window_size])
                + tuple(past_key_values[0].shape[-1:])
            )

            ones_mask = torch.ones(ones_key_val_shape)
            zero_mask = torch.zeros(zeros_key_val_shape)

            window_mask = torch.cat(
                (ones_mask, zero_mask),
                dim=-2
            ).to(self.device)
        else:
            window_mask = torch.ones_like(past_key_values[0]).to(device)

        unpert_past = unpert_output['past_key_values']
        unpert_last_hidden = unpert_output['hidden_states'][-1]

        # Store sum of unpert hidden representation
        sum_unpert_repr = unpert_last_hidden[:, :-1, :].sum(dim=1)

        grad_accumulator = [
            (np.zeros(p.shape).astype('float32'))
            for p in past_key_values
        ]

        # Repeat perturb past_key_value `max_iter` times.
        for _ in range(max_iter):
            curr_perturbation = [
                torch.tensor(p_, requires_grad=True, device=self.device)
                for p_ in grad_accumulator
            ]

            # Get perturb past.
            perturb_past = list(map(add, past_key_values, curr_perturbation))

            # Calculate new perturb distribution(x_t+1).
            new_output = self.gpt_model.model(
                last,
                past_key_values=perturb_past
            )
            probs = torch.nn.functional.softmax(
                new_output['logits'][:, -1, :],
                dim=-1
            ).unsqueeze(dim=1)

            loss = 0.0
            if classifier == "discrim":
                discrim_loss = self.calc_discrim_loss(
                    sum_unpert_repr=sum_unpert_repr,
                    new_output=new_output,
                    unpert_past=unpert_past,
                    probs=probs,
                    cls_label=cls_label,
                    repr_num=unpert_last_hidden.shape[1] + 1
                )
                loss += discrim_loss
            elif classifier == "BoW":
                word_list = self.get_BoW_words(cls_label.split(','))
                one_hot_vector = self.get_one_hot_vectors(word_list)
                BoW_loss = self.calc_BoW_loss(probs, one_hot_vector)
                loss += BoW_loss

            # Calc KL_divergence
            if self.kl_scale > 0.0:
                unpert_probs = torch.nn.functional.softmax(
                    unpert_output['logits'][:, -1, :],
                    dim=-1
                )
                unpert_probs = (
                    unpert_probs + SMALL_CONST *
                    (unpert_probs <= SMALL_CONST).float().to(self.device).detach()
                )
                temp = SMALL_CONST * (probs <= SMALL_CONST).float().to(
                    self.device).detach()
                pert_probs = probs + temp.detach()
                kl_loss = self.kl_scale * (
                    (pert_probs * (pert_probs / unpert_probs).log()).sum())
                loss += kl_loss

            # Compute gradient
            loss.backward()

            # calc grad norm factor
            grad_norms = [
                (torch.norm(p_.grad * window_mask) + SMALL_CONST) + SMALL_CONST
                for p_ in curr_perturbation
            ]

            # normalize gradients
            grad = [
                -self.step_size * (
                    p_.grad * window_mask / grad_norms[
                        index] ** self.gamma).data.cpu().numpy()
                for index, p_ in enumerate(curr_perturbation)
            ]

            # Add to delta H
            grad_accumulator = list(map(add, grad, grad_accumulator))

            # reset gradient
            for p_ in curr_perturbation:
                p_.grad.data.zero_()

            # removing past from the graph
            new_past = []
            for p_ in past_key_values:
                new_past.append(p_.detach())
            past_key_values = new_past

        # apply the accumulated perturbations to the past
        grad_accumulator = [
            torch.tensor(p_, requires_grad=True, device=self.device)
            for p_ in grad_accumulator
        ]
        pert_past = list(map(add, past_key_values, grad_accumulator))

        return pert_past

    def generate_text(
        self,
        prefix,
        max_iter,
        cls_label,
        k,
        classifier
    ):
        x = torch.tensor([prefix]).to(self.device)

        # split last input and another past input
        past = x[:, :-1]
        last = x[:, -1:]

        # Get unpert_past and unpert_last_hidden
        unpert_output = self.gpt_model.model(x)

        # Get past_key_values
        past_key_values = self.gpt_model.model(past)['past_key_values']

        pert_past = self.get_pert_past(
            unpert_output=unpert_output,
            last=last,
            past_key_values=past_key_values,
            max_iter=max_iter,
            cls_label=cls_label,
            classifier=classifier
        )

        pert_output = self.gpt_model.model(last, past_key_values=pert_past)
        pert_logits = pert_output['logits']
        pert_probs = torch.nn.functional.softmax(
            pert_logits[:, -1, :],
            dim=-1
        )

        unpert_probs = torch.nn.functional.softmax(
            unpert_output['logits'][:, -1, :],
            dim=-1
        )

        pert_probs = ((pert_probs ** self.gm_scale) * (
            unpert_probs ** (1 - self.gm_scale)))
        topk_probs = self.topk_filter(pert_probs, k=k)

        if topk_probs.sum(dim=-1) <= 1:
            topk_probs = topk_probs / topk_probs.sum(dim=-1)

        # greedy or sample
        if self.sample:
            text = topk_probs.multinomial(num_samples=1)
        else:
            _, text = topk_probs.topk(k=1, dim=-1)

        return text.item()

    def generate_full_text(
        self,
        max_length,
        pplm_iteration,
        prefix,
        k,
        label,
        classifier
    ):
        prefix = [50256] + self.gpt_model.tokenizer.encode(prefix)

        for _ in range(max_length):
            text = self.generate_text(
                prefix=prefix,
                max_iter=pplm_iteration,
                cls_label=label,
                k=k,
                classifier=classifier
            )
            prefix += [text]

        result = self.gpt_model.tokenizer.decode(prefix)

        return result


if __name__ == "__main__":
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    gpt = GPT2().to(device)
    # attribute_model = ClassificationHead(5, gpt.embed_size).to(device)
    # attribute_model.load_state_dict(
    #     torch.load('checkpoint/discrim/1/discrim_model-paper.pt', map_location=device))
    attribute_model = Discrim(embed_size=gpt.embed_size, cls_num=5).to(device)
    attribute_model.load_model(
        '1', 'discrim_model-epoch_29_loss0.533.pt')

    text = pplm(attribute_model, gpt, window_size=10, kl_scale=0.01,
                gamma=1.0, step_size=0.05, gm_scale=0.95, sample=True, device=device)
    # for i in range(4):
    #     print(text.generate_full_text(max_length=20, pplm_iteration=10, prefix="My dog died", k=40, label=2))
    # for i in range(4):
    #     print(text.generate_full_text(max_length=20, pplm_iteration=10, prefix="My dog died", k=40, label=3))
    print('unpert:', text.unpert_generate(
        prefix='The dog', max_length=30, k=40))
    print('pos:', text.generate_full_text(max_length=30,
                                          pplm_iteration=10, prefix="The dog", k=40, label="space", classifier="BoW"))
    print('neg:', text.generate_full_text(max_length=30,
                                          pplm_iteration=40, prefix="The dog", k=40, label=3, classifier="discrim"))
