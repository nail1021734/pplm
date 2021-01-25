import torch
import numpy as np
from operator import add
from model.gpt2 import GPT2
from model.gpt2discrim import Discrim
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
        device
    ):
        super().__init__()
        self.device = device
        self.attr_model = attr_model
        self.gpt_model = gpt_model
        self.kl_scale = kl_scale
        self.step_size = step_size
        self.gamma = gamma
        self.window_size = window_size
        self.gpt_model.model.eval()
        for param in self.gpt_model.model.parameters():
            param.requires_grad = False

    def unpert_generate(self):
        pass

    def KL_divergence(self):
        pass

    def post_norm(self):
        pass

    def pass_pert(self):
        pass
        # return pert_grad

    def generate_text(
        self,
        prefix,
        max_iter,
        cls_label,
    ):
        r"""
        1. 先餵到model一次來取得unpert的latent repersentation(之後要加delta H上去)
        2. 將delta H加到unpert_latent_representation，再餵給model取得$\tilde distribution$
        3. 將第一步的unpert_past當作past(past不能是第2步取出來的因為第2步的被pert過了)，並將第2步取出的每個
        字的pert_prob當權重乘上各自embed當作input_embed參數餵給gpt_model，取得當前的latent_representation
        (這個representation是用來讓discrim分類的，不是用來生成字)
        4. 將取得的latent_representation分為`label`類，用crossEntropy取得loss
        5. 算kl loss
        6. 反向傳播取得gradient，將gradient norm
        """
        SMALL_CONST = 1e-10
        x = self.gpt_model.tokenizer.encode(prefix)
        x = torch.tensor([x]).to(self.device)

        # split last input and another past input
        past = x[:, :-1]
        last = x[:, -1:]

        # Get unpert_past and unpert_last_hidden
        unpert_output = self.gpt_model.model(x)
        unpert_past = unpert_output['past_key_values']
        unpert_last_hidden = unpert_output['hidden_states'][-1]

        # Get past_key_values
        past_key_values = self.gpt_model.model(past)['past_key_values']

        # Store sum of unpert hidden representation
        sum_unpert_repr = unpert_last_hidden[:, :-1, :].sum(dim=1)

        grad_accumulator = [
            (np.zeros(p.shape).astype('float32'))
            for p in past_key_values
        ]

        discrim_input = None
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

            # Add x_t+1 repr to discrim input
            discrim_input = sum_unpert_repr +\
                new_output['hidden_states'][-1].sum(dim=1).detach()

            loss = 0.0

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
            discrim_input = discrim_input / (unpert_last_hidden.shape[1] + 1)

            # Calc discrim loss
            pred = self.attr_model.attr_model_forward(discrim_input)
            label = torch.LongTensor(
                [cls_label]
            ).to(device)
            discrim_loss = ce_loss(pred, label)
            loss += discrim_loss

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
                (torch.norm(p_.grad) + SMALL_CONST) + SMALL_CONST
                for p_ in curr_perturbation
            ]

            # normalize gradients
            grad = [
                -self.step_size * (
                p_.grad / grad_norms[index] ** self.gamma).data.cpu().numpy()
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


        return text

    def generate_full_text(
        self,
        max_length,
        pplm_iteration,
        prefix,
        label
    ):
    # 若prefix為1則先生成一個last(last為perturb目標)
        pert_generated = []

        for _ in range(max_length):
            text = self.generate_text(
                prefix=prefix,
                max_iter=pplm_iteration,
                cls_label=label
            )
            pert_generated += [text]

        return pert_generated

    def run_pplm(self, output_so_far):
        pass
if __name__ == "__main__":
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    gpt = GPT2().to(device)
    attribute_model = Discrim(gpt.embed_size, 5).to(device)
    attribute_model.load_model('1', 2)
    text = pplm(attribute_model, gpt, window_size=3, kl_scale=3, gamma=0.95, step_size=0.01, device=device)
    text.generate_text('Who was Jim Henson ?', 10, 1)