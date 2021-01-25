import torch

from tqdm import tqdm
from model.gpt2 import GPT2
from dataset.discrim_dataset.SST_dataset import SSTdataset
from model.gpt2discrim import Discrim

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    dataset = SSTdataset()
    batch_size=128
    data_loader = torch.utils.data.DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        collate_fn=dataset.collate_fn,
        shuffle=True
    )

    gpt_model = GPT2().to(device)
    dis_model = Discrim(embed_size=gpt_model.embed_size, cls_num=dataset.clsnum).to(device)

    criterion = torch.nn.functional.nll_loss
    optim = torch.optim.Adam(dis_model.parameters(), lr=0.001)
    optim.zero_grad()
    epoch = 100

    for i in range(epoch):
        epoch_iter = tqdm(
            data_loader,
            desc=f'epoch: {i}, loss: {0:.6f}'
        )
        for x, y in epoch_iter:
            x, y = x.to(device), y.to(device)
            mask = x.ne(0).unsqueeze(2).repeat(
                1, 1, gpt_model.embed_size
            ).float().to(device).detach()

            gpt_repr = gpt_model.get_repr(x)
            pred_y = dis_model(gpt_repr, mask)

            loss = criterion(pred_y.reshape(-1, dataset.clsnum), y)
            epoch_iter.set_description(
                f'epoch: {i}, loss: {loss.item():.6f}'
            )
            loss.backward()
            optim.step()
            optim.zero_grad()
        dis_model.save_model('1', log=i)
