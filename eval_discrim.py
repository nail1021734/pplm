import torch

from tqdm import tqdm
from model.gpt2 import GPT2
from dataset.discrim_dataset.SST_dataset import SSTdataset
from model.gpt2discrim import Discrim
from model.pplm_classification_head import ClassificationHead


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    dataset = SSTdataset(typ='test')
    batch_size = 128
    data_loader = torch.utils.data.DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        collate_fn=dataset.collate_fn,
        shuffle=True
    )

    gpt_model = GPT2().to(device)
    dis_model = Discrim(embed_size=gpt_model.embed_size,
                        cls_num=dataset.clsnum).to(device)
    dis_model.load_model(
        '1', filename='discrim_model-epoch_22_loss0.5377536080792612.pt')
    all_data = len(dataset)
    correct = 0
    for x, y in tqdm(data_loader):
        x, y = x.to(device), y.to(device)
        mask = x.ne(0).unsqueeze(2).repeat(
            1, 1, gpt_model.embed_size
        ).float().to(device).detach()

        gpt_repr = gpt_model.get_repr(x)
        pred_y = dis_model.train_forward(gpt_repr, mask)
        pred_y = pred_y.argmax(dim=-1)

        correct += (pred_y == y).sum(dim=-1)
    print('accuracy:', correct / all_data)
