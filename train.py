from tqdm import tqdm
import vae
import data
import torch
import torch.nn as nn

def train():
    epochs = 50
    hidden = 64
    model = vae.VAE(hidden)
    recon_loss = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters())

    loss_metrics = {
        'train_recon': [],
        'train_kl': [],
        'train_loss': [],

        'val_recon': [],
        'val_kl':[],
        'val_loss':[]
    }
    train_dataloader, val_dataloader, test_dataloader = data.dataloaders()
    for epoch in range(epochs):
      model.train()
      train_loss = 0
      for x, y in tqdm(train_dataloader, desc = f"Training - Epoch: {epoch} "):
        beta = 1e-4
        optimizer.zero_grad()
        generated = model(x)

        recon = recon_loss(generated, x)
        kl = model.kl_loss()
        loss = recon + beta * kl
        train_loss += recon.item()

        loss.backward()
        loss_metrics['train_recon'].append(recon.item())
        loss_metrics['train_kl'].append(kl.item())
        loss_metrics['train_loss'].append(loss.item())
        optimizer.step()

      with torch.no_grad():
        model.eval()
        val_loss = 0
        for x, y in tqdm(val_dataloader, desc = "Validation"):
          beta = 1e-4
          generated = model(x)

          recon = recon_loss(generated, x)
          kl = model.kl_loss()
          loss = recon + beta * kl

          val_loss += recon.item()
          loss_metrics['val_recon'].append(recon.item())
          loss_metrics['val_kl'].append(kl.item())
          loss_metrics['val_loss'].append(loss.item())
      print(f'Train_loss: {train_loss/len(train_dataloader):.4f} Val_loss: {val_loss/len(val_dataloader):.4f}')

if __name__ == "__main__":
    train()

