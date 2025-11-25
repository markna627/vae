import torch.nn
import torch
import torch.nn as nn



#Standard wiring 
class VAE(nn.Module):
  def __init__(self, hidden):
    super().__init__()
    self.hidden = hidden
    self.encoder = Encoder(self.hidden)
    self.decoder = Decoder(self.hidden)
    self.mu = 0
    self.logvar = 0
  def forward(self, x):
    self.mu, self.logvar = self.encoder(x)
    z = self.reparam(self.mu, self.logvar)
    decoder_out = self.decoder(z)
    return decoder_out

  def reparam(self, mu, logvar):
    std = torch.exp(0.5 * logvar)
    eps = torch.randn_like(std)
    return mu + eps * std

  def kl_loss(self):
    return (-0.5 * torch.sum((1 + self.logvar - self.mu**2 - torch.exp(self.logvar)), dim = 1)).mean()
  def generate(self, x):
    return self.decoder(x)

class Encoder(nn.Module):
  def __init__(self, hidden):
    super().__init__()
    self.conv = nn.Sequential(
        nn.Conv2d(3, 32, 3, 2, 1),
        nn.ReLU(),
        nn.Conv2d(32, 64, 3, 2, 1),
        nn.ReLU(),
        nn.Conv2d(64, 128, 3, 2, 1),
        nn.ReLU(),
        nn.Flatten()
    )
    self.mu = nn.Linear(128 * 4 * 4, hidden)
    self.var = nn.Linear(128 * 4 * 4, hidden)
  def forward(self, x):
    conv_out = self.conv(x)
    mu_out = self.mu(conv_out)
    var_out = self.var(conv_out)
    return mu_out, var_out


class Decoder(nn.Module):
  def __init__(self, hidden):
    super().__init__()
    self.fc = nn.Linear(hidden, 128 * 4 * 4)
    self.transpose_conv = nn.Sequential(
        nn.ConvTranspose2d(128, 64, 4, 2, 1),
        nn.ReLU(),
        nn.ConvTranspose2d(64, 32, 4, 2, 1),
        nn.ReLU(),
        nn.ConvTranspose2d(32, 3, 4, 2, 1),
        nn.Sigmoid()
    )
  def forward(self, x):
    fc_out = self.fc(x)
    fc_out = fc_out.reshape(-1, 128, 4, 4)
    trans_conv_out = self.transpose_conv(fc_out)
    return trans_conv_out













