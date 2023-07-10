#!/usr/bin/env python
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchvision.utils import save_image
import os
import numpy as np

batch_size = 128
epochs = 300
latent_dim = 100
channels = 3
image_size = 64
learning_rate = 0.0002
beta1 = 0.5

transform = transforms.Compose([
    transforms.Resize(image_size),
    transforms.CenterCrop(image_size),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
])

anime_faces = datasets.ImageFolder("/kaggle/input/animefacedataset/", transform=transform)
dataloader = DataLoader(anime_faces, batch_size=batch_size, shuffle=True, num_workers=4)

class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        self.main = nn.Sequential(
            # Input is a latent_dim-dimensional noise
            nn.ConvTranspose2d(latent_dim, 512, 4, 1, 0, bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(True),
            # Output size: (512, 4, 4)
            nn.ConvTranspose2d(512, 256, 4, 2, 1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(True),
            # Output size: (256, 8, 8)
            nn.ConvTranspose2d(256, 128, 4, 2, 1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(True),
            # Output size: (128, 16, 16)
            nn.ConvTranspose2d(128, 64, 4, 2, 1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            # Output size: (64, 32, 32)
            nn.ConvTranspose2d(64, channels, 4, 2, 1, bias=False),
            nn.Tanh()
            # Output size: (channels, 64, 64)
        )

    def forward(self, input):
        return self.main(input)

class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.main = nn.Sequential(
            # Input size: (channels, 64, 64)
            nn.Conv2d(channels, 64, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            # Output size: (64, 32, 32)
            nn.Conv2d(64, 128, 4, 2, 1, bias=False),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
             # Output size: (128, 16, 16)
            nn.Conv2d(128, 256, 4, 2, 1, bias=False),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),
            # Output size: (256, 8, 8)
            nn.Conv2d(256, 512, 4, 2, 1, bias=False),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True),
            # Output size: (512, 4, 4)
            nn.Conv2d(512, 1, 4, 1, 0, bias=False),
            nn.Sigmoid()
            # Output size: (1, 1, 1)
        )

    def forward(self, input):
        return self.main(input).view(-1, 1).squeeze(1)


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

generator = Generator()
discriminator = Discriminator()

if torch.cuda.device_count() > 1:
  print("Using ", torch.cuda.device_count(), " GPUs")
  generator = nn.DataParallel(generator)
  discriminator = nn.DataParallel(discriminator)

generator = generator.to(device)
discriminator = discriminator.to(device)

optimizer_G = optim.Adam(generator.parameters(), lr=learning_rate, betas=(beta1, 0.999))
optimizer_D = optim.Adam(discriminator.parameters(), lr=learning_rate, betas=(beta1, 0.999))

criterion = nn.MSELoss()


output_dir = '/home/aakashsphy21.itbhu/animegans/output'
os.makedirs(output_dir, exist_ok=True)

fixed_noise = torch.randn(64, latent_dim, 1, 1, device=device)

real_label = 0.9
fake_label = 0.1

for epoch in range(epochs):
    for i, (real_images, _) in enumerate(dataloader):
        batch_size = real_images.size(0)
        real_images = real_images.to(device)

        # Train the discriminator
        optimizer_D.zero_grad()
        real_labels = torch.full((batch_size,), real_label, device=device, dtype=torch.float)
        real_output = discriminator(real_images)
        d_loss_real = criterion(real_output, real_labels)
        d_loss_real.backward()

        noise = torch.randn(batch_size, latent_dim, 1, 1, device=device)
        fake_images = generator(noise)
        fake_labels = torch.full((batch_size,), fake_label, device=device, dtype=torch.float)
        fake_output = discriminator(fake_images.detach())
        d_loss_fake = criterion(fake_output, fake_labels)
        d_loss_fake.backward()

        d_loss = d_loss_real + d_loss_fake
        optimizer_D.step()

        # Train the generator
        optimizer_G.zero_grad()
        output = discriminator(fake_images)
        g_loss = criterion(output, real_labels)
        g_loss.backward()
        optimizer_G.step()

    # Save generated images
    if (epoch + 1) % 10 == 0:
        with torch.no_grad():
            fake = generator(fixed_noise).detach().cpu()
        save_image(fake, f"{output_dir}/fake_{epoch + 1}.png", nrow=8, normalize=True)

    print(f"[Epoch {epoch+1}/{epochs}] D_loss: {d_loss.item():.4f} G_loss: {g_loss.item():.4f}")


model_dir = '/home/aakashsphy21.itbhu/animegans/model'
os.makedirs(model_dir, exist_ok=True)

torch.save(generator.state_dict(), f'{model_dir}/generator.pth')
torch.save(discriminator.state_dict(), f'{model_dir}/discriminator.pth')
