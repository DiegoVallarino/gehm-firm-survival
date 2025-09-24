# ============================================
# Synthetic Dataset Generator with GAN (PyTorch)
# ============================================
import os
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

# ----------------------------
# 1. Load original dataset
# ----------------------------
data_path = r"real_data.csv"
df = pd.read_csv(data_path)

print("✅ Original dataset loaded:", df.shape)

# Drop ID and categorical variables if already encoded
df_num = df.drop(columns=["Firm_ID"])  # keep Firm_ID aside
df_num = df_num.apply(pd.to_numeric, errors="coerce").fillna(0)

data = torch.tensor(df_num.values, dtype=torch.float32)

# ----------------------------
# 2. Define GAN components
# ----------------------------
latent_dim = 32   # size of random noise input
data_dim = data.shape[1]

class Generator(nn.Module):
    def __init__(self, latent_dim, data_dim):
        super(Generator, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(latent_dim, 128),
            nn.ReLU(True),
            nn.Linear(128, 128),
            nn.ReLU(True),
            nn.Linear(128, data_dim)
        )
    def forward(self, z):
        return self.model(z)

class Discriminator(nn.Module):
    def __init__(self, data_dim):
        super(Discriminator, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(data_dim, 128),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(128, 128),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(128, 1),
            nn.Sigmoid()
        )
    def forward(self, x):
        return self.model(x)

generator = Generator(latent_dim, data_dim)
discriminator = Discriminator(data_dim)

criterion = nn.BCELoss()
optimizer_G = optim.Adam(generator.parameters(), lr=0.001)
optimizer_D = optim.Adam(discriminator.parameters(), lr=0.001)

# ----------------------------
# 3. Training loop
# ----------------------------
epochs = 3000
batch_size = 128

for epoch in range(epochs):
    # Sample real data minibatch
    idx = np.random.randint(0, data.shape[0], batch_size)
    real = data[idx]

    # Generate fake data
    z = torch.randn(batch_size, latent_dim)
    fake = generator(z)

    # Labels
    real_labels = torch.ones(batch_size, 1)
    fake_labels = torch.zeros(batch_size, 1)

    # --- Train Discriminator ---
    optimizer_D.zero_grad()
    out_real = discriminator(real)
    loss_real = criterion(out_real, real_labels)
    out_fake = discriminator(fake.detach())
    loss_fake = criterion(out_fake, fake_labels)
    loss_D = (loss_real + loss_fake) / 2
    loss_D.backward()
    optimizer_D.step()

    # --- Train Generator ---
    optimizer_G.zero_grad()
    out_fake = discriminator(fake)
    loss_G = criterion(out_fake, real_labels)  # fool discriminator
    loss_G.backward()
    optimizer_G.step()

    if epoch % 500 == 0:
        print(f"Epoch {epoch}: Loss_D={loss_D.item():.4f}, Loss_G={loss_G.item():.4f}")

print("✅ Training completed.")

# ----------------------------
# 4. Generate synthetic dataset
# ----------------------------
n_samples = len(df)
z = torch.randn(n_samples, latent_dim)
synthetic_data = generator(z).detach().numpy()

df_synt = pd.DataFrame(synthetic_data, columns=df_num.columns)

# Keep Firm_ID sequential
df_synt.insert(0, "Firm_ID", range(1, n_samples+1))

# ----------------------------
# 5. Save synthetic dataset
# ----------------------------
output_path = r"firms_features_clean.csv"
df_synt.to_csv(output_path, index=False)

print(f"✅ Synthetic dataset saved at: {output_path}")
print(df_synt.head())
