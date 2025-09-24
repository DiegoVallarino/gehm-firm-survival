# ============================================
# GNN-Enhanced Survival Dataset Upload
# ============================================

import os
import pandas as pd
import torch
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv

# ----------------------------
# 1. Load firm-level features
# ----------------------------
data_path = "data/firms_features_clean.csv"   # relative path (repo structure)
firms = pd.read_csv(data_path)

# Encode categorical variables (Sector, Region)
firms_enc = pd.get_dummies(firms, columns=["Sector", "Region"])

# Convert everything to numeric and force float32
firms_enc = firms_enc.apply(pd.to_numeric, errors="coerce").fillna(0)

# Save Firm_ID separately
firm_ids = firms_enc["Firm_ID"].values

# Create feature tensor (excluding Firm_ID)
x = torch.tensor(
    firms_enc.drop(columns=["Firm_ID"]).astype("float32").values,
    dtype=torch.float32
)

print("✅ Feature tensor created:", x.shape)

# ----------------------------
# 2. Build the firm graph
# ----------------------------
edges = []

# Connect firms within the same sector
sector_groups = firms.groupby("Sector").groups
for _, idx in sector_groups.items():
    idx = list(idx)
    for i in range(len(idx) - 1):
        edges.append([idx[i], idx[i + 1]])

# Connect firms within the same region
region_groups = firms.groupby("Region").groups
for _, idx in region_groups.items():
    idx = list(idx)
    for i in range(len(idx) - 1):
        edges.append([idx[i], idx[i + 1]])

edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()

# PyTorch Geometric data object
data = Data(x=x, edge_index=edge_index)

# ----------------------------
# 3. Define GCN model
# ----------------------------
class GCN(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super(GCN, self).__init__()
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, out_channels)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = self.conv2(x, edge_index)
        return x

# ----------------------------
# 4. Train embeddings
# ----------------------------
model = GCN(in_channels=x.shape[1], hidden_channels=32, out_channels=4)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

for epoch in range(201):
    model.train()
    optimizer.zero_grad()
    out = model(x, edge_index)
    # Simplified autoencoder: align first 4 features with embeddings
    loss = ((out - x[:, :4])**2).mean()
    loss.backward()
    optimizer.step()
    if epoch % 50 == 0:
        print(f"Epoch {epoch}, Loss {loss.item():.4f}")

# ----------------------------
# 5. Save embeddings
# ----------------------------
embeddings = model(x, edge_index).detach().numpy()
emb_df = pd.DataFrame(embeddings, columns=["Embed_1", "Embed_2", "Embed_3", "Embed_4"])

# ----------------------------
# 6. Final dataset
# ----------------------------
# Merge embeddings with the original dataset (which already contains Event_Time and Status)
firms_full = pd.concat([firms, emb_df], axis=1)

# Save CSV file
output_path = "data/firms_survival_full_clean.csv"
firms_full.to_csv(output_path, index=False)

print(f"✅ File generated: {output_path}")
print(firms_full.head())



