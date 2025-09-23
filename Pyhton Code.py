# ================================
# Graph Embeddings with PyTorch Geometric
# ================================
import pandas as pd
import torch
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv
import torch.nn.functional as F

# ----------------------------
# 1. Cargar datos de firmas (sin embeddings previos)
# ----------------------------
firms = pd.read_csv("firms_features.csv")  # archivo con Firm_ID, Sector, Region, Size, Leverage, Profit_Margin, Org_Complexity, RD_Intensity

# Codificar variables categóricas (Sector, Region)
firms_enc = pd.get_dummies(firms, columns=["Sector", "Region"])

# Features tensor
x = torch.tensor(firms_enc.drop(columns=["Firm_ID"]).values, dtype=torch.float)

# ----------------------------
# 2. Construir grafo (ejemplo simple: conectar por sector/region)
# ----------------------------
edges = []
sector_groups = firms.groupby("Sector").groups
for _, idx in sector_groups.items():
    idx = list(idx)
    for i in range(len(idx)-1):
        edges.append([idx[i], idx[i+1]])

region_groups = firms.groupby("Region").groups
for _, idx in region_groups.items():
    idx = list(idx)
    for i in range(len(idx)-1):
        edges.append([idx[i], idx[i+1]])

edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()

# ----------------------------
# 3. Definir modelo GCN
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
# 4. Entrenar embeddings
# ----------------------------
model = GCN(in_channels=x.shape[1], hidden_channels=32, out_channels=4)  # 4 dimensiones para comparar con los PCA
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

for epoch in range(201):
    model.train()
    optimizer.zero_grad()
    out = model(x, edge_index)
    # self-supervised: minimizar reconstrucción (ejemplo simplificado L2)
    loss = ((out - x[:, :4])**2).mean()
    loss.backward()
    optimizer.step()
    if epoch % 50 == 0:
        print(f"Epoch {epoch}, Loss {loss.item()}")

# ----------------------------
# 5. Guardar embeddings
# ----------------------------
embeddings = model(x, edge_index).detach().numpy()
emb_df = pd.DataFrame(embeddings, columns=["Embed_1", "Embed_2", "Embed_3", "Embed_4"])
result = pd.concat([firms["Firm_ID"], emb_df], axis=1)

result.to_csv("firm_embeddings_gnn.csv", index=False)
print("Embeddings guardados en firm_embeddings_gnn.csv")
