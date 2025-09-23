#  Repository: Graph-Based Survival under AI Disruption

##  Overview
This repository contains the full computational pipeline, code, and synthetic datasets used in the paper:

**"When Firms Fail to Adopt: Network-Mediated Survival under AI and General-Purpose Technologies"**

The project combines Graph Neural Networks (GNNs), structural embeddings, and survival analysis to study how firms' network positions influence their survival under technological disruption.

---

##  Repository Structure

```text
GraphSurvival-AI/
├── README.md                 <- Repository overview and instructions
├── LICENSE                   <- License terms (e.g., CC-BY-NC or MIT)
├── data/
│   ├── synthetic_firms.csv   <- Synthetic firm-level dataset
│   └── synthetic_network_edges.csv <- Synthetic edge list for firm graph
├── scripts/
│   ├── 01_data_preprocessing.R      <- Cleaning, imputation, harmonization
│   ├── 02_graph_construction.py     <- Network creation from edges
│   ├── 03_embedding_training.py     <- GCN / GraphSAGE training (PyTorch)
│   ├── 04_survival_model.R          <- Cox and GEHM models (R)
│   ├── 05_dynamic_analysis.R        <- Time-varying effects, SHAP
│   └── utils/
│       ├── plot_helpers.R
│       └── parameters.yml
├── output/
│   ├── figures/              <- Survival plots, embedding visualizations
│   └── models/               <- Trained GNN and survival model outputs
└── docs/
    ├── Extra_Material.pdf    <- Computational appendix (as uploaded)
    └── figures/              <- Figures used in the paper (uploaded PNGs)
```

---

##  Requirements

### R Packages
- `survival`
- `survivalmodels`
- `ggplot2`
- `mice`
- `tidyverse`

### Python Packages
- `torch`, `torch_geometric`
- `networkx`
- `shap`
- `pandas`, `numpy`, `matplotlib`

Create virtual environments or use `renv`/`conda` to isolate dependencies.

---

##  Reproducing Results

1. Clone the repo:
```bash
git clone https://github.com/yourusername/GraphSurvival-AI.git
cd GraphSurvival-AI
```

2. Load synthetic data (in `/data/`).

3. Run preprocessing:
```bash
Rscript scripts/01_data_preprocessing.R
```

4. Build the graph:
```bash
python scripts/02_graph_construction.py
```

5. Train embeddings:
```bash
python scripts/03_embedding_training.py
```

6. Fit survival models:
```bash
Rscript scripts/04_survival_model.R
```

7. Analyze results:
```bash
Rscript scripts/05_dynamic_analysis.R
```

---

##  Outputs
- Survival curves by sector and region
- Time-varying hazard effects
- GNN-based embeddings
- SHAP-based interpretability visualizations

---

##  License & Citation
Use of this repository is granted for **academic purposes only**. Please cite the working paper if using the code or data:

```bibtex
@article{Vallarino2025,
  title={When Firms Fail to Adopt: Network-Mediated Survival under AI and General-Purpose Technologies},
  author={Vallarino, Diego},
  journal={Working Paper},
  year={2025}
}
```

---

## ✉️ Contact
For access to the private GitHub repository, real data, or further replication files, contact: **diegoval@iadb.org**
