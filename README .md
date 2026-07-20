# COMET: Ablation Study Experimental Suite

Comprehensive ablation study repository containing **101 experimental configurations** across **7 experimental groups (A-G)** for the COMET (Contrastive Molecular Ensemble Transformer) model.

## Repository Structure

```
.
├── A_code/              # Group A: LambdaRank loss function
│   ├── train.py
│   ├── run_experiments.py
│   └── utils/
├── B_code/              # Group B: Hard Negative Mining
│   ├── train_hard.py
│   ├── run_experiments_B.py
│   └── utils/
├── C_code/              # Group C: Stochastic Weight Averaging
│   ├── train_swa.py
│   ├── run_experiments_C.py
│   └── utils/
├── D_code/              # Group D: Structural Data Augmentation
│   ├── train_aug.py
│   ├── run_experiments_D.py
│   └── utils/
├── E_code/              # Group E: Descriptor Enhancement
│   ├── train_ds.py
│   ├── run_experiments_2B.py
│   └── utils/
├── F_code/              # Group F: Fusion Strategy / Soft Exit
│   ├── train_com.py
│   ├── run_experiments_F1.py
│   └── utils/
├── G_code/              # Group G: Additional ablations
│   ├── train.py
│   ├── run_experiments_G.py
│   └── utils/
├── data_result/         # Raw experimental results (all 101 configurations)
├── scripts/             # Data preprocessing and analysis scripts
│   ├── preprocess_data.py
│   ├── generate_embeddings.py
│   ├── compute_metrics.py
│   ├── summarize_results.py
│   └── plot_figures.py
├── chemberta_embeds/    # Pre-computed ChemBERTa-2 embeddings
│   └── mol_embeddings.pt
├── ckp/                 # Pretrained model weights
│   └── mol_pre_no_h_220816.pt
├── task_schemas/        # Task schema JSON files
├── requirements.txt
└── README.md
```

---

## 1. Environment Setup

### 1.1 Create Conda Environment

```bash
conda create -n comet_env python=3.10
conda activate comet_env
```

### 1.2 Install Dependencies

```bash
# Install PyTorch (adjust CUDA version as needed)
conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia

# Install other dependencies
pip install -r requirements.txt
```

**requirements.txt:**

```
torch>=1.12.0
numpy>=1.21.0
pandas>=1.3.0
scipy>=1.7.0
scikit-learn>=1.0.0
transformers>=4.20.0
tokenizers>=0.12.0
rdkit>=2022.03.1
tqdm
pyyaml
pyprojroot
```

### 1.3 Install UniMol Framework

The project depends on a modified version of UniMol:

```bash
# For A/B/C/D experiments
git clone https://github.com/dptech-corp/Uni-Mol.git
# Copy the modified unimol modules from each group code folder
cp -r A_code/A_unimol/* Uni-Mol/unimol/

# For E/F experiments (use COM/DS variants)
cp -r E_code/DS_unimol/* Uni-Mol/unimol/
cp -r F_code/COM_unimol/* Uni-Mol/unimol/
```

### 1.4 Directory Setup

```bash
# Create necessary directories
mkdir -p save_demo tmp_save_demo save_aug tmp_save_aug save_demo/logs
mkdir -p infer_results eval_results explanations chemberta_embeds
mkdir -p data_result/{group_A,group_B,group_C,group_D,group_E,group_F,group_G}
mkdir -p logs tmp
```

---

## 2. Data Preprocessing

### 2.1 Prepare Raw Data

Place the raw LNP dataset files in the project root:

```
./
├── processed_data_dirs/
│   └── OS_demo_in_house_lnp_data_overall_new_full_without_pbae_NPratios_updated_09222023_npratios_09252023gen_fig3di/
│       ├── fold_V0/
│       ├── fold_V1/
│       └── fold_V2/
```

### 2.2 Generate ChemBERTa-2 Embeddings (for E/F groups)

```bash
# Activate environment
conda activate comet_env

# Generate SMILES embeddings using ChemBERTa-2
python scripts/generate_embeddings.py \
    --input ./processed_data_dirs/OS_demo_in_house_lnp_data_overall_new_full_without_pbae_NPratios_updated_09222023_npratios_09252023gen_fig3di/ \
    --output ./chemberta_embeds/mol_embeddings.pt \
    --model chemberta-v2 \
    --batch_size 64
```

**Output:** `./chemberta_embeds/mol_embeddings.pt`

### 2.3 Validate Data Structure

```bash
# Check that all folds are properly formatted
python scripts/preprocess_data.py \
    --data_path ./processed_data_dirs/ \
    --task_schema task_schemas/in_house_lnp_master_schema_NPratio_AOvolratio.json \
    --check_only
```

---

## 3. Running Experiments

### General Rules

- Each group code folder is **self-contained** and can be run independently.
- All experiments use **3-fold cross-validation** (fold_V0, V1, V2).
- Default: **150 epochs**, patience=1000 (no early stopping), batch_size=128.
- Results are automatically saved to `data_result/group_X/`.

### Environment Variables (Optional)
In article

### 3.1 Group A: LambdaRank Loss Function (12 configs)

```bash
cd A_code

# Run all A experiments (A1-0 ~ A3-3, 3 folds each)
python run_experiments.py

# Run a single configuration manually
python run.sh

# Results saved to:
# data_result/group_A/A1-0_fold_V0.log
# data_result/group_A/A1-0_fold_V1.log
# data_result/group_A/A1-0_fold_V2.log
```

**Subgroups:**

| Subgroup | Description | Config IDs |
|:---|:---|:---|
| A1 | Contrastive margin K | A1-0 (K=10), A1-1 (K=20), A1-2 (K=50) |
| A2 | Label margin λ | A2-0 (λ=0.01), A2-1 (λ=0.005), A2-2 (λ=0.02) |
| A3 | Noise proportion | A3-0 (0.1), A3-1 (0.2), A3-2 (0.3) |

---

### 3.2 Group B: Hard Negative Mining (18 configs)

```bash
cd B_code

# Run all B experiments
python run_experiments_B.py

# Results saved to data_result/group_B/
```

**Subgroups:**

| Subgroup | Description | Config IDs |
|:---|:---|:---|
| B1 | Sampling ratio | B1-0 ~ B1-5 (0% to 10→50%) |
| B2 | Mining frequency | B2-1 ~ B2-4 (every 1/5/10/20 epochs) |
| B3 | Weight scaling | B3-0 ~ B3-4 (none/×0.5/×1.0/×2.0/adaptive) |
| B4 | Sampling strategy | B4-1 ~ B4-4 (uniform/curriculum/hard-priority/adaptive) |

---

### 3.3 Group C: Stochastic Weight Averaging (15 configs)

```bash
cd C_code

# Run all C experiments
python run_experiments_C.py

# Results saved to data_result/group_C/
```

---

### 3.4 Group D: Structural Data Augmentation (18 configs)

```bash
cd D_code

# Run all D experiments
python run_experiments_D.py

# Results saved to data_result/group_D/
```

---

### 3.5 Group E: Descriptor Enhancement (12 configs)

```bash
cd E_code

# Run all E experiments
python run_e1.py

# Results saved to data_result/group_E/
```

---

### 3.6 Group F: Fusion Strategy / Soft Exit (15 configs)

```bash
cd F_code

# Run all F experiments
python run_e1.py --dataset fig3di_b16f10
python run_e1.py --dataset fig3di_dc24
