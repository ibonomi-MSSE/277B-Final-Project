
# standard import

# card_data = get_card_dataframe().encode_with(columns=['chembera_embedding'])

# trainx, testx, trainy, testy = get_our_data()

# ConfusionMatrixDisplay()
# plt.save('results.png')





"""
To run, run from project root:
python -m models.mlp_on_cryptic_custom_embeddings.train_model

"""



import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

from data.cryptic_consortium_data.query import get_cryptic_data



AA_TO_IDX = {
    'A': 0,   # Alanine
    'R': 1,   # Arginine
    'N': 2,   # Asparagine
    'D': 3,   # Aspartate
    'C': 4,   # Cysteine
    'Q': 5,   # Glutamine
    'E': 6,   # Glutamate
    'G': 7,   # Glycine
    'H': 8,   # Histidine
    'I': 9,   # Isoleucine
    'L': 10,  # Leucine
    'K': 11,  # Lysine
    'M': 12,  # Methionine
    'F': 13,  # Phenylalanine
    'P': 14,  # Proline
    'S': 15,  # Serine
    'T': 16,  # Threonine
    'W': 17,  # Tryptophan
    'Y': 18,  # Tyrosine
    'V': 19,  # Valine
    'X': 20,  # Unknown
}


class CrypticDataset(Dataset):
    def __init__(self, df, gene_to_idx, drug_to_idx, position_to_idx):
        self.gene_idx = torch.tensor(
            [gene_to_idx.get(g, 0) for g in df['GENE']], dtype=torch.long)
        self.pos = torch.tensor(
            [position_to_idx.get(p, 0) for p in df['AMINO_ACID_NUMBER']], dtype=torch.long)
        self.wt_aa = torch.tensor(
            [AA_TO_IDX.get(a, 20) for a in df['REF']], dtype=torch.long)
        self.mut_aa = torch.tensor(
            [AA_TO_IDX.get(a, 20) for a in df['ALT']], dtype=torch.long)
        self.drug_idx = torch.tensor(
            [drug_to_idx.get(d, 0) for d in df['DRUG_NAME']], dtype=torch.long)
        self.log2mic = torch.tensor(
            df['LOG2MIC'].values, dtype=torch.float32)

    def __len__(self):
        return len(self.log2mic)

    def __getitem__(self, idx):
        return {
            'gene':    self.gene_idx[idx],
            'pos':     self.pos[idx],
            'wt_aa':   self.wt_aa[idx],
            'mut_aa':  self.mut_aa[idx],
            'drug':    self.drug_idx[idx],
            'target':  self.log2mic[idx],
        }


def build_vocabs(df):
    gene_to_idx = {g: i for i, g in enumerate(df['GENE'].unique())}
    drug_to_idx = {d: i for i, d in enumerate(df['DRUG_NAME'].unique())}
    position_to_idx = {pos: i for i, pos in enumerate(df['AMINO_ACID_NUMBER'].unique())}
    return gene_to_idx, drug_to_idx, position_to_idx


def plot_losses(train_losses, val_losses):
    fig, ax = plt.subplots()
    ax.set(
        title='Losses for Training and Validation',
        xlabel='Epoch',
        ylabel='Loss',
    )
    ax.plot(range(len(train_losses)), train_losses)
    ax.plot(range(len(val_losses)), val_losses)
    plt.savefig('losses.png')

    # TODO: plot the expected vs predicted MIC values

def parse_aa_change(aa_change):
    # "S315T" → wt='S', pos=315, mut='T'
    wt  = aa_change[0]          # 'S'
    pos = int(aa_change[1:-1])  # 315
    mut = aa_change[-1]         # 'T'
    return AA_TO_IDX[wt], pos, AA_TO_IDX[mut]

class MICPredictor(nn.Module):
    def __init__(self, num_genes, num_positions, num_drugs):
        super().__init__()
        self.gene_emb = nn.Embedding(num_genes, 64)
        self.pos_emb  = nn.Embedding(num_positions, 32)
        self.aa_emb   = nn.Embedding(21, 16) # 21 amino acids, 1 is unknown
        self.drug_emb = nn.Embedding(num_drugs, 16)
        
        self.mlp = nn.Sequential(
            nn.Linear(144, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )

    def forward(self, gene, pos, wt_aa, mut_aa, drug):
        x = torch.cat([
            self.gene_emb(gene),
            self.pos_emb(pos),
            self.aa_emb(wt_aa),
            self.aa_emb(mut_aa),
            self.drug_emb(drug),
        ], dim=-1)
        return self.mlp(x)


if __name__ == '__main__':
    # --- Check for existing checkpoint ---
    checkpoint_dir = Path(__file__).parent / 'checkpoints'
    checkpoint_dir.mkdir(exist_ok=True)
    checkpoint_path = checkpoint_dir / 'best_model.pt'

    # --- Load and split data ---
    df = get_cryptic_data()
    train_df, val_df = train_test_split(df, test_size=0.2, random_state=42)

    start_epoch = 0
    best_val_loss = float('inf')

    # If checkpoint exists, use its vocabularies; otherwise build new ones
    if checkpoint_path.exists():
        print(f"Loading checkpoint from {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path)
        gene_to_idx = checkpoint['gene_to_idx']
        drug_to_idx = checkpoint['drug_to_idx']
        position_to_idx = checkpoint['position_to_idx']
        start_epoch = checkpoint['epoch'] + 1
        best_val_loss = checkpoint['val_loss']
        print(f"Resuming from epoch {start_epoch}, best val loss: {best_val_loss:.4f}")
    else:
        gene_to_idx, drug_to_idx, position_to_idx = build_vocabs(train_df)

    train_ds = CrypticDataset(train_df, gene_to_idx, drug_to_idx, position_to_idx)
    val_ds   = CrypticDataset(val_df,   gene_to_idx, drug_to_idx, position_to_idx)

    train_loader = DataLoader(train_ds, batch_size=512, shuffle=True)
    val_loader   = DataLoader(val_ds,   batch_size=512, shuffle=False)

    # --- Model, loss, optimizer ---
    num_genes = len(gene_to_idx)
    num_drugs = len(drug_to_idx)
    num_positions = len(position_to_idx)

    model = MICPredictor(num_genes=num_genes, num_positions=num_positions, num_drugs=num_drugs)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = nn.MSELoss()  # predicting continuous LOG2MIC

    # Load model and optimizer state if resuming
    if checkpoint_path.exists():
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

    # --- Training loop ---
    num_epochs = 1000
    train_losses = [None] * num_epochs
    val_losses = [None] * num_epochs

    for epoch in range(start_epoch, num_epochs):
        # Train
        model.train()
        train_loss = 0
        for batch in train_loader:
            optimizer.zero_grad()
            preds = model(
                batch['gene'], batch['pos'],
                batch['wt_aa'], batch['mut_aa'],
                batch['drug']
            ).squeeze()
            loss = loss_fn(preds, batch['target'])
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        # Validate
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for batch in val_loader:
                preds = model(
                    batch['gene'], batch['pos'],
                    batch['wt_aa'], batch['mut_aa'],
                    batch['drug']
                ).squeeze()
                val_loss += loss_fn(preds, batch['target']).item()

        avg_train_loss = train_loss/len(train_loader)
        avg_val_loss = val_loss/len(val_loader)

        print(f"Epoch {epoch+1:3d} | "
              f"Train Loss: {avg_train_loss:.4f} | "
              f"Val Loss: {avg_val_loss:.4f}")
        train_losses[epoch] = avg_train_loss
        val_losses[epoch] = avg_val_loss

        # Save checkpoint if validation loss improved
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': avg_train_loss,
                'val_loss': avg_val_loss,
                'gene_to_idx': gene_to_idx,
                'drug_to_idx': drug_to_idx,
                'position_to_idx': position_to_idx,
            }, checkpoint_path)
            print(f"  → Saved new best model (val loss: {best_val_loss:.4f})")

    plot_losses(train_losses, val_losses)
