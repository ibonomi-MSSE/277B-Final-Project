
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
from tqdm import tqdm


AA_TO_IDX = {
    'A': 0,  'R': 1,  'N': 2,  'D': 3,  'C': 4,
    'Q': 5,  'E': 6,  'G': 7,  'H': 8,  'I': 9,
    'L': 10, 'K': 11, 'M': 12, 'F': 13, 'P': 14,
    'S': 15, 'T': 16, 'W': 17, 'Y': 18, 'V': 19,
    'X': 20,
}


class CrypticDataset(Dataset):
    def __init__(self, tensors, indices):
        self.gene_idx = tensors['gene'][indices]
        self.pos     = tensors['pos'][indices]
        self.wt_aa   = tensors['wt_aa'][indices]
        self.mut_aa  = tensors['mut_aa'][indices]
        self.drug_idx = tensors['drug'][indices]
        self.log2mic = tensors['log2mic'][indices]

    def __len__(self):
        return len(self.log2mic)

    def normalize_target(self, mean, std):
        self.log2mic = (self.log2mic - mean) / std

    def iter_batches(self, batch_size, shuffle):
        n = len(self)
        if shuffle:
            perm = torch.randperm(n, device=self.gene_idx.device)
        else:
            perm = torch.arange(n, device=self.gene_idx.device)
        for start in range(0, n, batch_size):
            idx = perm[start:start + batch_size]
            yield {
                'gene':   self.gene_idx[idx],
                'pos':    self.pos[idx],
                'wt_aa':  self.wt_aa[idx],
                'mut_aa': self.mut_aa[idx],
                'drug':   self.drug_idx[idx],
                'target': self.log2mic[idx],
            }

    def num_batches(self, batch_size):
        return (len(self) + batch_size - 1) // batch_size


def build_vocabs(df):
    gene_to_idx = {g: i for i, g in enumerate(df['GENE'].unique())}
    drug_to_idx = {d: i for i, d in enumerate(df['DRUG_NAME'].unique())}
    position_to_idx = {pos: i for i, pos in enumerate(df['AMINO_ACID_NUMBER'].unique())}
    return gene_to_idx, drug_to_idx, position_to_idx


def plot_losses(train_losses, val_losses):
    # Filter Nones in case of early stopping
    train_losses = [l for l in train_losses if l is not None]
    val_losses = [l for l in val_losses if l is not None]

    fig, ax = plt.subplots()
    ax.set(title='Losses for Training and Validation',
           xlabel='Epoch', ylabel='Loss')
    ax.plot(range(len(train_losses)), train_losses, label='train')
    ax.plot(range(len(val_losses)), val_losses, label='val')
    ax.legend()
    plt.savefig('losses.png')
    plt.close(fig)


def parse_aa_change(aa_change):
    wt = aa_change[0]
    pos = int(aa_change[1:-1])
    mut = aa_change[-1]
    return AA_TO_IDX[wt], pos, AA_TO_IDX[mut]


class MICPredictor(nn.Module):
    def __init__(self, num_genes, num_positions, num_drugs):
        super().__init__()
        self.gene_emb = nn.Embedding(num_genes, 64)
        self.pos_emb  = nn.Embedding(num_positions, 32)
        self.aa_emb   = nn.Embedding(21, 16)
        self.drug_emb = nn.Embedding(num_drugs, 16)

        self.mlp = nn.Sequential(
            nn.Linear(144, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
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


def train():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    torch.set_float32_matmul_precision('high')

    checkpoint_dir = Path('./checkpoints_1')
    checkpoint_dir.mkdir(exist_ok=True)
    checkpoint_path = checkpoint_dir / 'best_model.pt'
    tensor_cache_path = checkpoint_dir / 'tensors.pt'

    # --- Resume logic ---
    start_epoch = 0
    best_val_loss = float('inf')
    checkpoint = None
    saved_vocabs = None

    if checkpoint_path.exists():
        print(f"Loading checkpoint from {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=device)
        saved_vocabs = (
            checkpoint['gene_to_idx'],
            checkpoint['drug_to_idx'],
            checkpoint['position_to_idx'],
        )
        start_epoch   = checkpoint['epoch'] + 1
        best_val_loss = checkpoint['val_loss']
        print(f"Resuming from epoch {start_epoch}, best val loss: {best_val_loss:.4f}")

    # --- Data: reuse vocabs from checkpoint if present ---
    tensors, gene_to_idx, drug_to_idx, position_to_idx = load_cryptic_tensors(
        device, vocabs=saved_vocabs, cache_path=tensor_cache_path
    )

    n = len(tensors['log2mic'])
    g = torch.Generator().manual_seed(42)
    perm = torch.randperm(n, generator=g)
    n_val = int(0.2 * n)
    val_indices   = perm[:n_val].to(device)
    train_indices = perm[n_val:].to(device)

    train_ds = CrypticDataset(tensors, train_indices)
    val_ds   = CrypticDataset(tensors, val_indices)
    del tensors

    target_mean = train_ds.log2mic.mean()
    target_std  = train_ds.log2mic.std()
    train_ds.normalize_target(target_mean, target_std)
    val_ds.normalize_target(target_mean, target_std)

    # --- Hyperparameters ---
    batch_size = 16384
    num_epochs = 100
    patience = 10  # early stopping: stop if no val improvement for this many epochs

    # --- Model ---
    model = MICPredictor(
        num_genes=len(gene_to_idx),
        num_positions=len(position_to_idx),
        num_drugs=len(drug_to_idx),
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = nn.MSELoss()

    if checkpoint is not None:
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

    # Optional: torch.compile for a free speedup on PyTorch 2.x.
    # Comment out if you hit compile issues.
    try:
        model = torch.compile(model)
    except Exception as e:
        print(f"torch.compile unavailable, continuing without it: {e}")

    # --- Training loop ---
    train_losses = [None] * num_epochs
    val_losses   = [None] * num_epochs
    epochs_since_improvement = 0

    print('starting training')
    for epoch in tqdm(range(start_epoch, num_epochs)):
        # Train
        model.train()
        train_loss_sum = 0.0
        n_train_batches = train_ds.num_batches(batch_size)
        for batch in train_ds.iter_batches(batch_size, shuffle=True):
            optimizer.zero_grad()
            preds = model(batch['gene'], batch['pos'],
                          batch['wt_aa'], batch['mut_aa'],
                          batch['drug']).squeeze()
            loss = loss_fn(preds, batch['target'])
            loss.backward()
            optimizer.step()
            train_loss_sum += loss.item()

        # Validate
        model.eval()
        val_loss_sum = 0.0
        n_val_batches = val_ds.num_batches(batch_size)
        with torch.no_grad():
            for batch in val_ds.iter_batches(batch_size, shuffle=False):
                preds = model(batch['gene'], batch['pos'],
                              batch['wt_aa'], batch['mut_aa'],
                              batch['drug']).squeeze()
                val_loss_sum += loss_fn(preds, batch['target']).item()

        avg_train_loss = train_loss_sum / n_train_batches
        avg_val_loss   = val_loss_sum / n_val_batches

        print(f"Epoch {epoch+1:3d} | "
              f"Train Loss: {avg_train_loss:.4f} | "
              f"Val Loss: {avg_val_loss:.4f}")
        train_losses[epoch] = avg_train_loss
        val_losses[epoch] = avg_val_loss

        # Save best + early stopping bookkeeping
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            epochs_since_improvement = 0
            # If we wrapped with torch.compile, the underlying module is
            # at model._orig_mod — save that to keep checkpoints portable.
            state_dict = getattr(model, '_orig_mod', model).state_dict()
            torch.save({
                'epoch': epoch,
                'model_state_dict': state_dict,
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': avg_train_loss,
                'val_loss': avg_val_loss,
                'gene_to_idx': gene_to_idx,
                'drug_to_idx': drug_to_idx,
                'position_to_idx': position_to_idx,
            }, checkpoint_path)
            print(f"  → Saved new best model (val loss: {best_val_loss:.4f})")
        else:
            epochs_since_improvement += 1
            if epochs_since_improvement >= patience:
                print(f"Early stopping: no improvement for {patience} epochs.")
                break

    plot_losses(train_losses, val_losses)

train()