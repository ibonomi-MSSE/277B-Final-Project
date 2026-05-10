

def load_cryptic_tensors(device, vocabs=None, cache_path=None):
    """Load data into tensors. If `vocabs` is provided, use them as-is so
    indices match a previous run; otherwise build fresh from the data.
    If `cache_path` is provided and exists, load tensors from there."""
    # Fast path: load cached tensors if available
    if cache_path is not None and Path(cache_path).exists():
        print(f"Loading cached tensors from {cache_path}")
        cached = torch.load(cache_path, map_location='cpu')
        tensors = {k: v.to(device) for k, v in cached['tensors'].items()}
        return (tensors,
                cached['gene_pos_to_idx'],
                cached['drug_to_idx'])

    con = duckdb.connect()
    con.execute("SET memory_limit='30GB'")
    con.execute("SET preserve_insertion_order=false")

    print("Scanning data...")
    # First, create a CTE of mutations with sufficient counts
    arrow_table = con.execute(f"""
        WITH mutation_counts AS (
            SELECT
                m.GENE,
                m.AMINO_ACID_NUMBER,
                m.REF,
                m.ALT,
                COUNT(*) as cnt
            FROM read_parquet('./data/MUTATIONS.parquet') m
            JOIN read_parquet('./data/UKMYC_PHENOTYPES.parquet') p
                ON m.UNIQUEID = p.UNIQUEID
            WHERE
                p.LOG2MIC IS NOT NULL
                AND m.CODES_PROTEIN = TRUE
                AND p.PHENOTYPE_QUALITY IS NOT NULL
                AND m.IS_MINOR = FALSE
                AND m.REF IS NOT NULL
                AND m.ALT IS NOT NULL
            GROUP BY m.GENE, m.AMINO_ACID_NUMBER, m.REF, m.ALT
            HAVING COUNT(*) >= 50000
        )
        SELECT
            m.GENE              AS gene,
            m.AMINO_ACID_NUMBER AS pos,
            m.REF               AS ref,
            m.ALT               AS alt,
            d.DRUG_NAME         AS drug,
            p.LOG2MIC::FLOAT    AS log2mic
        FROM read_parquet('./data/MUTATIONS.parquet') m
        JOIN read_parquet('./data/UKMYC_PHENOTYPES.parquet') p
            ON m.UNIQUEID = p.UNIQUEID
        LEFT JOIN read_csv('./data/DRUG_CODES.csv.gz') d
            ON p.DRUG = d.DRUG_3_LETTER_CODE
        INNER JOIN mutation_counts mc
            ON m.GENE = mc.GENE
            AND m.AMINO_ACID_NUMBER = mc.AMINO_ACID_NUMBER
            AND m.REF = mc.REF
            AND m.ALT = mc.ALT
        WHERE
            p.LOG2MIC IS NOT NULL
            AND m.CODES_PROTEIN = TRUE
            AND p.PHENOTYPE_QUALITY IS NOT NULL
            AND m.IS_MINOR = FALSE
            AND m.REF IS NOT NULL
            AND m.ALT IS NOT NULL
    """).to_arrow_table()
    print(f"  {arrow_table.num_rows:,} rows loaded (filtered to mutations with ≥50000 occurrences)")

    gene_col = arrow_table['gene']
    pos_col  = arrow_table['pos']
    ref_col  = arrow_table['ref']
    alt_col  = arrow_table['alt']
    drug_col = arrow_table['drug']
    log_col  = arrow_table['log2mic']

    # Build (gene, pos) pair list once — used for both vocab construction and mapping.
    gene_list = gene_col.to_pylist()
    pos_list  = pos_col.to_pylist()
    gene_pos_pairs = list(zip(gene_list, pos_list))

    if vocabs is None:
        import pyarrow.compute as pc
        drug_unique = pc.unique(drug_col).to_pylist()
        drug_to_idx = {d: i for i, d in enumerate(v for v in drug_unique if v is not None)}

        unique_pairs = sorted(
            {p for p in gene_pos_pairs if p[0] is not None and p[1] is not None}
        )
        gene_pos_to_idx = {pair: i for i, pair in enumerate(unique_pairs)}
    else:
        gene_pos_to_idx, drug_to_idx = vocabs
        # Sanity check: warn if the new data has (gene, pos) pairs the saved vocab doesn't cover
        new_pairs = {p for p in gene_pos_pairs if p[0] is not None and p[1] is not None}
        missing = new_pairs - set(gene_pos_to_idx.keys())
        if missing:
            print(f"WARNING: {len(missing)} (gene, pos) pairs in data not in saved vocab "
                  f"(will be mapped to 0): {list(missing)[:5]}...")

    print(f"  {len(gene_pos_to_idx)} gene-position sites, {len(drug_to_idx)} drugs")

    def map_column(col, mapping, default):
        py_list = col.to_pylist()
        return np.fromiter(
            (mapping.get(v, default) for v in py_list),
            dtype=np.int64,
            count=len(py_list),
        )

    def map_pairs(pairs, mapping, default):
        return np.fromiter(
            (mapping.get(p, default) for p in pairs),
            dtype=np.int64,
            count=len(pairs),
        )

    cpu_tensors = {
        'gene_pos': torch.from_numpy(map_pairs(gene_pos_pairs, gene_pos_to_idx, 0)),
        'wt_aa':    torch.from_numpy(map_column(ref_col,  AA_TO_IDX, 20)),
        'mut_aa':   torch.from_numpy(map_column(alt_col,  AA_TO_IDX, 20)),
        'drug':     torch.from_numpy(map_column(drug_col, drug_to_idx, 0)),
        'log2mic':  torch.from_numpy(log_col.to_numpy().astype(np.float32)),
    }

    del arrow_table
    con.close()

    # Cache for next run
    if cache_path is not None:
        print(f"Caching tensors to {cache_path}")
        torch.save({
            'tensors': cpu_tensors,
            'gene_pos_to_idx': gene_pos_to_idx,
            'drug_to_idx': drug_to_idx,
        }, cache_path)

    tensors = {k: v.to(device) for k, v in cpu_tensors.items()}
    return tensors, gene_pos_to_idx, drug_to_idx





AA_TO_IDX = {
    'A': 0,  'R': 1,  'N': 2,  'D': 3,  'C': 4,
    'Q': 5,  'E': 6,  'G': 7,  'H': 8,  'I': 9,
    'L': 10, 'K': 11, 'M': 12, 'F': 13, 'P': 14,
    'S': 15, 'T': 16, 'W': 17, 'Y': 18, 'V': 19,
    'X': 20,
}


class CrypticDataset(Dataset):
    def __init__(self, tensors, indices):
        self.gene_pos = tensors['gene_pos'][indices]
        self.wt_aa    = tensors['wt_aa'][indices]
        self.mut_aa   = tensors['mut_aa'][indices]
        self.drug_idx = tensors['drug'][indices]
        self.log2mic  = tensors['log2mic'][indices]

    def __len__(self):
        return len(self.log2mic)

    def normalize_target(self, mean, std):
        self.log2mic = (self.log2mic - mean) / std

    def iter_batches(self, batch_size, shuffle):
        n = len(self)
        if shuffle:
            perm = torch.randperm(n, device=self.gene_pos.device)
        else:
            perm = torch.arange(n, device=self.gene_pos.device)
        for start in range(0, n, batch_size):
            idx = perm[start:start + batch_size]
            yield {
                'gene_pos': self.gene_pos[idx],
                'wt_aa':    self.wt_aa[idx],
                'mut_aa':   self.mut_aa[idx],
                'drug':     self.drug_idx[idx],
                'target':   self.log2mic[idx],
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
    plt.savefig('losses3.png')
    plt.close(fig)


def plot_predictions_scatter(model, dataset, target_mean, target_std, n_samples=50000, filename='predictions_scatter.png'):
    """Create a scatter plot of predictions vs true values."""
    model.eval()

    # Sample randomly from the dataset
    n_total = len(dataset)
    sample_size = min(n_samples, n_total)
    sample_indices = torch.randperm(n_total)[:sample_size]

    # Get predictions for the sample
    predictions = []
    true_values = []

    with torch.no_grad():
        # Process in batches to avoid memory issues
        batch_size = 8192
        for i in range(0, sample_size, batch_size):
            batch_idx = sample_indices[i:i + batch_size]

            batch_gene_pos = dataset.gene_pos[batch_idx]
            batch_wt_aa = dataset.wt_aa[batch_idx]
            batch_mut_aa = dataset.mut_aa[batch_idx]
            batch_drug = dataset.drug_idx[batch_idx]
            batch_target = dataset.log2mic[batch_idx]

            preds = model(batch_gene_pos, batch_wt_aa, batch_mut_aa, batch_drug).squeeze()

            predictions.append(preds.cpu())
            true_values.append(batch_target.cpu())

    predictions = torch.cat(predictions).numpy()
    true_values = torch.cat(true_values).numpy()

    # Denormalize to get back to original scale
    predictions = predictions * target_std.cpu().numpy() + target_mean.cpu().numpy()
    true_values = true_values * target_std.cpu().numpy() + target_mean.cpu().numpy()

    # Create scatter plot
    fig, ax = plt.subplots(figsize=(10, 10))
    ax.scatter(predictions, true_values, alpha=0.3, s=1)

    # Add diagonal line for perfect predictions
    min_val = min(predictions.min(), true_values.min())
    max_val = max(predictions.max(), true_values.max())
    ax.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, label='Perfect prediction')

    ax.set_xlabel('Predicted log2(MIC)', fontsize=12)
    ax.set_ylabel('True log2(MIC)', fontsize=12)
    ax.set_title(f'Predictions vs True Values (n={sample_size:,})', fontsize=14)
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Calculate and display R² on the plot
    correlation = np.corrcoef(predictions, true_values)[0, 1]
    r2 = correlation ** 2
    ax.text(0.05, 0.95, f'R² = {r2:.4f}\nCorrelation = {correlation:.4f}',
            transform=ax.transAxes, fontsize=12, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    plt.tight_layout()
    plt.savefig(filename, dpi=150)
    plt.close(fig)
    print(f"Scatter plot saved to {filename}")
    print(f"  R² = {r2:.4f}")
    print(f"  Correlation = {correlation:.4f}")


def parse_aa_change(aa_change):
    wt = aa_change[0]
    pos = int(aa_change[1:-1])
    mut = aa_change[-1]
    return AA_TO_IDX[wt], pos, AA_TO_IDX[mut]


class MICPredictor(nn.Module):
    def __init__(self, num_gene_pos, num_drugs, dropout=0.3):
        super().__init__()
        self.gene_pos_emb = nn.Embedding(num_gene_pos, 256)  # was 96
        self.wt_aa_emb = nn.Embedding(21, 32)  # was 16
        self.mut_aa_emb = nn.Embedding(21, 32)  # was 16
        self.drug_emb = nn.Embedding(num_drugs, 32)  # was 16

        # Total: 256 + 32 + 32 + 32 + 32 (mutation interaction) = 384
        self.mlp = nn.Sequential(
            nn.Linear(384, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.2),  # reduced from 0.3
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 1),
        )

    def forward(self, gene_pos, wt_aa, mut_aa, drug):
        gene_emb = self.gene_pos_emb(gene_pos)
        wt_emb = self.wt_aa_emb(wt_aa)
        mut_emb = self.mut_aa_emb(mut_aa)
        drug_emb = self.drug_emb(drug)

        # Add interaction: mutation effect vector (same dims, so subtraction works)
        mutation = mut_emb - wt_emb  # mutation effect vector (32 dims)

        x = torch.cat([
            gene_emb,    # 256
            wt_emb,      # 32
            mut_emb,     # 32
            drug_emb,    # 32
            mutation     # 32 (mutation effect)
        ], dim=-1)  # Total: 384
        return self.mlp(x)


def train():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    torch.set_float32_matmul_precision('high')

    # New checkpoint dir since vocab + architecture changed.
    checkpoint_dir = Path('./checkpoints_4')
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
            checkpoint['gene_pos_to_idx'],
            checkpoint['drug_to_idx'],
        )
        start_epoch   = checkpoint['epoch'] + 1
        best_val_loss = checkpoint['val_loss']
        print(f"Resuming from epoch {start_epoch}, best val loss: {best_val_loss:.4f}")

    # --- Data: reuse vocabs from checkpoint if present ---
    tensors, gene_pos_to_idx, drug_to_idx = load_cryptic_tensors(
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

    # --- Baseline: predict the train mean for every val example ---
    baseline_mse = ((val_ds.log2mic - train_ds.log2mic.mean()) ** 2).mean().item()
    print(f"Baseline val MSE (predict train mean): {baseline_mse:.4f}")

    # --- Hyperparameters ---
    batch_size = 65536
    num_epochs = 100
    patience = 5
    dropout = 0.3
    weight_decay = 1e-4

    # --- Model ---
    model = MICPredictor(
        num_gene_pos=len(gene_pos_to_idx),
        num_drugs=len(drug_to_idx),
        dropout=dropout,
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=weight_decay)
    loss_fn = nn.MSELoss()

    if checkpoint is not None:
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

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
            preds = model(batch['gene_pos'],
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
                preds = model(batch['gene_pos'],
                              batch['wt_aa'], batch['mut_aa'],
                              batch['drug']).squeeze()
                val_loss_sum += loss_fn(preds, batch['target']).item()

        avg_train_loss = train_loss_sum / n_train_batches
        avg_val_loss   = val_loss_sum / n_val_batches
        r2 = 1 - avg_val_loss / baseline_mse  # since val target variance ≈ baseline_mse

        print(f"Epoch {epoch+1:3d} | "
              f"Train Loss: {avg_train_loss:.4f} | "
              f"Val Loss: {avg_val_loss:.4f} | "
              f"Val R²: {r2:.4f}")
        train_losses[epoch] = avg_train_loss
        val_losses[epoch] = avg_val_loss

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            epochs_since_improvement = 0
            state_dict = getattr(model, '_orig_mod', model).state_dict()
            torch.save({
                'epoch': epoch,
                'model_state_dict': state_dict,
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': avg_train_loss,
                'val_loss': avg_val_loss,
                'gene_pos_to_idx': gene_pos_to_idx,
                'drug_to_idx': drug_to_idx,
            }, checkpoint_path)
            print(f"  → Saved new best model (val loss: {best_val_loss:.4f})")
        else:
            epochs_since_improvement += 1
            if epochs_since_improvement >= patience:
                print(f"Early stopping: no improvement for {patience} epochs.")
                break

    plot_losses(train_losses, val_losses)

    # Generate scatter plot using the best model
    print("\nGenerating scatter plot of predictions vs true values...")
    best_checkpoint = torch.load(checkpoint_path, map_location=device)
    best_model = MICPredictor(
        num_gene_pos=len(gene_pos_to_idx),
        num_drugs=len(drug_to_idx),
        dropout=dropout,
    ).to(device)
    best_model.load_state_dict(best_checkpoint['model_state_dict'])
    plot_predictions_scatter(best_model, val_ds, target_mean, target_std,
                            n_samples=50000, filename='predictions_scatter.png')

train()