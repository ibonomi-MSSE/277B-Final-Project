# Model Improvement Suggestions

## Current Performance
- Training Loss: 0.4825
- Validation Loss: 0.475
- The validation loss being lower than training loss suggests the model might be underfitting.

## Suggested Improvements (in order of priority)

### 1. **Increase Model Capacity** ⭐ (Most Important)
Your model is quite small and likely underfitting. Try:
- **Larger embeddings**: Increase gene_pos embedding from 96 → 256 or 512
- **Deeper/wider MLP**: Try `[144 → 256 → 128 → 64 → 1]` or even deeper
- **Reduce dropout**: Lower from 0.3 → 0.1 or 0.2 (less regularization since you're underfitting)

```python
# Example:
self.gene_pos_emb = nn.Embedding(num_gene_pos, 256)  # was 96
self.wt_aa_emb = nn.Embedding(21, 32)  # was 16
self.mut_aa_emb = nn.Embedding(21, 32)  # was 16
self.drug_emb = nn.Embedding(num_drugs, 32)  # was 16

# Total: 256 + 32 + 32 + 32 = 352
self.mlp = nn.Sequential(
    nn.Linear(352, 512),
    nn.ReLU(),
    nn.Dropout(0.2),  # reduced from 0.3
    nn.Linear(512, 256),
    nn.ReLU(),
    nn.Dropout(0.2),
    nn.Linear(256, 128),
    nn.ReLU(),
    nn.Dropout(0.2),
    nn.Linear(128, 1),
)
```

### 2. **Learning Rate Schedule**
Add a learning rate scheduler to help find better minima:

```python
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode='min', factor=0.5, patience=2, verbose=True
)

# In training loop after validation:
scheduler.step(avg_val_loss)
```

### 3. **Interaction Features**
The model treats each feature independently. Add explicit interactions:

```python
def forward(self, gene_pos, wt_aa, mut_aa, drug):
    gene_emb = self.gene_pos_emb(gene_pos)
    wt_emb = self.wt_aa_emb(wt_aa)
    mut_emb = self.mut_aa_emb(mut_aa)
    drug_emb = self.drug_emb(drug)
    
    # Add pairwise interactions
    gene_drug = gene_emb * drug_emb  # element-wise product
    mutation = mut_emb - wt_emb  # mutation effect vector
    
    x = torch.cat([
        gene_emb, wt_emb, mut_emb, drug_emb,
        gene_drug, mutation  # added interactions
    ], dim=-1)
    return self.mlp(x)
```

### 4. **Better Amino Acid Representation**
Instead of simple embeddings, use biochemical properties (hydrophobicity, charge, etc.):

```python
# Add to model init:
self.aa_property_emb = nn.Linear(5, 16)  # 5 properties → 16 dims

# Use properties like: hydrophobicity, charge, molecular weight, etc.
```

### 5. **Batch Normalization**
Add batch norm layers to help training:

```python
self.mlp = nn.Sequential(
    nn.Linear(144, 256),
    nn.BatchNorm1d(256),  # added
    nn.ReLU(),
    nn.Dropout(0.2),
    nn.Linear(256, 128),
    nn.BatchNorm1d(128),  # added
    nn.ReLU(),
    nn.Dropout(0.2),
    nn.Linear(128, 1),
)
```

### 6. **Include More Mutations**
You're filtering to only mutations with ≥50,000 occurrences, which is very restrictive:
- Try lowering to 5,000 or 10,000 to include more diversity
- This gives the model more examples to learn from

```python
# In SQL query:
HAVING COUNT(*) >= 5000  # instead of 50000
```

### 7. **Ensemble Methods**
Train multiple models with different seeds and average predictions:

```python
# Train 5 models with different seeds
models = []
for seed in [42, 123, 456, 789, 1011]:
    torch.manual_seed(seed)
    model = train_single_model(seed)
    models.append(model)

# At inference:
predictions = torch.stack([m(x) for m in models]).mean(dim=0)
```

### 8. **Loss Function Alternatives**
Try different loss functions:
- **Huber Loss**: More robust to outliers
- **Weighted MSE**: Weight rare mutations more heavily

```python
loss_fn = nn.HuberLoss(delta=1.0)  # instead of MSELoss
```

### 9. **Data Augmentation**
For biochemistry, synonymous mutations or conservative substitutions could help:
- Add noise to embeddings during training
- Use amino acid similarity matrices

### 10. **Attention Mechanism**
Add attention between gene positions and drugs to learn which combinations matter:

```python
class AttentionMICPredictor(nn.Module):
    def __init__(self, ...):
        # ... embeddings ...
        self.attention = nn.MultiheadAttention(embed_dim=96, num_heads=4)
        # ... rest ...
```

## Quick Wins to Try First:
1. **Increase model size** (embeddings 96→256, MLP wider/deeper)
2. **Reduce dropout** (0.3 → 0.1 or 0.2)
3. **Add learning rate scheduler**
4. **Lower mutation count threshold** (50000 → 5000)

These should give you noticeable improvements quickly!

## Expected Improvements:
- Current R²: ~0.05-0.10 (estimated from your losses)
- Target R²: 0.3-0.5 with these improvements
- Best case: 0.6-0.7 with ensemble + all improvements
