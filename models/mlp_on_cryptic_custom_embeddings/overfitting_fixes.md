# Overfitting Mitigation Strategies

## Changes Applied

### 1. **Dropout (0.3)** ✅
Added dropout layers after each ReLU activation in the MLP. This randomly drops 30% of neurons during training, preventing co-adaptation and improving generalization.

### 2. **L2 Regularization (weight_decay=1e-4)** ✅
Added weight decay to the Adam optimizer to penalize large weights and encourage simpler models.

### 3. **Early Stopping (patience=5)** ✅
Reduced patience from 10 to 5 epochs since your validation loss plateaus quickly around epoch 5.

### 4. **Data Filtering (≥500 occurrences)** ✅
Filtered to only include mutations that appear at least 500 times in the dataset. This:
- Reduces from ~221M to ~203M rows (removes ~18M noisy examples)
- Keeps 16,038 high-quality mutations instead of all variants
- Reduces noise from rare variants that may not generalize well

## Additional Strategies to Try

### 5. **Batch Normalization**
Add `nn.BatchNorm1d()` layers after each Linear layer (before ReLU). This can help with training stability:

```python
self.mlp = nn.Sequential(
    nn.Linear(144, 128),
    nn.BatchNorm1d(128),
    nn.ReLU(),
    nn.Dropout(dropout),
    nn.Linear(128, 64),
    nn.BatchNorm1d(64),
    nn.ReLU(),
    nn.Dropout(dropout),
    nn.Linear(64, 1),
)
```

**Note:** BatchNorm + Dropout can sometimes conflict. Try one or the other first.

### 6. **Reduce Model Capacity**
If overfitting persists, try smaller hidden layers:
```python
# Option A: Fewer neurons
nn.Linear(144, 64),
nn.ReLU(),
nn.Dropout(dropout),
nn.Linear(64, 32),
nn.ReLU(),
nn.Dropout(dropout),
nn.Linear(32, 1),

# Option B: Fewer layers
nn.Linear(144, 64),
nn.ReLU(),
nn.Dropout(dropout),
nn.Linear(64, 1),
```

### 7. **Reduce Embedding Dimensions**
Your current embeddings might be too large:
```python
self.gene_pos_emb = nn.Embedding(num_gene_pos, 64)  # was 96
self.aa_emb = nn.Embedding(21, 12)  # was 16
self.drug_emb = nn.Embedding(num_drugs, 12)  # was 16
```

### 8. **Learning Rate Scheduling**
Reduce learning rate when validation loss plateaus:
```python
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode='min', factor=0.5, patience=3, verbose=True
)
# In training loop after validation:
scheduler.step(avg_val_loss)
```

### 9. **Increase Dropout Rate**
If 0.3 doesn't help enough, try 0.4 or 0.5.

### 10. **Data Augmentation**
Since you have gene-position-drug combinations, you could:
- Add noise to embeddings during training
- Use MixUp or similar techniques

### 11. **Ensemble Methods**
Train multiple models with different random seeds and average their predictions.

### 12. **Filter by Gene as Well**
Currently you filter mutations by count. You could also filter to only genes with ≥500 rows:
```sql
AND m.GENE IN (
    SELECT GENE
    FROM read_parquet('./data/MUTATIONS.parquet') m
    JOIN read_parquet('./data/UKMYC_PHENOTYPES.parquet') p ON m.UNIQUEID = p.UNIQUEID
    WHERE p.LOG2MIC IS NOT NULL
    GROUP BY GENE
    HAVING COUNT(*) >= 500
)
```

## What to Monitor

After retraining with these changes, look for:
1. **Smaller gap** between train and val loss
2. **Validation loss** continues to improve (not plateauing at 0.46)
3. **R² score** on validation set improves
4. **Training loss** doesn't drop as quickly (sign that regularization is working)

## Expected Results

With dropout + weight decay + filtered data:
- Training loss should decrease more slowly
- Validation loss should improve beyond 0.46
- The gap between train/val should narrow
- Model should generalize better to unseen data

## Testing Strategy

1. Train with current changes (dropout + weight_decay + data filtering)
2. If still overfitting → increase dropout to 0.4 or 0.5
3. If still overfitting → reduce model capacity (smaller hidden layers)
4. If still overfitting → add BatchNorm (but remove dropout first)
5. If still overfitting → reduce embedding dimensions

Start simple and add complexity only if needed!
