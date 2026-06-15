# ⚾ MLB Pitch Type Classification
### Random Forest vs XGBoost · Statcast Movement Features

Multi-class classifier that identifies MLB pitch types from their **physical movement signatures** — velocity, spin rate, release position, and kinematic vectors — using ~700,000 Statcast pitch records.

The model deliberately excludes pitcher identity. The goal is a classifier that understands *what a pitch does in space*, not *who threw it*, making it generalizable to pitchers not seen during training.

---

## Results

| Model | Accuracy | Macro F1 | Macro AUC |
|-------|----------|----------|-----------|
| Random Forest | 0.89 | 0.86 | 0.9922 |
| **XGBoost (GPU)** | **0.93** | **0.91** | **0.9961** |

XGBoost outperforms RF on every metric. The largest gain is on **FC (Cutter)** — F1 improved from 0.76 to 0.84 — the hardest class due to its physical overlap with both four-seamers and sliders.

5-fold stratified cross-validation confirms results are stable across data splits (σ ≤ 0.004).

---

## Pitch Types Modeled

Pitch types with fewer than 15,000 Statcast samples are excluded. The 8 retained classes cover the vast majority of MLB pitch usage.

| Code | Pitch Type |
|------|------------|
| FF | Four-Seam Fastball |
| SI | Sinker |
| SL | Slider |
| CH | Changeup |
| FC | Cutter |
| ST | Sweeper |
| CU | Curveball |
| FS | Splitter |

---

## Features

13 Statcast physical features across four dimensions:

| Dimension | Features |
|-----------|----------|
| Velocity & Spin | `release_speed`, `release_spin_rate` |
| Release Point | `release_pos_x`, `release_pos_z`, `arm_angle` |
| Movement | `horizontal_break`, `vertical_break` |
| Kinematics | `vx0`, `vy0`, `vz0`, `ax`, `ay`, `az` |

**Why no pitcher ID?** Including `pitcher` would let the model memorize tendencies (pitcher X throws mostly FF/SL) rather than learning physics. It would fail on unseen pitchers — exactly the situation where you need it to work.

**Why `arm_angle`?** Arm slot physically affects horizontal break. It captures pitcher-level release physics without hardcoding identity — the right way to include that context.

---

## Feature Importance

Both models agree: **movement signature** is the primary identifier — not velocity or spin alone.

| Feature | RF MDI | XGB Gain |
|---------|--------|----------|
| `horizontal_break` | 0.2118 | 0.3224 |
| `vertical_break` | 0.1717 | 0.3825 |
| `release_spin_rate` | 0.1192 | 0.0853 |
| `az` | 0.1100 | 0.0194 |
| `release_speed` | 0.0724 | 0.0719 |

The models diverge after the top two. RF distributes importance across correlated features (`az`, `ax`, `vy0` all rank top-6) because random feature subsampling forces different trees to use different variables. XGBoost concentrates on the two features that do most of the work — `horizontal_break` + `vertical_break` account for ~70% of normalized gain, and kinematic vectors nearly disappear.

This is not a contradiction — it reflects how each algorithm assigns importance. `az` (vertical acceleration) and `vertical_break` measure correlated quantities. XGBoost picks one; RF splits credit between them.

---

## Hard Cases

- **FC** — physically intermediate between fastball and slider; misclassification is analytically expected
- **SL vs ST** — Statcast only introduced Sweeper as a distinct category in 2023; historical re-labeling is imperfect, making this partly a data quality issue
- **CH vs FS** — similar velocity drop from fastball; separated primarily by vertical break signature

---

## Setup

**Requirements**
```
numpy
pandas
scikit-learn
xgboost
matplotlib
sqlalchemy
mysql-connector-python
```

**Data**

The notebook expects either:
- A local `pitches.csv` (auto-generated on first run), or
- A MySQL database with a Statcast pitch table

On first run the notebook pulls from MySQL and caches to CSV. All subsequent runs — including after kernel restarts — load from the fast local CSV.

**GPU**

XGBoost training uses `device='cuda'`. Requires CUDA toolkit and an NVIDIA GPU. To verify:
```python
import xgboost as xgb
print(xgb.build_info())  # USE_CUDA should be True
```

To run on CPU only, change `device='cuda'` to `device='cpu'` in the XGBoost cell.

---

## Notebook Structure

| Section | Description |
|---------|-------------|
| 1. Imports & Setup | Libraries, global plot theme |
| 2. Data Loading | MySQL pull + CSV caching |
| 3. Preprocessing | Pitch type filtering, feature selection, train/test split |
| 4. Random Forest | Model training + classification report |
| 5. XGBoost | GPU-accelerated training + classification report |
| 6. Cross-Validation | 5-fold stratified CV with box plot visualization |
| 7. Model Evaluation | AUC, confusion matrices, feature importance table |
| 8. Results Visualization | Six-panel dashboard |
| 9. Key Findings | Analysis and takeaways |
