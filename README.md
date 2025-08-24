# Milestone 3

## 1. Data Preprocessing and Feature Engineering

### Dataset Overview
- **Source**: MLB 2023 season games via pybaseball API
- **Initial Data**: 2,430 games with 29 features
- **Final Dataset**: 2,430 games with 81 engineered features
- **Target**: Binary classification (home team win/loss)

### Preprocessing Pipeline Implementation

Our preprocessing approach addressed scaling, imputation, encoding, and feature expansion through a systematic multi-step pipeline:

**Data Cleaning**: Removed outcome-leaking features (home_runs, away_runs, run_diff) to ensure predictions use only pre-game information.

**Feature Engineering**: Created baseball-specific metrics from team statistics:
- **Team Strength Differentials**: Delta features (home_OPS - away_OPS) 
- **Performance Ratios**: Ratio features (home_ERA / away_ERA)
- **Rolling Form**: Time-aware win percentages over 7, 10, 15-game windows
- **ERA Advantage**: away_ERA - home_ERA (positive indicates home pitching advantage)

```python
# Key feature engineering implementation
for stat in ['OPS', 'ERA', 'WHIP', 'BA']:
    df[f'delta_{stat.lower()}'] = df[f'home_{stat}'] - df[f'away_{stat}']
    df[f'ratio_{stat.lower()}'] = df[f'home_{stat}'] / df[f'away_{stat}']

# Rolling averages with leakage prevention
df['home_last10_win_pct'] = (
    df.groupby('home_team')['home_win']
    .shift(1).rolling(window=10, min_periods=1).mean()
)
```

**Scaling and Transformation**:
- **Numerical Features**: StandardScaler after median imputation
- **Categorical Features**: One-hot encoding with mode imputation
- **Polynomial Expansion**: Degree-2 features applied to high-signal variables (deltas, ratios)

**Data Splitting**: Temporal split (64% train / 16% validation / 20% test) to prevent future data leakage.

**Final Feature Space**: 1,689 training samples × 81 features with balanced classes (52% home wins).

---

## 2. Model Training and Performance Analysis

### Model Training Implementation

```python
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score

# Model configurations
models = {
    "SVM_RBF": SVC(C=2.0, gamma='scale', probability=True, 
                   class_weight='balanced', random_state=42),
    "DecisionTree": DecisionTreeClassifier(max_depth=5, min_samples_leaf=5, 
                                         random_state=42),
    "KNN": KNeighborsClassifier(n_neighbors=15),
    "NaiveBayes": GaussianNB()
}

# Training and evaluation loop
results = []
for name, model in models.items():
    model.fit(X_tr_dense, y_tr)
    train_pred = model.predict(X_tr_dense)
    val_pred = model.predict(X_va_dense)
    
    results.append({
        "model": name,
        "train_acc": accuracy_score(y_tr, train_pred),
        "val_acc": accuracy_score(y_va, val_pred)
    })
```

### Initial Model Comparison

Results from training four algorithms with systematic evaluation:

| Model | Training Accuracy | Validation Accuracy | Training-Val Gap |
|-------|------------------|-------------------|------------------|
| **SVM_RBF** | **67.3%** | **60.3%** | **7.0%** |
| Decision Tree | 60.8% | 59.6% | 1.2% |
| Naive Bayes | 59.1% | 59.8% | -0.7% |
| KNN | 64.7% | 56.7% | 8.0% |

### Hyperparameter Optimization Results

**SVM Grid Search** (Top 5 configurations):

| C | gamma | Training Acc | Validation Acc | Gap |
|---|-------|-------------|---------------|-----|
| **1.0** | **scale** | **63.0%** | **61.2%** | **1.8%** |
| 2.0 | scale | 67.3% | 60.3% | 7.0% |
| 0.5 | 0.05 | 71.5% | 60.0% | 11.5% |
| 0.5 | scale | 61.6% | 59.3% | 2.3% |
| 1.0 | 0.05 | 77.5% | 58.9% | 18.6% |

```python
# Hyperparameter optimization implementation
svm_grid = {"C": [0.5, 1.0, 2.0], "gamma": ['scale', 0.1, 0.05]}
best_svm = evaluate_grid(SVC, svm_grid, X_tr, y_tr, X_va, y_va)
# Result: C=1.0, gamma='scale' achieved optimal generalization
```

**Decision Tree Grid Search** (Top 4 configurations):

| max_depth | min_samples_leaf | Training Acc | Validation Acc |
|-----------|------------------|-------------|---------------|
| 4 | 1 | 59.7% | **59.8%** |
| 5 | 3 | 59.5% | **59.8%** |
| 6 | 5 | 59.4% | 59.6% |
| 10 | 5 | 60.8% | 59.6% |

### Training vs. Validation Performance Analysis

**Key Performance Improvement**:
- **Original SVM**: 67.3% train → 60.3% val (7.0% gap)
- **Optimized SVM**: 63.0% train → 61.2% val (1.8% gap)
- **Improvement**: 73% reduction in overfitting + 0.9% validation accuracy gain

**Sample Model Predictions**:

| Dataset | Actual | Predicted | Probability | Result |
|---------|--------|-----------|-------------|--------|
| Train | Home Win | Home Win | 60.9% | ✓ Correct |
| Train | Home Win | Away Win | 46.5% | ✗ Incorrect |
| Validation | Home Win | Home Win | 58.9% | ✓ Correct |
| Validation | Away Win | Home Win | 54.7% | ✗ Incorrect |
| Test | Home Win | Home Win | 62.7% | ✓ Correct |
| Test | Away Win | Away Win | 40.3% | ✓ Correct |

---

## 3. Bias-Variance Analysis and Model Selection

### Model Diagnosis Framework

We developed a systematic approach to classify model performance on the bias-variance spectrum:

```python
def fit_position(train_acc, val_acc, baseline=0.52, gap_thresh=0.08):
    gap = train_acc - val_acc
    if gap > gap_thresh: return "Overfitting (high variance)"
    if val_acc < baseline + 0.03: return "Underfitting (high bias)" 
    return "Near sweet-spot"
```

### Comprehensive Model Analysis

| Model | Train Acc | Val Acc | Gap | Diagnosis |
|-------|-----------|---------|-----|----------|
| **SVM_RBF** | **0.673** | **0.603** | **0.070** | **Near sweet-spot** |
| Decision Tree | 0.608 | 0.596 | 0.012 | Near sweet-spot |
| Naive Bayes | 0.591 | 0.598 | -0.007 | Near sweet-spot  |
| KNN | 0.647 | 0.567 | 0.080 | Near sweet-spot  |

### Hyperparameter Impact on Bias-Variance

**SVM Optimization Journey**:
- **Initial**: C=2.0 → 7.0% gap (moderate overfitting)
- **Optimized**: C=1.0 → 1.8% gap (near optimal complexity)
- **Over-regularized**: C=0.5 → potential underfitting

The optimization successfully moved our model from moderate overfitting to the optimal complexity region while improving validation performance.

### Next Model Recommendations

**Based on systematic bias-variance analysis and "Near sweet-spot" positioning**:

1. **Random Forest (Primary Choice)**
   - **Rationale**: Ensemble bagging reduces variance while maintaining high performance
   - **Expected Performance**: 62-64% validation accuracy with improved stability
   - **Implementation Plan**: 100-200 trees, max_depth=8-12, bootstrap sampling

2. **XGBoost/Gradient Boosting**
   - **Rationale**: Sequential boosting can capture complex feature interactions our SVM might miss
   - **Expected Performance**: 63-65% accuracy potential with proper regularization
   - **Risk Management**: Requires careful early stopping and regularization to prevent overfitting

3. **Ensemble Combination (SVM + Random Forest)**
   - **Rationale**: Combines SVM's optimal complexity with Random Forest's variance reduction
   - **Method**: Weighted voting or stacking approach
   - **Expected**: Best of both algorithms while reducing individual model weaknesses

**Why These Specific Models**: Since our current SVM sits in the optimal complexity region, improvements should focus on ensemble methods that reduce variance and capture feature interactions, rather than single models requiring extensive hyperparameter tuning.

---

## 4. Conclusions

### Model Performance Summary

Our hyperparameter-optimized SVM achieved **61.2% validation accuracy** with excellent generalization (1.8% training-validation gap). This represents a 9.2 percentage point improvement over the 52% baseline while demonstrating proper model complexity control.

**Key Achievements**:
- **73% reduction in overfitting** through systematic hyperparameter optimization
- **Feature engineering success**: 29 → 81 meaningful features capturing baseball dynamics
- **Leakage-free methodology**: Temporal splitting and careful preprocessing maintained prediction integrity
- **Calibrated predictions**: Probability outputs (40-63%) reflect appropriate confidence levels

### Model Limitations

**Performance Constraints**: 61.2% accuracy reflects baseball's inherent unpredictability where many games are genuinely close contests influenced by factors beyond team statistics.

**Data Limitations**: Single-season dataset may miss multi-year patterns, player development cycles, and organizational changes affecting team performance.

**Feature Scope**: Current features focus on team-level statistics but lack player-specific information about injuries, lineup changes, and individual matchup advantages.

### Improvement Strategies

Target: 63-65% accuracy**:
- **Ensemble Methods**: Random Forest implementation for variance reduction
- **Dataset Expansion**: Include 2021-2023 seasons for broader pattern recognition
- **Advanced Features**: Player-level statistics, weather conditions, rest days, travel distance

---

## 5. Repository Structure and Documentation

### Current Project Organization
```
CSE-151A-Group-Project/
├── .venv/                          # Virtual environment
├── data/
│   └── raw/
│       └── games_2023.csv          # Raw MLB game data
├── figs/                           # Generated visualizations  
├── notebooks/
│   ├── milestone2_exploration.ipynb # Data exploration & preprocessing
│   └── milestone3_modeling.ipynb   # Model training & evaluation
├── .gitignore
├── README.md                       # Project documentation
└── requirements.txt               # Python dependencies
```

## Project Files
- **[milestone2_exploration.ipynb](notebooks/milestone2_exploration.ipynb)** - Data preprocessing and feature engineering
- **[milestone3_modeling.ipynb](notebooks/milestone3_modeling.ipynb)** - Model training and optimization
- **[games_2023.csv](data/raw/games_2023.csv)** - Raw MLB data

## Next Phase Development
- Random Forest ensemble implementation
- Multi-season dataset expansion
- Advanced feature engineering (player stats, weather)
- Target: 63-65% validation accuracy


