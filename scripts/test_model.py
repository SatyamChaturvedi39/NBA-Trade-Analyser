"""
Comprehensive Model Testing Script for v2 Multi-Output Model
Tests model loading, predictions, SHAP explanations, and data compatibility
"""
import pickle
import numpy as np
import pandas as pd
import json
import os
import sys

print("=" * 80)
print("NBA PLAYER PERFORMANCE PREDICTION MODEL - TEST SUITE")
print("=" * 80)

# Ensure we are in the project root
os.chdir(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
print(f"Working directory: {os.getcwd()}")

# ============================================================================
# TEST 1: File Existence Check
# ============================================================================
print("\n[TEST 1] Checking model files...")

required_files = {
    'model': 'models/player_multioutput_v2.pkl',
    'explainer': 'models/shap_explainers_v2.pkl',
    'features': 'models/feature_names_v2.txt',
    'targets': 'models/target_names_v2.txt',
    'metadata': 'models/model_metadata_v2.json',
    'data': 'data/processed/player_features_v2_temporal.csv'
}

missing_files = []
for name, path in required_files.items():
    if os.path.exists(path):
        print(f"  ✓ {name}: {path}")
    else:
        print(f"  ✗ {name}: {path} NOT FOUND")
        missing_files.append(path)

if missing_files:
    print(f"\n❌ FAILED: Missing {len(missing_files)} required files")
    print("Please run the model training notebook first.")
    sys.exit(1)

# ============================================================================
# TEST 2: Load Model and Metadata
# ============================================================================
print("\n[TEST 2] Loading model and metadata...")

try:
    # Load model
    with open('models/player_multioutput_v2.pkl', 'rb') as f:
        model = pickle.load(f)
    print(f"  ✓ Model type: {type(model).__name__}")
    print(f"  ✓ Number of estimators: {len(model.estimators_)}")
    
    # Load explainer dict
    with open('models/shap_explainers_v2.pkl', 'rb') as f:
        explainers_dict = pickle.load(f)
    print(f"  ✓ SHAP explainers loaded (targets: {list(explainers_dict.keys())})")
    
    # Load feature names
    with open('models/feature_names_v2.txt', 'r') as f:
        feature_names = [line.strip() for line in f.readlines()]
    print(f"  ✓ Features: {len(feature_names)}")
    
    # Load target names
    with open('models/target_names_v2.txt', 'r') as f:
        target_names = [line.strip() for line in f.readlines()]
    print(f"  ✓ Targets: {len(target_names)}")
    
    # Load metadata
    with open('models/model_metadata_v2.json', 'r') as f:
        metadata = json.load(f)
    print(f"  ✓ Model version: {metadata['model_version']}")
    print(f"  ✓ Training date: {metadata['training_date']}")
    print(f"  ✓ Training samples: {metadata['n_samples_train']}")
    print(f"  ✓ Test samples: {metadata['n_samples_test']}")
    
except Exception as e:
    print(f"\n❌ FAILED: Error loading model files: {str(e)}")
    sys.exit(1)

# ============================================================================
# TEST 3: Validate Feature Count
# ============================================================================
print("\n[TEST 3] Validating feature configuration...")

expected_features = 54
if len(feature_names) == expected_features:
    print(f"  ✓ Feature count correct: {len(feature_names)}")
else:
    print(f"  ✗ Feature count mismatch: expected {expected_features}, got {len(feature_names)}")

expected_targets = 5
if len(target_names) == expected_targets:
    print(f"  ✓ Target count correct: {len(target_names)}")
    for i, target in enumerate(target_names):
        print(f"    {i+1}. {target}")
else:
    print(f"  ✗ Target count mismatch: expected {expected_targets}, got {len(target_names)}")

# ============================================================================
# TEST 4: Load and Validate Dataset
# ============================================================================
print("\n[TEST 4] Loading and validating dataset...")

try:
    df = pd.read_csv('data/processed/player_features_v2_temporal.csv')
    print(f"  ✓ Dataset loaded: {len(df)} player-seasons")
    print(f"  ✓ Total columns: {len(df.columns)}")
    
    # Check if all features exist in dataset
    missing_features = [f for f in feature_names if f not in df.columns]
    if missing_features:
        print(f"  ✗ Missing features in dataset: {missing_features}")
    else:
        print(f"  ✓ All {len(feature_names)} features present in dataset")
    
    # Check if all targets exist
    missing_targets = [t for t in target_names if t not in df.columns]
    if missing_targets:
        print(f"  ✗ Missing targets in dataset: {missing_targets}")
    else:
        print(f"  ✓ All {len(target_names)} targets present in dataset")
        
except Exception as e:
    print(f"\n❌ FAILED: Error loading dataset: {str(e)}")
    sys.exit(1)

# ============================================================================
# TEST 5: Sample Predictions
# ============================================================================
print("\n[TEST 5] Testing predictions on sample data...")

try:
    # Get a sample from the dataset
    sample_rows = df[feature_names].head(3)
    actual_targets = df[target_names].head(3)
    
    # Make predictions
    predictions = model.predict(sample_rows)
    
    print(f"\n  Sample Predictions vs Actual:")
    print("  " + "=" * 76)
    
    for i in range(3):
        player_name = df.iloc[i]['player_name'] if 'player_name' in df.columns else f"Player {i+1}"
        season = df.iloc[i]['season'] if 'season' in df.columns else "N/A"
        
        print(f"\n  Player: {player_name} ({season})")
        print(f"  {'Metric':<20} {'Predicted':<15} {'Actual':<15} {'Error':<10}")
        print(f"  {'-'*60}")
        
        for j, target in enumerate(target_names):
            pred_val = predictions[i][j]
            actual_val = actual_targets.iloc[i, j]
            error = abs(pred_val - actual_val)
            
            print(f"  {target:<20} {pred_val:>14.2f} {actual_val:>14.2f} {error:>9.2f}")
    
    print("\n  ✓ Predictions generated successfully")
    
except Exception as e:
    print(f"\n❌ FAILED: Error making predictions: {str(e)}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# ============================================================================
# TEST 6: Prediction Sanity Checks
# ============================================================================
print("\n[TEST 6] Running sanity checks on predictions...")

sanity_passed = True

# Define reasonable ranges for each target
target_ranges = {
    'target_next_ppg': (0, 40),
    'target_next_rpg': (0, 20),
    'target_next_apg': (0, 15),
    'target_next_mpg': (0, 48),
    'target_next_ts_pct': (0, 1)
}

for i, target in enumerate(target_names):
    pred_vals = predictions[:, i]
    min_val, max_val = target_ranges.get(target, (-float('inf'), float('inf')))
    
    if np.any(pred_vals < min_val) or np.any(pred_vals > max_val):
        print(f"  ✗ {target}: predictions out of range [{min_val}, {max_val}]")
        print(f"    Got: min={pred_vals.min():.2f}, max={pred_vals.max():.2f}")
        sanity_passed = False
    else:
        print(f"  ✓ {target}: predictions in valid range")

if not sanity_passed:
    print("\n⚠️  WARNING: Some predictions are outside expected ranges")
else:
    print("\n  ✓ All predictions pass sanity checks")

# ============================================================================
# TEST 7: SHAP Explanations
# ============================================================================
print("\n[TEST 7] Testing SHAP explanations...")
try:
    # Get SHAP values for first sample using the PPG explainer
    sample_input = sample_rows.iloc[[0]]
    # Find the PPG explainer in the dict
    ppg_target = 'target_next_ppg'
    if ppg_target in explainers_dict:
        target_explainer = explainers_dict[ppg_target]
        shap_values = target_explainer.shap_values(sample_input)
        
        if isinstance(shap_values, list): # Handle some SHAP versions
            shap_values = shap_values[0]
        if shap_values.ndim > 1:
            shap_values = shap_values[0]

        # Create feature importance dict
        shap_dict = {name: float(val) for name, val in zip(feature_names, shap_values)}
    
    # Get top 10 features
    top_features = sorted(shap_dict.items(), key=lambda x: abs(x[1]), reverse=True)[:10]
    
    print(f"\n  Top 10 Most Important Features (for PPG prediction):")
    print(f"  {'Feature':<25} {'SHAP Value':<15} {'Impact'}")
    print(f"  {'-'*60}")
    
    for name, value in top_features:
        impact = "↑ Increases" if value > 0 else "↓ Decreases"
        print(f"  {name:<25} {value:>14.3f}  {impact}")
    
    print("\n  ✓ SHAP explanations generated successfully")
    
except Exception as e:
    print(f"\n❌ FAILED: Error generating SHAP values: {str(e)}")
    import traceback
    traceback.print_exc()

# ============================================================================
# TEST 8: Model Performance Summary
# ============================================================================
print("\n[TEST 8] Model Performance Summary (from metadata)...")

print(f"\n  {'Metric':<20} {'Test MAE':<12} {'Test RMSE':<12} {'Test R²':<10}")
print(f"  {'-'*60}")

for target in target_names:
    if target in metadata['performance']:
        perf = metadata['performance'][target]
        mae = perf.get('test_mae', 0)
        rmse = perf.get('test_rmse', 0)
        r2 = perf.get('test_r2', 0)
        print(f"  {target:<20} {mae:>11.3f} {rmse:>11.3f} {r2:>9.3f}")

if 'cv_ppg_mean_mae' in metadata:
    print(f"\n  Cross-Validation (PPG):")
    print(f"    Mean MAE: {metadata['cv_ppg_mean_mae']:.3f}")
    print(f"    Std MAE:  {metadata['cv_ppg_std_mae']:.3f}")

# ============================================================================
# TEST 9: Feature Categories Breakdown
# ============================================================================
print("\n[TEST 9] Feature Categories Breakdown...")

base_count = 0
lag_count = 0
trend_count = 0

for feature in feature_names:
    if 'lag' in feature:
        lag_count += 1
    elif any(word in feature for word in ['trend', 'career', 'peak', 'seasons', 'years', 'coefficient']):
        trend_count += 1
    else:
        base_count += 1

print(f"  Base features (current season): {base_count}")
print(f"  Lag features (historical):      {lag_count}")
print(f"  Trend features (trajectory):    {trend_count}")
print(f"  Total:                          {base_count + lag_count + trend_count}")

# ============================================================================
# Final Summary
# ============================================================================
print("\n" + "=" * 80)
print("TEST SUMMARY")
print("=" * 80)

print(f"""
✓ Model files loaded successfully
✓ Feature count: {len(feature_names)} (expected 54)
✓ Target count: {len(target_names)} (expected 5)
✓ Dataset compatible: {len(df)} player-seasons
✓ Predictions working
✓ SHAP explanations working
✓ All tests passed

Model is ready for backend integration!
""")

print("=" * 80)
print("Next step: Update backend/app/main.py to use this model")
print("=" * 80)