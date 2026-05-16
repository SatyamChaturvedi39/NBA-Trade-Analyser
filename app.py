"""
NBA Trade Analyzer - Production Flask Backend
With industry-standard SHAP explainability (per-target TreeExplainer)
"""
from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
from dotenv import load_dotenv
import os
import pickle
import joblib
import json
import shap
import numpy as np
import pandas as pd
from datetime import datetime
from pymongo import MongoClient

# Internal imports
from services.trade_analyzer import (
    normalize_name,
    calculate_player_medical_score,
    calculate_roster_health_score,
    run_injury_adjusted_simulation,
    calc_trade_score,
    get_trade_rating
)

# Load environment variables
load_dotenv()

# ============================================================================
# App Configuration
# ============================================================================
app = Flask(__name__)

CORS(app)

MONGODB_URI = os.getenv('MONGODB_URI')
DATABASE_NAME = os.getenv('DATABASE_NAME', 'nba_db')
COLLECTION_NAME = os.getenv('COLLECTION_NAME', 'players')
FLASK_PORT = int(os.getenv('FLASK_PORT', 5000))

# ============================================================================
# SHAP Feature Label Map & Reason Templates
# ============================================================================

FEATURE_LABEL_MAP = {
    # --- Current season base stats ---
    'age':                        'Player Age',
    'games_played':               'Games Played',
    'minutes_per_game':           'Minutes Per Game',
    'points_per_game':            'Current Season PPG',
    'rebounds_per_game':          'Current Season RPG',
    'assists_per_game':           'Current Season APG',
    'steals_per_game':            'Steals Per Game',
    'blocks_per_game':            'Blocks Per Game',
    'turnovers_per_game':         'Turnovers Per Game',
    'field_goal_pct':             'Field Goal %',
    'free_throw_pct':             'Free Throw %',
    'true_shooting_pct':          'True Shooting %',
    'points_per_minute':          'Scoring Efficiency (Pts/Min)',
    'usage_rate':                 'Usage Rate',
    'assist_rate':                'Assist Rate',
    'rebound_rate':               'Rebound Rate',
    # --- Lag features ---
    'ppg_lag1':                   'Last Season PPG',
    'ppg_lag2':                   '2 Seasons Ago PPG',
    'ppg_lag3':                   '3 Seasons Ago PPG',
    'ppg_lag4':                   '4 Seasons Ago PPG',
    'ppg_lag5':                   '5 Seasons Ago PPG',
    'mpg_lag1':                   'Last Season Minutes',
    'mpg_lag2':                   '2 Seasons Ago Minutes',
    'mpg_lag3':                   '3 Seasons Ago Minutes',
    'mpg_lag4':                   '4 Seasons Ago Minutes',
    'mpg_lag5':                   '5 Seasons Ago Minutes',
    'rpg_lag1':                   'Last Season RPG',
    'rpg_lag2':                   '2 Seasons Ago RPG',
    'rpg_lag3':                   '3 Seasons Ago RPG',
    'apg_lag1':                   'Last Season APG',
    'apg_lag2':                   '2 Seasons Ago APG',
    'apg_lag3':                   '3 Seasons Ago APG',
    'spg_lag1':                   'Last Season Steals',
    'bpg_lag1':                   'Last Season Blocks',
    'games_lag1':                 'Games Played (Last Season)',
    'games_lag2':                 'Games Played (2 Seasons Ago)',
    'games_lag3':                 'Games Played (3 Seasons Ago)',
    'ts_pct_lag1':                'True Shooting % (Last Season)',
    'ts_pct_lag2':                'True Shooting % (2 Seasons Ago)',
    'ts_pct_lag3':                'True Shooting % (3 Seasons Ago)',
    'fg_pct_lag1':                'Field Goal % (Last Season)',
    'fg_pct_lag2':                'Field Goal % (2 Seasons Ago)',
    # --- Trend features ---
    'ppg_trend_2yr':              '2-Year PPG Trajectory',
    'ppg_trend_3yr':              '3-Year PPG Trajectory',
    'ppg_trend_4yr':              '4-Year PPG Trajectory',
    'mpg_trend_2yr':              '2-Year Minutes Trajectory',
    # --- Career features ---
    'seasons_in_dataset':         'NBA Seasons Played',
    'years_since_peak_ppg':       'Years Since Career Peak',
    'peak_ppg':                   'Career Peak PPG',
    'career_ppg_avg':             'Career Average PPG',
    'career_ppg_std':             'Career Scoring Variability',
    'career_games_avg':           'Career Games Average',
    'career_mpg_avg':             'Career Minutes Average',
    'ppg_coefficient_variation':  'Scoring Consistency (CV)',
}

# Target display names
TARGET_DISPLAY_MAP = {
    'target_next_ppg':    {'label': 'Next Season PPG',    'unit': 'pts',  'key': 'ppg'},
    'target_next_rpg':    {'label': 'Next Season RPG',    'unit': 'reb',  'key': 'rpg'},
    'target_next_apg':    {'label': 'Next Season APG',    'unit': 'ast',  'key': 'apg'},
    'target_next_mpg':    {'label': 'Next Season MPG',    'unit': 'min',  'key': 'mpg'},
    'target_next_ts_pct': {'label': 'Next Season TS%',    'unit': '%',    'key': 'ts_pct'},
}

def _fmt_value(feature, value):
    """Format a feature's raw value for display in reason strings."""
    pct_features = {'field_goal_pct', 'free_throw_pct', 'true_shooting_pct',
                    'ts_pct_lag1', 'ts_pct_lag2', 'ts_pct_lag3',
                    'fg_pct_lag1', 'fg_pct_lag2', 'usage_rate', 'assist_rate', 'rebound_rate'}
    if feature in pct_features:
        return f"{value*100:.1f}%"
    if 'trend' in feature:
        return f"{value:+.2f}/yr"
    if feature == 'age':
        return f"{int(value)}"
    if feature.endswith('pct') or feature.endswith('variation'):
        return f"{value:.3f}"
    return f"{value:.1f}"


def _build_reason(feature, value, shap_val, target_key, current_age=None):
    """Build a plain-English fan-friendly reason sentence for a SHAP factor."""
    label = FEATURE_LABEL_MAP.get(feature, feature.replace('_', ' ').title())
    up_or_down = "pushing the estimate up" if shap_val > 0 else "pulling the estimate down"
    formatted_val = _fmt_value(feature, value)
    abs_shap = abs(shap_val)

    # For the age feature, prefer the current real age over the raw dataset age
    display_age = current_age if (feature == 'age' and current_age is not None) else int(value)

    # ── Plain-English templates (no jargon) ─────────────────────────────────────
    templates = {
        # Scoring history
        'points_per_game': f"His current season scoring of {formatted_val} PPG is {'above' if shap_val > 0 else 'below'} the model's league-wide average baseline, which {'lifts' if shap_val > 0 else 'pulls down'} the projection. {'Scoring above average signals a strong offensive role.' if shap_val > 0 else 'As a lower-volume scorer, the model naturally projects him below the league mean.'}",
        'ppg_lag1':     f"He scored {formatted_val} points per game last season, which is the strongest recent signal — {'a good sign for continued output' if shap_val > 0 else 'suggesting a possible dip ahead'}.",
        'ppg_lag2':     f"Two seasons ago he put up {formatted_val} points per game — that historical scoring record {'helps' if shap_val > 0 else 'slightly lowers'} the projection.",
        'ppg_lag3':     f"His points tally from three seasons ago ({formatted_val} PPG) still anchors the long-term baseline {'upward' if shap_val > 0 else 'downward'}.",
        'ppg_lag4':     f"Four seasons of scoring history ({formatted_val} PPG) {'reinforces' if shap_val > 0 else 'tempers'} the long-run forecast.",
        'ppg_lag5':     f"A five-season-old scoring record of {formatted_val} PPG {'slightly lifts' if shap_val > 0 else 'slightly lowers'} the baseline estimate.",
        # Scoring trend
        'ppg_trend_2yr': f"His scoring has been {'climbing' if shap_val > 0 else 'declining'} over the last two seasons — that trajectory {'boosts' if shap_val > 0 else 'lowers'} confidence in high output next year.",
        'ppg_trend_3yr': f"Looking at three years of scoring data, the trend is {'upward' if shap_val > 0 else 'downward'} — a {'positive' if shap_val > 0 else 'cautionary'} signal for next season.",
        'ppg_trend_4yr': f"His four-year scoring arc is {'improving' if shap_val > 0 else 'declining'}, which {'strengthens' if shap_val > 0 else 'weakens'} the overall projection.",
        # Minutes / workload
        'minutes_per_game': f"He plays around {formatted_val} minutes per game — {'heavy court time shows the coaching staff rely on him heavily' if shap_val > 0 else 'a lighter role limits how many stats he can accumulate per game'}.",
        'mpg_lag1':     f"Last season he averaged {formatted_val} minutes per game — {'consistent heavy workload supports strong numbers' if shap_val > 0 else 'reduced playing time signals a smaller role going forward'}.",
        'mpg_lag2':     f"Two seasons ago he logged {formatted_val} minutes per night — {'sustained heavy use' if shap_val > 0 else 'reduced role'} in his history.",
        'mpg_trend_2yr': f"His minutes per game have been {'increasing' if shap_val > 0 else 'decreasing'} over the last two seasons, suggesting {'growing' if shap_val > 0 else 'shrinking'} opportunity.",
        # Shooting efficiency (translated away from TS%)
        'true_shooting_pct': f"He converts shots at an {'above-average' if shap_val > 0 else 'below-average'} rate overall, meaning every possession he's involved in {'generates more' if shap_val > 0 else 'generates fewer'} points for the team.",
        'ts_pct_lag1':  f"Last season his shot-making efficiency was {'strong' if shap_val > 0 else 'below par'} — that efficiency track record {'lifts' if shap_val > 0 else 'lowers'} the forecast.",
        'ts_pct_lag2':  f"His shot quality two seasons ago was {'solid' if shap_val > 0 else 'shaky'}, which still {'feeds positively into' if shap_val > 0 else 'slightly drags on'} the projection.",
        'ts_pct_lag3':  f"A three-year-old efficiency rating of {formatted_val} {'anchors a positive' if shap_val > 0 else 'suggests historically lower'} shooting baseline.",
        'fg_pct_lag1':  f"He shot {formatted_val} from the field last season — {'efficient scoring that boosts' if shap_val > 0 else 'below-average accuracy that lowers'} the outlook.",
        'fg_pct_lag2':  f"Two seasons ago he connected on {formatted_val} of his shots — {'a solid' if shap_val > 0 else 'a modest'} historical mark.",
        # Role / ball-handling (usage/assist rate — purely direction driven, no jargon)
        'usage_rate':   f"He {'plays a primary offensive role and has the ball in his hands frequently — the model expects high output from him' if shap_val > 0 else 'operates in a secondary role and does not dominate the ball — the model expects leaner counting stats as a result'}.",
        'assist_rate':  f"He creates scoring chances for teammates {'frequently' if shap_val > 0 else 'less often than average'} — {'strong playmaking volume positively affects' if shap_val > 0 else 'a lower passing role negatively affects'} the overall forecast.",
        'rebound_rate': f"He grabs {'more' if shap_val > 0 else 'fewer'} rebounds per possession than the average player at his position, which {'helps' if shap_val > 0 else 'slightly drags'} the projection.",
        # Age
        'age':          f"At {display_age} years old, he is {'in the prime years of his athletic career — a positive signal' if shap_val >= 0 else 'entering the later stage of his career, where a gradual performance decline is normal and expected'}.",
        # Career peaks & averages
        'peak_ppg':     f"His career-best scoring season was {formatted_val} PPG — {'a high ceiling that raises' if shap_val > 0 else 'a modest peak that tempers'} long-term expectations.",
        'years_since_peak_ppg': f"He reached his scoring peak {'recently' if value <= 1 else f'{int(value)} years ago'} — {'still near his best years' if shap_val >= 0 else 'likely past his statistical prime'}.",
        'career_ppg_avg': f"His career average of {formatted_val} PPG {'sets a strong baseline' if shap_val > 0 else 'reflects a historically lighter scoring role'}.",
        'ppg_coefficient_variation': f"His season-to-season scoring has been {'very consistent — reliable output year after year' if shap_val > 0 else 'somewhat unpredictable — his numbers vary more than most players'}.",
        # Games played / availability
        'games_played': f"He played {formatted_val} games this season — {'staying healthy all year is a big positive for the projection' if shap_val > 0 else 'missing games signals an injury or rotation concern the model weighs negatively'}.",
        'games_lag1':   f"Last season he appeared in {formatted_val} games — {'a full, healthy season is encouraging' if shap_val > 0 else 'limited appearances last year raises durability questions'}.",
        'games_lag2':   f"Two seasons ago he played {formatted_val} games — {'consistent availability' if shap_val > 0 else 'reduced availability'} in his history.",
        'games_lag3':   f"Three seasons ago he suited up {formatted_val} times — {'sustained durability' if shap_val > 0 else 'recurring absence patterns'} in his track record.",
        # Rebounding & assists history
        'rpg_lag1':     f"Last season he grabbed {formatted_val} rebounds per game — {'strong board presence' if shap_val > 0 else 'limited rebounding'}  that {'lifts' if shap_val > 0 else 'lowers'} the forecast.",
        'rpg_lag2':     f"Two seasons ago he averaged {formatted_val} rebounds per game — {'solid' if shap_val > 0 else 'modest'} historical rebounding.",
        'rpg_lag3':     f"Three seasons of rebounding at {formatted_val} RPG {'reinforces' if shap_val > 0 else 'tempers'} the expectation.",
        'apg_lag1':     f"He averaged {formatted_val} assists per game last season — {'a strong passer who creates for others, lifting' if shap_val > 0 else 'limited playmaking that lowers'} the estimate.",
        'apg_lag2':     f"Two seasons ago he averaged {formatted_val} assists per game — {'an established playmaking role' if shap_val > 0 else 'a historically limited passing role'}.",
        'apg_lag3':     f"Three seasons of dishing {formatted_val} APG {'anchors playmaking expectations' if shap_val > 0 else 'points to a non-primary passer role'}.",
        # Defensive stats
        'spg_lag1':     f"He averaged {formatted_val} steals per game last season — {'active defense and high effort level that correlates with' if shap_val > 0 else 'limited defensive activity that slightly reduces'} projected output.",
        'bpg_lag1':     f"He blocked {formatted_val} shots per game last season — {'rim protection that adds value and lifts' if shap_val > 0 else 'limited shot-blocking that softly lowers'} the estimate.",
        # Turnovers
        'turnovers_per_game': f"He coughs up the ball {formatted_val} times per game — {'acceptable for his role' if shap_val >= 0 else 'higher turnovers signal ball-handling risk that slightly reduces the projection'}.",
        # Free throws
        'free_throw_pct': f"He makes {formatted_val} of his free throws — {'a reliable foul-line shooter' if shap_val > 0 else 'below-average free throw accuracy that reduces scoring efficiency'}.",
        # NBA experience
        'seasons_in_dataset': f"He has {formatted_val} seasons of NBA data in the model — {'a deep track record makes the projection more reliable' if shap_val > 0 else 'limited data means the model is extrapolating more, adding uncertainty'}.",
        # Scoring Efficiency
        'points_per_minute': f"He scores {formatted_val} points per minute — {'highly efficient scoring that lifts the offensive projection' if shap_val > 0 else 'lower per-minute scoring that suggests a more limited offensive output'}.",
    }

    if feature in templates:
        return templates[feature]

    # Generic fan-friendly fallback
    direction = 'positively influences' if shap_val > 0 else 'negatively influences'
    return f"{label} {direction} the prediction for this player."


# ============================================================================
# Database Connection & Data Loading
# ============================================================================

def load_data_from_db():
    """Load player data from MongoDB and flatten it into a DataFrame."""
    print(f"Connecting to MongoDB at {DATABASE_NAME}...")
    try:
        if not MONGODB_URI:
            raise ValueError("MONGODB_URI not set")

        client = MongoClient(MONGODB_URI, serverSelectionTimeoutMS=5000)
        db = client[DATABASE_NAME]
        collection = db[COLLECTION_NAME]

        if collection.count_documents({}) == 0:
            print("[WARN] MongoDB is empty. Falling back to CSV.")
            return pd.read_csv('data/processed/player_features_v2_temporal.csv')

        print("Fetching documents from MongoDB...")
        cursor = collection.find({})

        flattened_rows = []
        for doc in cursor:
            player_name = doc.get('player_name', 'Unknown')
            stats = doc.get('stats', [])
            for s in stats:
                s['player_name'] = player_name
                flattened_rows.append(s)

        df = pd.DataFrame(flattened_rows)
        print(f"[OK] Successfully loaded {len(df)} rows from MongoDB")
        client.close()
        return df

    except Exception as e:
        print(f"[WARN] Database connection failed: {e}")
        print("Falling back to local CSV file...")
        try:
            return pd.read_csv('data/processed/player_features_v2_temporal.csv')
        except Exception:
            print("[ERROR] Critical: Could not load data from DB or CSV")
            return None


# ============================================================================
# Load ML Model and Artifacts
# ============================================================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

print("Loading ML model and artifacts...")

try:
    with open(os.path.join(BASE_DIR, 'models', 'player_multioutput_v2.pkl'), 'rb') as f:
        ml_model = pickle.load(f)
    print("[OK] Loaded multi-output model")

    with open(os.path.join(BASE_DIR, 'models', 'feature_names_v2.txt'), 'r') as f:
        feature_names = [line.strip() for line in f.readlines()]
    print(f"[OK] Loaded {len(feature_names)} feature names")

    with open(os.path.join(BASE_DIR, 'models', 'target_names_v2.txt'), 'r') as f:
        target_names = [line.strip() for line in f.readlines()]
    print(f"[OK] Loaded {len(target_names)} target names")

    with open(os.path.join(BASE_DIR, 'models', 'model_metadata_v2.json'), 'r') as f:
        model_metadata = json.load(f)
    print(f"[OK] Loaded model metadata (version {model_metadata.get('model_version')})")

    # ── Load Injury Model ───────────────────────────────────────────────────
    print("Loading injury classification model...")
    try:
        injury_model = joblib.load(os.path.join(BASE_DIR, 'models', 'injury_clf.pkl'))
        print("[OK] Loaded injury model")
    except Exception as e:
        print(f"[WARN] Could not load injury_clf.pkl: {e}")
        injury_model = None

    # ── Load pre-built SHAP TreeExplainers ────────────────────────────────────
    print("Loading SHAP explainers...")
    try:
        with open(os.path.join(BASE_DIR, 'models', 'shap_explainers_v2.pkl'), 'rb') as f:
            shap_explainers = pickle.load(f)
        print(f"[OK] Loaded {len(shap_explainers)} SHAP explainers")
    except Exception as e:
        print(f"[WARN] Could not load shap_explainers_v2.pkl. Please run the notebook builder. Error: {e}")
        shap_explainers = {}

    # ── Load player data ──────────────────────────────────────────────────────
    players_df = load_data_from_db()

    if players_df is not None:
        # Load auxiliary data for trade analysis
        print("Enriching player data with injury and team features...")
        try:
            try:
                from pymongo import MongoClient
                client = MongoClient(MONGODB_URI, serverSelectionTimeoutMS=5000)
                db = client[DATABASE_NAME]
                
                # Fetch Injuries
                if 'injuries' in db.list_collection_names() and db['injuries'].count_documents({}) > 0:
                    print("Fetching injuries from MongoDB...")
                    injury_df = pd.DataFrame(list(db['injuries'].find({})))
                else:
                    print("Falling back to local injuries CSV...")
                    injury_df = pd.read_csv(os.path.join(BASE_DIR, 'data', 'processed', 'player_injury_score_2020_2025_cleaned.csv'))
                
                # Fetch Teams
                if 'teams' in db.list_collection_names() and db['teams'].count_documents({}) > 0:
                    print("Fetching teams from MongoDB...")
                    team_df = pd.DataFrame(list(db['teams'].find({})))
                else:
                    print("Falling back to local teams CSV...")
                    team_df = pd.read_csv(os.path.join(BASE_DIR, 'data', 'processed', 'team_features_temporal.csv'))
                
                client.close()
            except Exception as dbe:
                print(f"[WARN] Failed to fetch auxiliary data from MongoDB: {dbe}. Falling back to CSVs.")
                injury_df = pd.read_csv(os.path.join(BASE_DIR, 'data', 'processed', 'player_injury_score_2020_2025_cleaned.csv'))
                team_df = pd.read_csv(os.path.join(BASE_DIR, 'data', 'processed', 'team_features_temporal.csv'))
            
            # Normalize names for merging
            players_df['normalized_name'] = players_df['player_name'].apply(normalize_name)
            
            # Group injury/team data to get latest stats
            latest_injury = injury_df.sort_values(['Name', 'season']).groupby('Name').last().reset_index()
            latest_injury['normalized_name'] = latest_injury['Name'].apply(normalize_name)
            
            # Select relevant injury features to merge
            injury_features = [
                'normalized_name', 'games_missed', 'games_missed_last_season', 
                'total_days_missed', 'minor_count', 'moderate_count', 
                'severe_count', 'has_severe_injury'
            ]
            
            # Merge injury data
            players_df = players_df.merge(
                latest_injury[injury_features],
                on='normalized_name',
                how='left'
            )
            
            # Fill missing values for players NOT in injury database (assume healthy)
            for col in injury_features[1:]:
                players_df[col] = players_df[col].fillna(0)
            
            # Prediction with injury model if available
            if injury_model:
                inj_feat_list = ['games_missed', 'games_missed_last_season', 'total_days_missed',
                                'minor_count', 'moderate_count', 'severe_count', 'has_severe_injury']
                try:
                    players_df['injury_risk_prob'] = injury_model.predict_proba(players_df[inj_feat_list])[:, 1]
                except AttributeError as e:
                    # Fix Scikit-Learn 1.5 vs 1.8 version mismatches with LogisticRegression
                    # Some pickled models lose the multi_class attribute down-level. We dummy the feature.
                    print(f"[WARN] Handling Scikit-Learn Version Mismatch: {e}")
                    import warnings
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore")
                        # Hack to inject the missing attribute into the local object instance
                        if hasattr(injury_model, 'named_steps'):
                            # It's a pipeline
                            classifier = injury_model.named_steps.get('classifier') or injury_model.steps[-1][1]
                            if not hasattr(classifier, 'multi_class'):
                                classifier.multi_class = 'auto'
                        elif not hasattr(injury_model, 'multi_class'):
                            injury_model.multi_class = 'auto'
                        
                        players_df['injury_risk_prob'] = injury_model.predict_proba(players_df[inj_feat_list])[:, 1]
                        
                players_df['injury_risk_category'] = pd.cut(
                    players_df['injury_risk_prob'],
                    bins=[0, 0.3, 0.5, 0.7, 1.0],
                    labels=['Low', 'Moderate', 'High', 'Very High']
                ).astype(str)
            else:
                players_df['injury_risk_prob'] = 0.05
                players_df['injury_risk_category'] = 'Low'

            # --- MASS PERFORMANCE PREDICTIONS (Align with M2.ipynb) ---
            print("Generating performance predictions for all players...")
            try:
                # Ensure all features exist, fill with 0 if missing
                X_perf = players_df[feature_names].fillna(0)
                perf_preds_all = ml_model.predict(X_perf.values)
                
                # Assign back to main dataframe
                for i, target in enumerate(target_names):
                    players_df[target] = perf_preds_all[:, i]
                print(f"[OK] Generated predictions for {len(players_df)} players across {len(target_names)} targets")
            except Exception as e:
                print(f"[WARN] Failed to generate mass performance predictions: {e}")

            # Pre-calculate medical scores
            print("Pre-calculating medical scores...")
            medical_results = players_df.apply(calculate_player_medical_score, axis=1)
            players_df['medical_score'] = [m['medical_score'] for m in medical_results]
            players_df['medical_grade'] = [m['medical_grade'] for m in medical_results]
            
            # --- SALARY INTEGRATION (Align with M2_fix.ipynb) ---
            print("Loading salary data...")
            salary_paths = ["nba_salary_cleaned.csv", "data/nba_salary_cleaned.csv", os.path.join(BASE_DIR, "notebooks", "nba_salary_cleaned.csv")]
            salary_df = None
            for path in salary_paths:
                full_path = os.path.join(BASE_DIR, path) if not os.path.isabs(path) else path
                if os.path.exists(full_path):
                    salary_df = pd.read_csv(full_path)
                    print(f"[OK] Found salary data at {full_path}")
                    break
            
            if salary_df is not None:
                # Most recent salary (get the latest available year for every player)
                latest_salary = salary_df.sort_values('season').groupby('name').last().reset_index()
                latest_salary['normalized_name'] = latest_salary['name'].apply(normalize_name)
                
                salary_cols = ['normalized_name', 'salary', 'total_contract_value', 'years_remaining', 'cap_hit_percent', 'luxury_tax_risk', 'npv_salary']
                cols_to_use = [c for c in salary_cols if c in latest_salary.columns]
                
                players_df = players_df.merge(latest_salary[cols_to_use], on='normalized_name', how='left')
                
                # Fill missing
                for col in ['salary', 'total_contract_value', 'years_remaining', 'cap_hit_percent', 'luxury_tax_risk', 'npv_salary']:
                    if col in players_df.columns:
                        players_df[col] = players_df[col].fillna(0)
                print(f"[OK] Merged salary information for {len(latest_salary)} players")
            else:
                print("[WARN] No salary data found. Skipping financial analysis.")
            
            print("[OK] Successfully enriched player data")
        except Exception as e:
            print(f"[WARN] Failed to enrich player data: {e}. Trade analysis features may be limited.")

        latest_season = players_df['season'].max()
        print(f"[OK] Loaded {len(players_df)} player-season records")
        print(f"[OK] Latest season detected: {latest_season}")
        model_loaded = True
    else:
        model_loaded = False
        injury_model = None

    # ── Load 2025-26 team overrides ───────────────────────────────────────────
    # The ML model uses 2024-25 stats for predictions, but the TEAM shown in
    # the UI must reflect where each player plays RIGHT NOW in 2025-26.
    TEAM_OVERRIDES_PATH = os.path.join(BASE_DIR, 'data', 'team_overrides_2025_26.json')
    team_overrides = {}
    try:
        with open(TEAM_OVERRIDES_PATH, 'r', encoding='utf-8') as f:
            overrides_data = json.load(f)
        team_overrides = overrides_data.get('overrides', {})
        print(f"[OK] Loaded {len(team_overrides)} team overrides for 2025-26 season")
    except FileNotFoundError:
        print(f"[WARN] No team overrides file found at {TEAM_OVERRIDES_PATH}")
    except Exception as e:
        print(f"[WARN] Error loading team overrides: {e}")

except Exception as e:
    print(f"[ERROR] Error loading model: {str(e)}  (app runs in demo mode)")
    ml_model = None
    shap_explainers = {}
    feature_names = []
    target_names = []
    model_metadata = {}
    players_df = None
    latest_season = None
    model_loaded = False
    team_overrides = {}


# ============================================================================
# Helper Functions
# ============================================================================

def get_player_photo_url(player_name):
    """Generate the GitHub raw URL for a player's headshot with exact First-Last formatting."""
    if not player_name:
        return None
    
    # 1. Strip and split name portions (handles "Anthony Davis " or "  Anthony Davis")
    parts = player_name.strip().split()
    
    # 2. Reconstruct as First-Last (skipping middle names or suffixes like Jr. for GitHub compatibility)
    if len(parts) >= 2:
        formatted_name = f"{parts[0].capitalize()}-{parts[1].capitalize()}"
    else:
        # Fallback for single-name players (rare) or malformed strings
        formatted_name = player_name.strip().replace(" ", "-").title()
        
    return f"https://raw.githubusercontent.com/GreenGuitar0/nba-players/main/player_images/{formatted_name}.jpg"


def get_latest_season_data():
    """
    Get data from the most recent season, deduplicated and with team
    assignments updated to the current 2025-26 season.

    Step 1: Dedup — players traded mid-season (2024-25) appear multiple
            times; keep only the last row (final team that season).
    Step 2: Override — apply team_overrides dict so that players who
            changed teams between 2024-25 and 2025-26 (via trades,
            free agency, draft) are shown on their CURRENT team.

    The ML feature vector is never modified — only the 'team' column
    used for display and roster grouping is updated.
    """
    if not model_loaded or players_df is None:
        return None

    season_df = players_df[players_df['season'] == latest_season].copy()

    # Step 1: Drop earlier trade stints — keep only the final team per player
    season_df = season_df.drop_duplicates(subset='player_name', keep='last')

    # Step 2: Apply 2025-26 team overrides
    if team_overrides:
        season_df['team'] = season_df.apply(
            lambda row: team_overrides.get(row['player_name'], row['team']),
            axis=1
        )

    return season_df


def _explain_predictions(player_row, model, explainers, feature_names, target_names, current_age=None):
    """
    Helper for multi-output predictions and SHAP explanations for a single player row.
    Returns: (predictions_dict, confidence_ranges, shap_results_list)
    """
    # 1. Feature Extraction
    feature_vals = np.array([player_row[f] for f in feature_names])
    feature_vector = feature_vals.reshape(1, -1)
    
    # 2. Predictions
    preds_raw = model.predict(feature_vector)[0]
    pred_dict = {target_names[i].replace('target_next_', ''): round(float(preds_raw[i]), 2) for i in range(len(target_names))}
    
    # 3. SHAP (Multi-target)
    shap_results = []
    for i, target in enumerate(target_names):
        target_key = target.replace('target_next_', '')
        explainer = explainers.get(target)
        if explainer:
            try:
                sv = explainer.shap_values(feature_vector)
                if sv.ndim == 1: sv = sv.reshape(1, -1)
                shap_row = sv[0]
                
                # Get top features for this target
                indices = np.argsort(np.abs(shap_row))[::-1][:3] 
                for idx in indices:
                    shap_results.append((feature_names[idx], float(shap_row[idx]), target_key, float(feature_vals[idx])))
            except Exception:
                continue
                
    # Sort by absolute SHAP across all targets to get "most defining" characteristics
    shap_results.sort(key=lambda x: abs(x[1]), reverse=True)
    
    return pred_dict, {}, shap_results


def calculate_current_age(dataset_age, dataset_season):
    """Calculate player's current age accounting for time since season."""
    current_year = datetime.now().year
    season_parts = dataset_season.split('-')
    season_start_year_str = season_parts[0]
    season_start_year = int(season_start_year_str)
    if len(season_start_year_str) == 2:
        season_start_year += 2000 if season_start_year < 50 else 1900
    years_passed = current_year - season_start_year
    return int(dataset_age) + years_passed


def explain_prediction(feature_vector, feature_vals, top_n=6, current_age=None):
    """
    Compute per-target SHAP explanations using one TreeExplainer per target.

    Returns a dict keyed by short target name (ppg, rpg, apg, mpg, ts_pct)
    with structured factors and plain-English reasons.
    """
    explanation = {}

    for target_full, explainer in shap_explainers.items():
        target_info = TARGET_DISPLAY_MAP.get(target_full, {})
        target_key = target_info.get('key', target_full)

        try:
            # shap_values returns array (n_samples, n_features); we pass 1 row
            sv = explainer.shap_values(feature_vector)
            # Ensure shape (1, n_features)
            if sv.ndim == 1:
                sv = sv.reshape(1, -1)
            shap_row = sv[0]           # shape: (n_features,)
            base_val = float(explainer.expected_value)

        except Exception as ex:
            print(f"SHAP error for {target_full}: {ex}")
            continue

        # Sort features by |shap| descending, take top_n
        indices = np.argsort(np.abs(shap_row))[::-1][:top_n]

        factors = []
        for idx in indices:
            fname = feature_names[idx]
            raw_val = float(feature_vals[idx])
            sval = float(shap_row[idx])
            # Pass current_age so the age reason shows the correct present-day age
            reason = _build_reason(fname, raw_val, sval, target_key,
                                   current_age=(current_age if fname == 'age' else None))
            factors.append({
                "feature":   fname,
                "label":     FEATURE_LABEL_MAP.get(fname, fname.replace('_', ' ').title()),
                "raw_value": round(raw_val, 4),
                "shap":      round(sval, 3),
                "direction": "positive" if sval >= 0 else "negative",
                "reason":    reason,
            })

        # Net SHAP sum (prediction = base_val + sum of all shap values)
        total_shap = float(np.sum(shap_row))

        explanation[target_key] = {
            "target_label":  target_info.get('label', target_key),
            "unit":          target_info.get('unit', ''),
            "base_value":    round(base_val, 2),
            "shap_sum":      round(total_shap, 3),
            "top_factors":   factors,
        }

    return explanation


# ============================================================================
# Routes
# ============================================================================

@app.route('/')
def home():
    return render_template('index.html')


@app.route('/api/health', methods=['GET'])
def health():
    return jsonify({
        'status': 'ok' if model_loaded else 'degraded',
        'model_loaded': model_loaded,
        'model_version': model_metadata.get('model_version', 'N/A'),
        'features': len(feature_names),
        'targets': len(target_names),
        'shap_targets': list(shap_explainers.keys()),
        'latest_season': latest_season,
        'player_records': len(players_df) if players_df is not None else 0
    })


@app.route('/api/teams', methods=['GET'])
def get_teams():
    if not model_loaded:
        return jsonify({'error': 'Model not loaded'}), 503

    season_data = get_latest_season_data()
    if season_data is None or len(season_data) == 0:
        return jsonify({'error': 'No data available'}), 500

    teams = sorted(season_data['team'].unique().tolist())

    team_map = {
        'ATL': 'Atlanta Hawks',         'BOS': 'Boston Celtics',
        'BKN': 'Brooklyn Nets',         'BRK': 'Brooklyn Nets',
        'CHA': 'Charlotte Hornets',     'CHO': 'Charlotte Hornets',
        'CHI': 'Chicago Bulls',         'CLE': 'Cleveland Cavaliers',
        'DAL': 'Dallas Mavericks',      'DEN': 'Denver Nuggets',
        'DET': 'Detroit Pistons',       'GSW': 'Golden State Warriors',
        'HOU': 'Houston Rockets',       'IND': 'Indiana Pacers',
        'LAC': 'Los Angeles Clippers',  'LAL': 'Los Angeles Lakers',
        'MEM': 'Memphis Grizzlies',     'MIA': 'Miami Heat',
        'MIL': 'Milwaukee Bucks',       'MIN': 'Minnesota Timberwolves',
        'NOP': 'New Orleans Pelicans',  'NYK': 'New York Knicks',
        'OKC': 'Oklahoma City Thunder', 'ORL': 'Orlando Magic',
        'PHI': 'Philadelphia 76ers',    'PHX': 'Phoenix Suns',
        'PHO': 'Phoenix Suns',          'POR': 'Portland Trail Blazers',
        'SAC': 'Sacramento Kings',      'SAS': 'San Antonio Spurs',
        'TOR': 'Toronto Raptors',       'UTA': 'Utah Jazz',
        'WAS': 'Washington Wizards',
    }

    formatted_teams = [{'code': t, 'name': team_map.get(t, t)} for t in teams]
    return jsonify({'teams': formatted_teams, 'count': len(teams), 'season': latest_season})


@app.route('/api/players', methods=['GET'])
def get_players():
    if not model_loaded:
        return jsonify({'error': 'Model not loaded'}), 503

    team = request.args.get('team')
    season_data = get_latest_season_data()
    if season_data is None or len(season_data) == 0:
        return jsonify({'error': 'No data available'}), 500

    if team:
        season_data = season_data[season_data['team'] == team]

    players_list = season_data.sort_values('points_per_game', ascending=False)[
        ['player_name', 'team', 'age', 'points_per_game', 'salary']
    ].to_dict('records')

    # Add photo_url
    for p in players_list:
        p['photo_url'] = get_player_photo_url(p['player_name'])

    return jsonify({
        'players': players_list,
        'team': team,
        'count': len(players_list),
        'season': latest_season
    })


@app.route('/api/search', methods=['GET'])
def search_players():
    if not model_loaded:
        return jsonify({'error': 'Model not loaded'}), 503

    query = request.args.get('q', '').lower()
    if not query:
        return jsonify({'results': [], 'query': query})

    season_data = get_latest_season_data()
    if season_data is None:
        return jsonify({'results': [], 'query': query})

    matches = season_data[
        season_data['player_name'].str.lower().str.contains(query, na=False)
    ][['player_name', 'team', 'points_per_game']].to_dict('records')

    # Add photo_url
    for m in matches:
        m['photo_url'] = get_player_photo_url(m['player_name'])

    return jsonify({'results': matches[:10], 'query': query})


@app.route('/api/predict', methods=['POST'])
def predict():
    """
    Predict player performance for next season.
    Returns predictions + full per-target SHAP explanations.
    """
    if not model_loaded:
        return jsonify({'error': 'Model not loaded'}), 503

    data = request.get_json()
    player_name = data.get('player_name', '').strip()

    if not player_name:
        return jsonify({'error': 'Player name required'}), 400

    # Find player — use the deduplicated latest-season view so the team shown
    # matches where the player currently plays (handles mid-season trades).
    latest_season_df = get_latest_season_data()
    current_row = None
    if latest_season_df is not None:
        match = latest_season_df[
            latest_season_df['player_name'].str.lower() == player_name.lower()
        ]
        if len(match) > 0:
            current_row = match.iloc[0]

    # Fall back to any season if not found in latest (e.g. retired player lookup)
    if current_row is None:
        all_seasons = players_df[
            players_df['player_name'].str.lower() == player_name.lower()
        ].sort_values('season', ascending=False)
        if len(all_seasons) == 0:
            return jsonify({'error': f'Player "{player_name}" not found'}), 404
        current_row = all_seasons.iloc[0].copy()
        # Apply team override if available
        pname = current_row.get('player_name', '')
        if pname in team_overrides:
            current_row['team'] = team_overrides[pname]

    latest = current_row
    current_age = calculate_current_age(latest['age'], latest['season'])

    # Build display season string
    season_parts = latest['season'].split('-')
    if len(season_parts[0]) == 2:
        start_yr = 2000 + int(season_parts[0]) if int(season_parts[0]) < 50 else 1900 + int(season_parts[0])
    else:
        start_yr = int(season_parts[0])
    display_season = f"{start_yr}-{start_yr + 1}"

    # Check for missing features
    missing_features = [f for f in feature_names if f not in latest.index or pd.isna(latest[f])]
    if missing_features:
        return jsonify({
            'error': f'Insufficient data for {player_name}',
            'missing_features': missing_features[:5]
        }), 400

    try:
        feature_vals = np.array([latest[f] for f in feature_names])
        feature_vector = feature_vals.reshape(1, -1)

        # ── Predictions ───────────────────────────────────────────────────────
        predictions_raw = ml_model.predict(feature_vector)[0]
        pred_dict = {
            target_names[i].replace('target_next_', ''): round(float(predictions_raw[i]), 2)
            for i in range(len(target_names))
        }

        # ── Confidence ranges (MAE from metadata) ─────────────────────────────
        confidence_ranges = {}
        for i, target in enumerate(target_names):
            target_key = target.replace('target_next_', '')
            mae = model_metadata.get('performance', {}).get(target, {}).get('test_mae', 2.0)
            r2  = model_metadata.get('performance', {}).get(target, {}).get('test_r2',  0.5)
            pv  = pred_dict[target_key]
            confidence_ranges[target_key] = {
                'lower': round(max(0, pv - mae), 2),
                'upper': round(pv + mae, 2),
                'mae':   round(mae, 3),
                'r2':    round(r2, 3),
                'confidence_label': (
                    'high'   if r2 >= 0.75 else
                    'medium' if r2 >= 0.50 else
                    'low'
                )
            }

        # ── Full SHAP explanation (per target) ────────────────────────────────
        shap_explanation = explain_prediction(feature_vector, feature_vals,
                                              top_n=6, current_age=current_age)

        # ── Build response ─────────────────────────────────────────────────────
        response = {
            'player_name':   latest['player_name'],
            'team':          latest['team'],
            'current_season': latest['season'],
            'last_season_year': display_season,
            'current_age':   current_age,
            'current_stats': {
                'ppg': round(float(latest['points_per_game']), 1),
                'rpg': round(float(latest['rebounds_per_game']), 1),
                'apg': round(float(latest['assists_per_game']), 1),
                'age': current_age,
            },
            'predictions':        pred_dict,
            'confidence_ranges':  confidence_ranges,
            'shap_explanation':   shap_explanation,
            'photo_url':          get_player_photo_url(latest['player_name']),
            'model_version':      model_metadata.get('model_version', 'v2'),
            'injury_risk_prob':   float(latest.get('injury_risk_prob', 0.05)),
            'injury_risk_category': str(latest.get('injury_risk_category', 'Low')),
            'medical_score':      float(latest.get('medical_score', 0)),
            'medical_grade':      str(latest.get('medical_grade', 'N/A')),
        }

        return jsonify(response)

    except Exception as e:
        return jsonify({'error': f'Prediction failed: {str(e)}'}), 500


@app.route('/api/trade/evaluate', methods=['POST'])
def evaluate_trade():
    """
    Evaluate a trade between two teams.
    Expected JSON: 
    {
      "team_a": "LAL",
      "team_b": "GSW",
      "sent_a": ["LeBron James"],
      "sent_b": ["Stephen Curry"]
    }
    """
    if not model_loaded:
        return jsonify({'error': 'Model not loaded'}), 503

    data = request.get_json()
    team_a_code = data.get('team_a')
    team_b_code = data.get('team_b')
    sent_a_names = data.get('sent_a', [])
    sent_b_names = data.get('sent_b', [])

    if not team_a_code or not team_b_code:
        return jsonify({'error': 'Team codes are required'}), 400

    # Get full rosters for both teams (from latest data)
    season_data = get_latest_season_data()
    if season_data is None:
        return jsonify({'error': 'No season data available'}), 500

    # Rosters for simulation
    roster_a = season_data[season_data['team'] == team_a_code].copy()
    roster_b = season_data[season_data['team'] == team_b_code].copy()

    if roster_a.empty or roster_b.empty:
        return jsonify({'error': 'One or both team rosters not found'}), 404

    # Normalize outgoing names for matching
    sent_a_norm = [normalize_name(n) for n in sent_a_names]
    sent_b_norm = [normalize_name(n) for n in sent_b_names]

    # Split players
    traded_from_a = roster_a[roster_a['normalized_name'].isin(sent_a_norm)]
    traded_from_b = roster_b[roster_b['normalized_name'].isin(sent_b_norm)]

    # 1b. Financial Impact (Robust calculation)
    def calculate_sum_safely(df, col):
        if df.empty or col not in df.columns:
            return 0.0
        return float(df[col].sum())

    def calculate_mean_safely(df, col):
        if df.empty or col not in df.columns:
            return 0.0
        return float(df[col].mean())

    salary_out_a = calculate_sum_safely(traded_from_a, 'salary')
    salary_in_a = calculate_sum_safely(traded_from_b, 'salary')
    salary_delta_a = salary_in_a - salary_out_a

    tax_risk_out_a = calculate_mean_safely(traded_from_a, 'luxury_tax_risk')
    tax_risk_in_a = calculate_mean_safely(traded_from_b, 'luxury_tax_risk')
    tax_risk_ch_a = tax_risk_in_a - tax_risk_out_a

    # New rosters
    post_a = pd.concat([roster_a[~roster_a['normalized_name'].isin(sent_a_norm)], traded_from_b])
    post_b = pd.concat([roster_b[~roster_b['normalized_name'].isin(sent_b_norm)], traded_from_a])

    # 1. Health Analysis
    pre_health_a = calculate_roster_health_score(roster_a)
    post_health_a = calculate_roster_health_score(post_a)
    pre_health_b = calculate_roster_health_score(roster_b)
    post_health_b = calculate_roster_health_score(post_b)

    # 2. Win Simulation (1000 sims for notebook-level confidence)
    print(f"Simulating trade impact for {team_a_code} and {team_b_code}...")
    pre_wins_a = run_injury_adjusted_simulation(roster_a, n_sims=1000)
    post_wins_a = run_injury_adjusted_simulation(post_a, n_sims=1000)
    pre_wins_b = run_injury_adjusted_simulation(roster_b, n_sims=1000)
    post_wins_b = run_injury_adjusted_simulation(post_b, n_sims=1000)

    # 3. Metrics Calculation
    delta_a = float(post_wins_a.mean() - pre_wins_a.mean())
    delta_b = float(post_wins_b.mean() - pre_wins_b.mean())

    # Playoffs & Fit
    from services.trade_analyzer import wins_to_playoff_prob, calculate_fit_penalty
    playoff_ch_a = wins_to_playoff_prob(post_wins_a.mean()) - wins_to_playoff_prob(pre_wins_a.mean())
    playoff_ch_b = wins_to_playoff_prob(post_wins_b.mean()) - wins_to_playoff_prob(pre_wins_b.mean())
    
    pre_penalty_a, pre_reasons_a = calculate_fit_penalty(roster_a)
    post_penalty_a, post_reasons_a = calculate_fit_penalty(post_a)
    pre_penalty_b, pre_reasons_b = calculate_fit_penalty(roster_b)
    post_penalty_b, post_reasons_b = calculate_fit_penalty(post_b)

    new_reasons_a = [r for r in post_reasons_a if r not in pre_reasons_a]
    new_reasons_b = [r for r in post_reasons_b if r not in pre_reasons_b]

    # Risk/Medical changes
    def get_mean_safely(df, col, default=0.0):
        if df.empty: return default
        if col in df.columns: return float(df[col].mean())
        if col == 'injury_risk_prob': return 0.05
        if col == 'medical_score': return 50.0
        return default

    mean_inj_b = get_mean_safely(traded_from_b, 'injury_risk_prob')
    mean_inj_a = get_mean_safely(traded_from_a, 'injury_risk_prob')
    inj_ch_a = mean_inj_b - mean_inj_a if (not traded_from_a.empty and not traded_from_b.empty) else 0.0
    
    mean_med_b = get_mean_safely(traded_from_b, 'medical_score')
    mean_med_a = get_mean_safely(traded_from_a, 'medical_score')
    med_ch_a = mean_med_b - mean_med_a if (not traded_from_a.empty and not traded_from_b.empty) else 0.0
    
    # Calculate scores
    score_a = calc_trade_score(delta_a, float(playoff_ch_a), inj_ch_a, 
                              post_health_a['roster_health_score'] - pre_health_a['roster_health_score'],
                              med_ch_a)
    score_b = calc_trade_score(delta_b, float(playoff_ch_b), -inj_ch_a, 
                              post_health_b['roster_health_score'] - pre_health_b['roster_health_score'],
                              -med_ch_a)

    # SHAP explanations for the biggest incoming pieces
    def get_top_player_shap(traded_df):
        if traded_df.empty: return None
        try:
            ppg_col = 'target_next_ppg' if 'target_next_ppg' in traded_df.columns else 'points_per_game'
            top_player = traded_df.loc[traded_df[ppg_col].idxmax()]
            
            c_age = calculate_current_age(top_player['age'], latest_season) if 'age' in top_player else 25
            _, _, shap_exp = _explain_predictions(top_player, ml_model, shap_explainers, feature_names, target_names, current_age=c_age)
            
            reasons = [_build_reason(f, v, s, t, c_age) for f, s, t, v in shap_exp[:2]]
            return {"name": top_player['player_name'], "reasons": reasons}
        except Exception as e:
            return None

    shap_incoming_a = get_top_player_shap(traded_from_b)
    shap_incoming_b = get_top_player_shap(traded_from_a)

    # Safe select columns for traded players
    def safe_cols(df):
        cols = ['player_name', 'points_per_game', 'salary', 'total_contract_value', 'luxury_tax_risk']
        # Filter for existing columns
        active_cols = [c for c in cols if c in df.columns]
        if 'injury_risk_prob' in df.columns: active_cols.append('injury_risk_prob')
        if 'medical_score' in df.columns: active_cols.append('medical_score')
        if 'medical_grade' in df.columns: active_cols.append('medical_grade')
        
        result_dicts = df[active_cols].to_dict('records')
        for r in result_dicts:
            r['photo_url'] = get_player_photo_url(r['player_name'])
        return result_dicts

    # Result response
    result = {
        "traded_players": {
            "from_a": safe_cols(traded_from_a),
            "from_b": safe_cols(traded_from_b)
        },
        "team_a": {
            "code": team_a_code,
            "pre_wins": round(float(pre_wins_a.mean()), 2),
            "post_wins": round(float(post_wins_a.mean()), 2),
            "win_change": round(delta_a, 2),
            "win_ci": [round(float(np.percentile(post_wins_a - pre_wins_a, 5)), 2), 
                       round(float(np.percentile(post_wins_a - pre_wins_a, 95)), 2)],
            "playoff_change": round(float(playoff_ch_a) * 100, 1),
            "health_score_pre": pre_health_a['roster_health_score'],
            "health_score_post": post_health_a['roster_health_score'],
            "health_grade_post": post_health_a['health_grade'],
            "health_change": round(post_health_a['roster_health_score'] - pre_health_a['roster_health_score'], 1),
            "injury_risk_change": round(inj_ch_a * 100, 2),
            "medical_change": round(med_ch_a, 1),
            "salary_change": round(salary_delta_a, 1),
            "salary_out": round(salary_out_a, 1),
            "salary_in": round(salary_in_a, 1),
            "tax_risk_change": round(tax_risk_ch_a, 2),
            "grade": get_trade_rating(score_a),
            "score": round(score_a, 1)
        },
        "team_b": {
            "code": team_b_code,
            "pre_wins": round(float(pre_wins_b.mean()), 2),
            "post_wins": round(float(post_wins_b.mean()), 2),
            "win_change": round(delta_b, 2),
            "win_ci": [round(float(np.percentile(post_wins_b - pre_wins_b, 5)), 2), 
                       round(float(np.percentile(post_wins_b - pre_wins_b, 95)), 2)],
            "playoff_change": round(float(playoff_ch_b) * 100, 1),
            "health_score_pre": pre_health_b['roster_health_score'],
            "health_score_post": post_health_b['roster_health_score'],
            "health_grade_post": post_health_b['health_grade'],
            "health_change": round(post_health_b['roster_health_score'] - pre_health_b['roster_health_score'], 1),
            "injury_risk_change": round(-inj_ch_a * 100, 2),
            "medical_change": round(-med_ch_a, 1),
            "salary_change": round(-salary_delta_a, 1),
            "salary_out": round(salary_in_a, 1),
            "salary_in": round(salary_out_a, 1),
            "tax_risk_change": round(-tax_risk_ch_a, 2),
            "grade": get_trade_rating(score_b),
            "score": round(score_b, 1)
        },
        "assessment": {
            "fairness": round(100 - abs(score_a - score_b), 1),
            "winner": team_a_code if delta_a > delta_b else team_b_code,
            "win_margin": round(abs(delta_a - delta_b), 1),
            "explanations": {
                "fit_penalty_a_change": round(post_penalty_a - pre_penalty_a, 3),
                "fit_penalty_b_change": round(post_penalty_b - pre_penalty_b, 3),
                "fit_reasons_a": new_reasons_a,
                "fit_reasons_b": new_reasons_b,
                "shap_incoming_a": shap_incoming_a,
                "shap_incoming_b": shap_incoming_b
            }
        }
    }

    return jsonify(result)

@app.route('/api/roster/<team_code>', methods=['GET'])
def get_team_roster(team_code):
    if not model_loaded:
        return jsonify({'error': 'Model not loaded'}), 503

    season_data = get_latest_season_data()
    if season_data is None:
        return jsonify({'error': 'No data available'}), 500

    roster = season_data[season_data['team'] == team_code].sort_values('points_per_game', ascending=False)
    
    players = roster[[
        'player_name', 'position', 'age', 'points_per_game', 
        'rebounds_per_game', 'assists_per_game', 'injury_risk_category', 'medical_grade', 'salary'
    ]].to_dict('records')
    
    # Calculate team-level metrics
    health = calculate_roster_health_score(roster)
    
    return jsonify({
        "team": team_code,
        "players": players,
        "roster_health": health
    })
# ============================================================================
# Startup Banner
# ============================================================================
if __name__ == '__main__':
    print("\n" + "="*80)
    print("NBA TRADE ANALYZER — Flask Backend with Full SHAP Explainability")
    print("="*80)
    print(f"Model loaded:      {model_loaded}")
    print(f"Features:          {len(feature_names)}")
    print(f"Targets:           {len(target_names)}")
    print(f"SHAP explainers:   {len(shap_explainers)}")
    if latest_season:
        print(f"Latest season:     {latest_season}")
        print(f"Player records:    {len(players_df)}")
    print("="*80)
    print(f"\nStarting server at http://127.0.0.1:{FLASK_PORT}")
    print("Press Ctrl+C to stop")
    print("="*80 + "\n")

    debug_mode = os.getenv('FLASK_DEBUG', 'True').lower() == 'true'
    app.run(host='127.0.0.1', port=FLASK_PORT, debug=debug_mode)
