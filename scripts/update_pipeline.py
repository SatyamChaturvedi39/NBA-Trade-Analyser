"""
NBA Trade Analyzer - Automated Data Pipeline
Fetches current season stats from NBA API, computes rolling ML features (lags/trends),
and updates the MongoDB database. Can be scheduled via Cron.
"""

import os
import sys
import argparse
import time
import numpy as np
import pandas as pd
from datetime import datetime
from pymongo import MongoClient
from pymongo.errors import ConnectionFailure
from dotenv import load_dotenv

try:
    from nba_api.stats.endpoints import leaguedashplayerstats
except ImportError:
    print("ERROR: nba_api not installed. Run: pip install nba_api")
    sys.exit(1)

# Load environment variables
load_dotenv()
MONGODB_URI = os.getenv('MONGODB_URI')
DATABASE_NAME = os.getenv('DATABASE_NAME', 'nba_db')
COLLECTION_NAME = os.getenv('COLLECTION_NAME', 'players')

# ==============================================================================
# Helper Math Functions for ML Feature Engineering
# ==============================================================================

def safe_float(val, default=0.0):
    try:
        if pd.isna(val): return default
        return float(val)
    except:
        return default

def calculate_slope(y_values):
    """Calculate the linear trend (slope) over an array of historical values."""
    if len(y_values) < 2:
        return 0.0
    x = np.arange(len(y_values))
    try:
        return np.polyfit(x, y_values, 1)[0]
    except:
        return 0.0

def compute_ml_features(historical_stats, current_stats):
    """
    Appends the current season to the history and recalibrates all 54 lag, trend, 
    and career features required by the XGBoost multi-output model.
    """
    # Create the new season object mapping MBA API columns to our DB schema
    new_season = {
        'season': current_stats.get('season_str', 'Unknown'),
        'player_name': current_stats['PLAYER_NAME'],
        'team': current_stats['TEAM_ABBREVIATION'],
        'age': safe_float(current_stats['AGE']),
        'games_played': safe_float(current_stats['GP']),
        'minutes_per_game': safe_float(current_stats['MIN']),
        'points_per_game': safe_float(current_stats['PTS']),
        'rebounds_per_game': safe_float(current_stats['REB']),
        'assists_per_game': safe_float(current_stats['AST']),
        'steals_per_game': safe_float(current_stats['STL']),
        'blocks_per_game': safe_float(current_stats['BLK']),
        'turnovers_per_game': safe_float(current_stats['TOV']),
        'field_goal_pct': safe_float(current_stats['FG_PCT']),
        'free_throw_pct': safe_float(current_stats['FT_PCT']),
        # True Shooting Approximation if not provided natively
        'true_shooting_pct': safe_float(current_stats['PTS']) / (2 * (safe_float(current_stats['FGA']) + 0.44 * safe_float(current_stats['FTA']))) if safe_float(current_stats['FGA']) > 0 else 0
    }
    
    # Calculate advanced efficiency (Points Per Minute)
    new_season['points_per_minute'] = new_season['points_per_game'] / new_season['minutes_per_game'] if new_season['minutes_per_game'] > 0 else 0
    
    # Append to history
    updated_history = historical_stats.copy()
    updated_history.append(new_season)
    
    # Extract arrays for rolling calculations
    ppg_arr = [safe_float(s.get('points_per_game', 0)) for s in updated_history]
    mpg_arr = [safe_float(s.get('minutes_per_game', 0)) for s in updated_history]
    rpg_arr = [safe_float(s.get('rebounds_per_game', 0)) for s in updated_history]
    apg_arr = [safe_float(s.get('assists_per_game', 0)) for s in updated_history]
    ts_arr  = [safe_float(s.get('true_shooting_pct', 0)) for s in updated_history]
    fg_arr  = [safe_float(s.get('field_goal_pct', 0)) for s in updated_history]
    games_arr = [safe_float(s.get('games_played', 0)) for s in updated_history]
    
    # Build LAG features (Looking backwards from the new stat line)
    # lag1 = stats from 1 season ago, lag2 = 2 seasons ago, etc.
    # Note: If history doesn't go back that far, default to 0
    def get_lag(arr, steps_back):
        idx = len(arr) - 1 - steps_back
        return arr[idx] if idx >= 0 else 0.0

    new_season['ppg_lag1'] = get_lag(ppg_arr, 1)
    new_season['ppg_lag2'] = get_lag(ppg_arr, 2)
    new_season['ppg_lag3'] = get_lag(ppg_arr, 3)
    new_season['ppg_lag4'] = get_lag(ppg_arr, 4)
    new_season['ppg_lag5'] = get_lag(ppg_arr, 5)

    new_season['mpg_lag1'] = get_lag(mpg_arr, 1)
    new_season['mpg_lag2'] = get_lag(mpg_arr, 2)
    new_season['mpg_lag3'] = get_lag(mpg_arr, 3)
    new_season['mpg_lag4'] = get_lag(mpg_arr, 4)
    new_season['mpg_lag5'] = get_lag(mpg_arr, 5)
    
    new_season['rpg_lag1'] = get_lag(rpg_arr, 1)
    new_season['rpg_lag2'] = get_lag(rpg_arr, 2)
    new_season['rpg_lag3'] = get_lag(rpg_arr, 3)

    new_season['apg_lag1'] = get_lag(apg_arr, 1)
    new_season['apg_lag2'] = get_lag(apg_arr, 2)
    new_season['apg_lag3'] = get_lag(apg_arr, 3)
    
    new_season['ts_pct_lag1'] = get_lag(ts_arr, 1)
    new_season['ts_pct_lag2'] = get_lag(ts_arr, 2)
    new_season['ts_pct_lag3'] = get_lag(ts_arr, 3)

    new_season['fg_pct_lag1'] = get_lag(fg_arr, 1)
    new_season['fg_pct_lag2'] = get_lag(fg_arr, 2)

    new_season['games_lag1'] = get_lag(games_arr, 1)
    new_season['games_lag2'] = get_lag(games_arr, 2)
    new_season['games_lag3'] = get_lag(games_arr, 3)
    
    # Single level peripheral lags
    new_season['spg_lag1'] = get_lag([safe_float(s.get('steals_per_game', 0)) for s in updated_history], 1)
    new_season['bpg_lag1'] = get_lag([safe_float(s.get('blocks_per_game', 0)) for s in updated_history], 1)

    # Build TREND features (Slopes over the last N years)
    new_season['ppg_trend_2yr'] = calculate_slope(ppg_arr[-2:] if len(ppg_arr) >= 2 else ppg_arr)
    new_season['ppg_trend_3yr'] = calculate_slope(ppg_arr[-3:] if len(ppg_arr) >= 3 else ppg_arr)
    new_season['ppg_trend_4yr'] = calculate_slope(ppg_arr[-4:] if len(ppg_arr) >= 4 else ppg_arr)
    new_season['mpg_trend_2yr'] = calculate_slope(mpg_arr[-2:] if len(mpg_arr) >= 2 else mpg_arr)

    # Build CAREER ARCHITECTURE features
    new_season['seasons_in_dataset'] = len(updated_history)
    peak_ppg = max(ppg_arr) if ppg_arr else 0
    new_season['peak_ppg'] = peak_ppg
    
    # Find years since peak
    peak_idx = ppg_arr.index(peak_ppg) if ppg_arr else 0
    new_season['years_since_peak_ppg'] = len(ppg_arr) - 1 - peak_idx

    new_season['career_ppg_avg'] = np.mean(ppg_arr) if ppg_arr else 0
    new_season['career_ppg_std'] = np.std(ppg_arr) if ppg_arr else 0
    new_season['career_games_avg'] = np.mean(games_arr) if games_arr else 0
    new_season['career_mpg_avg'] = np.mean(mpg_arr) if mpg_arr else 0
    
    mean_ppg = np.mean(ppg_arr)
    new_season['ppg_coefficient_variation'] = (np.std(ppg_arr) / mean_ppg) if mean_ppg > 0 else 0

    return updated_history

# ==============================================================================
# Main Pipeline Implementation
# ==============================================================================

def update_pipeline(target_season):
    print("=" * 80)
    print(f"AUTOMATED DATA PIPELINE - Fetching {target_season} Season Data")
    print("=" * 80)
    
    # 1. Connect to MongoDB
    print("\n[1/4] Connecting to Database...")
    try:
        client = MongoClient(MONGODB_URI, serverSelectionTimeoutMS=5000)
        client.server_info()
        db = client[DATABASE_NAME]
        collection = db[COLLECTION_NAME]
        print("✓ Connected to MongoDB.")
    except Exception as e:
        print(f"✗ MongoDB Connection failed: {e}")
        sys.exit(1)

    # 2. Fetch Live NBA Stats via nba_api
    print("\n[2/4] Fetching Live NBA API Data...")
    try:
        # NBA API format is "2024-25"
        dashboard = leaguedashplayerstats.LeagueDashPlayerStats(
            season=target_season, 
            per_mode_detailed='PerGame'
        )
        live_df = dashboard.get_data_frames()[0]
        live_df['season_str'] = target_season
        print(f"✓ Retrieved active stats for {len(live_df)} players.")
    except Exception as e:
        print(f"✗ Failed to fetch data from NBA API: {e}")
        print("Note: If the internet hangs, the NBA API might be rate-limiting you.")
        client.close()
        sys.exit(1)

    # 3. Process and Update Records
    print("\n[3/4] Updating Player Histories and Calculating ML Features...")
    updated_count = 0
    new_count = 0

    for idx, row in live_df.iterrows():
        player_name = row['PLAYER_NAME']
        search_name = player_name.lower()
        
        # Look for existing player in DB
        doc = collection.find_one({"search_name": search_name})
        
        if doc:
            # Player exists. Check if this season is already in the array
            stats_array = doc.get('stats', [])
            if len(stats_array) > 0 and stats_array[-1].get('season') == target_season:
                # Stats for this season already exist, we should overwrite the last item
                stats_array.pop()
            
            # Compute new historical array with XGBoost features added
            new_stats_array = compute_ml_features(stats_array, row)
            
            # Update Document
            collection.update_one(
                {"_id": doc["_id"]},
                {
                    "$set": {
                        "team": row['TEAM_ABBREVIATION'],
                        "age": safe_float(row['AGE']),
                        "seasons_count": len(new_stats_array),
                        "stats": new_stats_array,
                        "last_updated": datetime.utcnow()
                    }
                }
            )
            updated_count += 1
        else:
            # Entirely new player (e.g., Rookie). Bootstrapped with 0 lag history.
            new_stats_array = compute_ml_features([], row)
            new_doc = {
                "player_name": player_name,
                "team": row['TEAM_ABBREVIATION'],
                "position": "N/A",  # Not provided directly via LeagueDash, would need separate call
                "age": safe_float(row['AGE']),
                "seasons_count": len(new_stats_array),
                "stats": new_stats_array,
                "search_name": search_name,
                "last_updated": datetime.utcnow()
            }
            collection.insert_one(new_doc)
            new_count += 1
            
        # Log progress every 100 players
        if (idx + 1) % 100 == 0:
            print(f"  ... processed {idx + 1}/{len(live_df)} players")
            
    print(f"\n✓ Database operation complete.")
    print(f"  - Updated {updated_count} existing ML player profiles")
    print(f"  - Created {new_count} new ML player profiles")

    # 4. Cleanup
    print("\n[4/4] Pipeline Execution Finished Successfully.")
    client.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="NBA Trade Analyzer Data Pipeline")
    parser.add_argument('--season', type=str, default='2024-25', help='Season string (e.g., 2024-25)')
    args = parser.parse_args()
    
    update_pipeline(args.season)
