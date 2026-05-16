import numpy as np
import pandas as pd
import unicodedata
import hashlib

def normalize_name(name):
    """Normalize player names for matching."""
    if not isinstance(name, str):
        return ""
    name = unicodedata.normalize('NFKD', name).encode('ascii', 'ignore').decode('ascii')
    return name.lower().strip()

def calculate_player_medical_score(player_data):
    """
    Calculate comprehensive medical score for a player (0-100).
    
    Components:
    - Injury History (40%): Based on games missed
    - Games Availability (30%): Consistency over 2 seasons
    - Injury Severity (20%): Type and count of injuries
    - Recovery Pattern (10%): Trend analysis
    """
    # Component 1: Injury History (40%)
    games_missed = player_data.get('games_missed', 0)
    games_missed_last = player_data.get('games_missed_last_season', 0)
    
    # Handle NaN or None
    if pd.isna(games_missed): games_missed = 0
    if pd.isna(games_missed_last): games_missed_last = 0
    
    injury_history_score = 100 * (1 - (games_missed / 82))
    injury_history_score = max(0, min(100, injury_history_score))
    
    # Component 2: Availability (30%)
    avg_games_missed = (games_missed + games_missed_last) / 2
    availability_score = 100 * (1 - (avg_games_missed / 82))
    availability_score = max(0, min(100, availability_score))
    
    # Component 3: Severity (20%)
    minor = player_data.get('minor_count', 0)
    moderate = player_data.get('moderate_count', 0)
    severe = player_data.get('severe_count', 0)
    has_severe = player_data.get('has_severe_injury', 0)
    
    if pd.isna(minor): minor = 0
    if pd.isna(moderate): moderate = 0
    if pd.isna(severe): severe = 0
    if pd.isna(has_severe): has_severe = 0
    
    severity_index = minor + (moderate * 3) + (severe * 5)
    severity_score = 100 * np.exp(-severity_index / 10)
    severity_score = max(0, min(100, severity_score))
    
    if has_severe:
        severity_score *= 0.7
    
    # Component 4: Recovery Pattern (10%)
    if games_missed_last > 0:
        recovery_ratio = games_missed / max(games_missed_last, 1)
        if recovery_ratio < 0.5:
            recovery_score = 100
        elif recovery_ratio < 1.0:
            recovery_score = 70
        elif recovery_ratio < 1.5:
            recovery_score = 40
        else:
            recovery_score = 20
    else:
        recovery_score = 100 if games_missed == 0 else 80
    
    # Weighted medical score
    medical_score = (
        0.40 * injury_history_score +
        0.30 * availability_score +
        0.20 * severity_score +
        0.10 * recovery_score
    )
    
    # Grading
    if medical_score >= 85:
        grade, risk = "EXCELLENT", "Very Low"
    elif medical_score >= 70:
        grade, risk = "GOOD", "Low"
    elif medical_score >= 55:
        grade, risk = "FAIR", "Moderate"
    elif medical_score >= 40:
        grade, risk = "POOR", "High"
    else:
        grade, risk = "CRITICAL", "Very High"
    
    return {
        'medical_score': round(medical_score, 1),
        'medical_grade': grade,
        'risk_level': risk,
        'injury_history_score': round(injury_history_score, 1),
        'availability_score': round(availability_score, 1),
        'severity_score': round(severity_score, 1),
        'recovery_score': round(recovery_score, 1),
        'games_missed_current': int(games_missed),
        'games_missed_last': int(games_missed_last),
        'severity_index': round(severity_index, 1),
        'has_severe_history': bool(has_severe)
    }

def calculate_roster_health_score(roster):
    """
    Calculate team roster health/fit score (0-100).
    """
    if roster.empty:
        return {
            'roster_health_score': 0,
            'health_grade': "POOR",
            'avg_medical': 0,
            'depth_health': 0,
            'risk_distribution': 0,
            'games_available': 0
        }
        
    # Get medical scores
    medical_scores = []
    for _, player in roster.iterrows():
        med = calculate_player_medical_score(player)
        medical_scores.append(med['medical_score'])
    
    # Component 1: Average Medical (40%)
    avg_medical = np.mean(medical_scores) if medical_scores else 50
    
    # Component 2: Depth Health (30%): Top 10 players
    top_10 = sorted(medical_scores, reverse=True)[:10]
    depth_health = np.mean(top_10) if top_10 else 50
    
    # Component 3: Risk Distribution (20%): Balance
    # Ensure injury_risk_prob exists
    if 'injury_risk_prob' in roster.columns:
        injury_risks = roster['injury_risk_prob'].fillna(0.05).values
        risk_std = np.std(injury_risks)
        risk_dist_score = 100 * np.exp(-risk_std * 2)
        risk_dist_score = max(0, min(100, risk_dist_score))
        
        # Risk categorization for summary
        high_risk = sum(injury_risks >= 0.7)
        moderate_risk = sum((injury_risks >= 0.5) & (injury_risks < 0.7))
        low_risk = sum(injury_risks < 0.5)
    else:
        risk_dist_score = 100
        high_risk, moderate_risk, low_risk = 0, 0, len(roster)
    
    # Component 4: Games Available (10%)
    expected_games = []
    for _, player in roster.iterrows():
        injury_prob = player.get('injury_risk_prob', 0.05)
        if pd.isna(injury_prob): injury_prob = 0.05
        expected = 82 * (1 - injury_prob * 0.5)
        expected_games.append(expected)
    
    avg_expected = np.mean(expected_games)
    games_avail_score = (avg_expected / 82) * 100
    
    # Weighted score
    health_score = (
        0.40 * avg_medical +
        0.30 * depth_health +
        0.20 * risk_dist_score +
        0.10 * games_avail_score
    )
    
    # Grade
    if health_score >= 80:
        grade = "EXCELLENT"
    elif health_score >= 65:
        grade = "GOOD"
    elif health_score >= 50:
        grade = "AVERAGE"
    elif health_score >= 35:
        grade = "BELOW AVERAGE"
    else:
        grade = "POOR"
    
    return {
        'roster_health_score': round(health_score, 1),
        'health_grade': grade,
        'avg_medical': round(avg_medical, 1),
        'depth_health': round(depth_health, 1),
        'risk_distribution': round(risk_dist_score, 1),
        'games_available': round(games_avail_score, 1),
        'high_risk_players': int(high_risk),
        'moderate_risk_players': int(moderate_risk),
        'low_risk_players': int(low_risk),
        'expected_games_per_player': round(avg_expected, 1)
    }

def calculate_fit_penalty(roster):
    """Position-based roster fit penalty with explicit reasoning."""
    penalty = 0.0
    reasons = []
    if 'position' not in roster.columns:
        return penalty, reasons
        
    pos_counts = roster['position'].value_counts()
    
    if pos_counts.get('PG', 0) > 3: 
        penalty += 0.05
        reasons.append("Too many Point Guards (4+ restricts rotation depth)")
    if pos_counts.get('C', 0) > 3: 
        penalty += 0.03
        reasons.append("Logjam at Center (4+ limits floor spacing and perimeter defense)")
    if pos_counts.get('PG', 0) < 1: 
        penalty += 0.08
        reasons.append("Severe lack of Point Guards (No primary playmaker)")
    
    # Check for scoring cannibalization (too many high volume scorers)
    if 'target_next_ppg' in roster.columns:
        top_scorers = roster.nlargest(3, 'target_next_ppg')['target_next_ppg'].sum()
        if top_scorers > 75:
            penalty += 0.08
            reasons.append("Scoring Cannibalization (Top 3 scorers demand too much usage, leading to diminishing returns)")
    
    return penalty, reasons

def calculate_team_score(roster, sampled_stats):
    """Team quality score based on top 8 rotation players."""
    # Ensure targets exist in sampled_stats
    ppg_col = 'target_next_ppg' if 'target_next_ppg' in sampled_stats.columns else 'points_per_game'
    ts_col = 'target_next_ts_pct' if 'target_next_ts_pct' in sampled_stats.columns else 'true_shooting_pct'
    
    top_8 = sampled_stats.nlargest(8, ppg_col)
    top_8_ppg = top_8[ppg_col].sum()
    efficiency = top_8[ts_col].mean() * 10
    
    penalty, _ = calculate_fit_penalty(roster)
    score = (top_8_ppg + efficiency) * (1 - penalty)
    
    return score

def predict_wins(score):
    """Convert team score to wins using regression-derived formula as recalibrated in M2_fix.ipynb."""
    wins = (score - 113.7) * 1.2 + 41
    return np.clip(wins, 5, 75)

def wins_to_playoff_prob(wins):
    """Convert projected wins to playoff probability."""
    if wins >= 42:
        return np.clip((wins - 42) * 0.05 + 0.5, 0.5, 1.0)
    else:
        return np.clip(0.5 - (42 - wins) * 0.05, 0.0, 0.5)

STD_DEVS = {
    'target_next_ppg': 3.0,
    'target_next_rpg': 1.2,
    'target_next_apg': 1.0,
    'target_next_mpg': 3.5,
    'target_next_ts_pct': 0.03
}

def run_injury_adjusted_simulation(roster, n_sims=1000):
    """
    Monte Carlo simulation with injury risk.
    """
    results = []
    
    # Pre-calculate common values
    injury_probs = roster.get('injury_risk_prob', pd.Series([0.05] * len(roster), index=roster.index)).fillna(0.05).values
    base_values = {col: roster[col].values for col in STD_DEVS if col in roster.columns}
    stds = {col: STD_DEVS[col] for col in STD_DEVS if col in roster.columns}
    # Create a deterministic seed based on roster composition
    roster_hash_str = "".join(sorted(roster['player_name'].astype(str).tolist()))
    seed_val = int(hashlib.md5(roster_hash_str.encode('utf-8')).hexdigest()[:8], 16)
    rng = np.random.default_rng(seed=seed_val)

    for _ in range(n_sims):
        sampled = roster.copy()
        
        # Sample from normal distributions for each player/stat
        for col, std in stds.items():
            sampled[col] = np.maximum(0, rng.normal(base_values[col], std))
        
        # Apply injury effects (stochastic)
        # 40-80% reduction in output if "injured" this season
        random_draws = rng.random(len(sampled))
        injured_mask = random_draws < injury_probs
        
        # Random severity for each injured player
        severities = 0.4 + (injury_probs * 0.4)
        
        for col in ['target_next_ppg', 'target_next_rpg', 'target_next_apg', 'target_next_mpg']:
            if col in sampled.columns:
                sampled.loc[injured_mask, col] *= (1 - severities[injured_mask])
        
        # Calculate wins for this iteration
        score = calculate_team_score(roster, sampled)
        wins = predict_wins(score)
        results.append(wins)
    
    return np.array(results)

def calc_trade_score(delta_w, playoff_ch, inj_ch, health_ch, med_ch):
    """
    Composite score for trade evaluation (0-100).
    """
    win_score = np.clip((delta_w / 5.0) * 50 + 50, 0, 100)
    playoff_score = np.clip(playoff_ch * 100 + 50, 0, 100)
    injury_score = np.clip(-inj_ch * 100 + 50, 0, 100)
    health_score = np.clip(health_ch * 2 + 50, 0, 100)
    medical_score = np.clip(med_ch * 2 + 50, 0, 100)
    
    return (
        0.50 * win_score +
        0.20 * playoff_score +
        0.15 * injury_score +
        0.10 * health_score +
        0.05 * medical_score
    )

def get_trade_rating(score):
    """Convert composite score to qualitative rating."""
    if score >= 70: return "BENEFICIAL"
    elif score >= 55: return "SLIGHTLY POSITIVE"
    elif score >= 45: return "NEUTRAL"
    elif score >= 30: return "SLIGHTLY NEGATIVE"
    else: return "HARMFUL"
