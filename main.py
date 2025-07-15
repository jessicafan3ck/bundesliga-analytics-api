from fastapi import FastAPI, Query
from fastapi.responses import JSONResponse
import pandas as pd
import matplotlib.pyplot as plt
import base64
from io import BytesIO
import uvicorn

# Additional imports for modeling
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.linear_model import LinearRegression

# -------------------------
# Load and normalize Bundesliga data
# -------------------------
all_data = pd.read_excel("./trimmed_file.xlsx")  # ✅ Load from Excel
all_data.columns = [c.strip().lower().replace(" ", "_") for c in all_data.columns]

# Ensure consistent player column
if 'full_name' in all_data.columns:
    all_data.rename(columns={"full_name": "player"}, inplace=True)

# Compute proxy metrics
if {'goals_per_90_overall', 'minutes_played_overall'}.issubset(all_data.columns):
    all_data['xg_proxy'] = all_data['goals_per_90_overall'] * all_data['minutes_played_overall'] / 90
if {'assists_per_90_overall', 'minutes_played_overall'}.issubset(all_data.columns):
    all_data['xa_proxy'] = all_data['assists_per_90_overall'] * all_data['minutes_played_overall'] / 90

# Contribution metrics
if {'goals_per_90_overall', 'assists_per_90_overall'}.issubset(all_data.columns):
    all_data['total_contributions_per_90'] = (
        all_data['goals_per_90_overall'] + all_data['assists_per_90_overall']
    )
if {'min_per_goal_overall', 'min_per_assist_overall'}.issubset(all_data.columns):
    all_data['min_per_goal_assist'] = (
        all_data['min_per_goal_overall'] + all_data['min_per_assist_overall']
    )

# Player role clustering setup
cluster_features = [
    'goals_per_90_overall',
    'assists_per_90_overall',
    'cards_per_90_overall',
    'conceded_per_90_overall',
    'clean_sheets_overall'
]
# Ensure features exist
exist_feats = [f for f in cluster_features if f in all_data.columns]
cluster_df = all_data.dropna(subset=exist_feats).copy()
if exist_feats:
    scaler = StandardScaler()
    X = scaler.fit_transform(cluster_df[exist_feats])
    kmeans = KMeans(n_clusters=4, random_state=42)
    cluster_labels = kmeans.fit_predict(X)
    cluster_df['player_role_cluster'] = cluster_labels
    # Merge cluster labels back to main DataFrame
    all_data = all_data.merge(
        cluster_df[['player', 'season', 'player_role_cluster']],
        on=['player', 'season'], how='left'
    )
else:
    all_data['player_role_cluster'] = None

# -------------------------
# FastAPI app setup
# -------------------------
app = FastAPI(title="Bundesliga Analytics Assistant")

@app.get("/")
def root():
    return {"message": "Bundesliga Analytics Assistant ready. Use /compare, /top, /filter, /plot, /play, /player_profile, /compare_models, /trajectory_predict, /clustering."}

@app.get("/players")
def list_players():
    players = sorted(all_data['player'].dropna().unique().tolist())
    return {"players": players}

@app.get("/stats")
def list_stats():
    numeric_cols = all_data.select_dtypes(include='number').columns.tolist()
    return {"available_stats": numeric_cols}

@app.get("/visualize")
def visualize_stat(player1: str, player2: str, stat: str):
    df1 = all_data[all_data['player'].str.lower() == player1.lower()]
    df2 = all_data[all_data['player'].str.lower() == player2.lower()]
    if df1.empty or df2.empty:
        return JSONResponse(status_code=404, content={"error": "Player(s) not found"})
    plot_df1 = df1.groupby('season')[stat].mean().reset_index()
    plot_df2 = df2.groupby('season')[stat].mean().reset_index()
    plt.figure()
    plt.plot(plot_df1['season'], plot_df1[stat], marker='o', label=player1)
    plt.plot(plot_df2['season'], plot_df2[stat], marker='x', label=player2)
    plt.title(f"{stat.title()} Comparison: {player1} vs {player2}")
    plt.xlabel("Season")
    plt.ylabel(stat.title())
    plt.legend()
    plt.grid(True)
    buf = BytesIO()
    plt.savefig(buf, format='png')
    plt.close()
    buf.seek(0)
    image_base64 = base64.b64encode(buf.read()).decode('utf-8')
    return {"image": f"data:image/png;base64,{image_base64}"}

@app.get("/compare")
def compare_players(player1: str, player2: str):
    df1 = all_data[all_data['player'].str.lower() == player1.lower()]
    df2 = all_data[all_data['player'].str.lower() == player2.lower()]
    if df1.empty or df2.empty:
        return JSONResponse(status_code=404, content={
            "error": "One or both players not found.",
            "player1_found": not df1.empty,
            "player2_found": not df2.empty
        })
    stats = ['goals_overall', 'assists_overall', 'shots_per_90_overall', 'passes_per_90_overall']
    valid_stats = [s for s in stats if s in df1.columns and s in df2.columns]
    summary = {
        player1: df1[valid_stats].mean(numeric_only=True).round(2).to_dict(),
        player2: df2[valid_stats].mean(numeric_only=True).round(2).to_dict()
    }
    return summary

@app.get("/top")
def top_players(stat: str = Query(...), season: str = Query(None), top_n: int = 5):
    df = all_data
    if season:
        df = df[df['season'] == season]
    df = df[['player', stat]].dropna()
    top_df = df.groupby('player')[stat].mean().sort_values(ascending=False).head(top_n)
    return top_df.reset_index().to_dict(orient="records")

@app.get("/filter")
def filter_stats(team: str = None, season: str = None, min_goals: int = 0):
    df = all_data
    if team:
        df = df[df['current_club'].str.lower() == team.lower()]
    if season:
        df = df[df['season'] == season]
    df = df[df['goals_overall'] >= min_goals]
    return (
        df[['player', 'current_club', 'season', 'goals_overall', 'assists_overall']]
        .sort_values(by="goals_overall", ascending=False)
        .to_dict(orient="records")
    )

@app.get("/plot")
def plot_stat(player: str, stat: str):
    df = all_data[all_data['player'].str.lower() == player.lower()]
    if df.empty:
        return JSONResponse(status_code=404, content={"error": "Player not found."})
    plot_df = df.groupby('season')[stat].mean().reset_index()
    plt.figure()
    plt.plot(plot_df['season'], plot_df[stat], marker='o')
    plt.title(f"{player} - {stat.title()} by Season")
    plt.xlabel("Season")
    plt.ylabel(stat.title())
    plt.grid(True)
    buf = BytesIO()
    plt.savefig(buf, format='png')
    plt.close()
    buf.seek(0)
    image_base64 = base64.b64encode(buf.read()).decode('utf-8')
    return {"image": f"data:image/png;base64,{image_base64}"}

@app.get("/play")
def play_summary(player: str, season: str):
    samples = {
        "thomas muller": ["Low cross from the right, first-time flick into the net.", "Backheel assist under pressure."],
        "jamal musiala": ["Solo run into the box, tight dribble and goal.", "Nutmeg assistant through a crowd."]
    }
    key = player.lower()
    plays = samples.get(key, ["No highlight data available for this player."])
    return {"player": player, "season": season, "highlight_reel": plays}

# -------------------------
# New MCP Endpoints
# -------------------------
@app.get("/player_profile")
def player_profile(player: str, season: str):
    df_season = all_data[all_data['season'] == season]
    if df_season.empty:
        return JSONResponse(status_code=404, content={"error": "Season not found."})
    features = ['goals_per_90_overall', 'assists_per_90_overall', 'shots_per_90_overall', 'passes_per_90_overall', 'dribbles_per_90_overall']
    feats = [f for f in features if f in df_season.columns]
    if not feats:
        return {"player": player, "season": season, "radar_stats": {}}
    scaler = StandardScaler()
    scaled = scaler.fit_transform(df_season[feats])
    df_scaled = pd.DataFrame(scaled, columns=feats, index=df_season.index)
    mask = df_scaled.index[df_season['player'].str.lower() == player.lower()]
    if mask.empty:
        return JSONResponse(status_code=404, content={"error": "Player not found in season."})
    player_stats = df_scaled.loc[mask[0]].round(2).to_dict()
    return {"player": player, "season": season, "radar_stats": player_stats}

@app.get("/compare_models")
def compare_models_models(player1: str, player2: str, season: str = None):
    df = all_data
    if season:
        df = df[df['season'] == season]
    def summarize(p):
        sub = df[df['player'].str.lower() == p.lower()]
        if sub.empty:
            return None
        xg = sub['xg_proxy'].mean()
        xa = sub['xa_proxy'].mean()
        avi = 0.6 * xg + 0.4 * xa
        return {"xg_proxy": round(xg, 2), "xa_proxy": round(xa, 2), "attacking_value_index": round(avi, 2)}
    sum1 = summarize(player1)
    sum2 = summarize(player2)
    if sum1 is None or sum2 is None:
        return JSONResponse(status_code=404, content={"error": "One or both players not found."})
    return {player1: sum1, player2: sum2}

@app.get("/trajectory_predict")
def trajectory_predict(player: str, stat: str):
    dfp = all_data[all_data['player'].str.lower() == player.lower()].dropna(subset=[stat]).copy()
    if len(dfp) < 3:
        return JSONResponse(status_code=400, content={"error": "Need at least 3 seasons of data to predict."})
    def season_to_year(s):
        try:
            return int(s.split('-')[0])
        except:
            return None
    dfp['season_year'] = dfp['season'].apply(season_to_year)
    dfp = dfp.dropna(subset=['season_year'])
    dfp = dfp.sort_values('season_year')
    recent = dfp.tail(3)
    X = recent[['season_year']].values
    y = recent[stat].values
    model = LinearRegression()
    model.fit(X, y)
    next_year = int(recent['season_year'].max()) + 1
    pred = model.predict([[next_year]])[0]
    next_season = f"{next_year}-{next_year+1}"
    return {"player": player, "stat": stat, "predicted_season": next_season, "predicted_value": round(pred, 2)}

@app.get("/clustering")
def clustering():
    clusters = {}
    for lab in sorted(all_data['player_role_cluster'].dropna().unique()):
        members = all_data[all_data['player_role_cluster'] == lab]['player'].unique().tolist()
        clusters[int(lab)] = members
    return {"clusters": clusters}

# -------------------------
# Run locally for testing
# -------------------------
if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=10000, reload=True)
