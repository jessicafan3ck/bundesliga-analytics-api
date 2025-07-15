from fastapi import FastAPI, Query
from fastapi.responses import JSONResponse
import pandas as pd
import matplotlib.pyplot as plt
import base64
from io import BytesIO
import uvicorn

# Modeling imports
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.linear_model import LinearRegression

# -------------------------
# Load and normalize Bundesliga data
# -------------------------
all_data = pd.read_excel("./trimmed_file.xlsx")

# 1. Normalize column names to snake_case
all_data.columns = [
    c.strip().lower().replace(" ", "_") 
    for c in all_data.columns
]

# 2. Rename a few critical columns for consistency
all_data.rename(columns={
    "full_name": "player",
    "current_club": "current_club",  # already normalized, but explicit
    "appearances_overall": "appearances_overall",
    "goals_overall": "goals_overall",
    "assists_overall": "assists_overall",
    "minutes_played_overall": "minutes_played_overall",
    # add more mapping here if you need custom names...
}, inplace=True)

# -------------------------
# Compute proxy and contribution metrics
# -------------------------
if {"goals_per_90_overall", "minutes_played_overall"}.issubset(all_data.columns):
    all_data["xg_proxy"] = (
        all_data["goals_per_90_overall"] 
        * all_data["minutes_played_overall"] / 90
    )
if {"assists_per_90_overall", "minutes_played_overall"}.issubset(all_data.columns):
    all_data["xa_proxy"] = (
        all_data["assists_per_90_overall"] 
        * all_data["minutes_played_overall"] / 90
    )

if {"goals_per_90_overall", "assists_per_90_overall"}.issubset(all_data.columns):
    all_data["total_contributions_per_90"] = (
        all_data["goals_per_90_overall"] 
        + all_data["assists_per_90_overall"]
    )
if {"min_per_goal_overall", "min_per_assist_overall"}.issubset(all_data.columns):
    all_data["min_per_goal_assist"] = (
        all_data["min_per_goal_overall"] 
        + all_data["min_per_assist_overall"]
    )

# -------------------------
# Player role clustering
# -------------------------
cluster_features = [
    "goals_per_90_overall",
    "assists_per_90_overall",
    "cards_per_90_overall",
    "conceded_per_90_overall",
    "clean_sheets_overall",
]
exist_feats = [f for f in cluster_features if f in all_data.columns]
cluster_df = all_data.dropna(subset=exist_feats).copy()

if exist_feats:
    scaler = StandardScaler()
    X = scaler.fit_transform(cluster_df[exist_feats])
    kmeans = KMeans(n_clusters=4, random_state=42)
    cluster_df["player_role_cluster"] = kmeans.fit_predict(X)

    all_data = all_data.merge(
        cluster_df[["player", "season", "player_role_cluster"]],
        on=["player", "season"],
        how="left",
    )
else:
    all_data["player_role_cluster"] = None

# -------------------------
# FastAPI app setup
# -------------------------
app = FastAPI(
    title="Bundesliga Analytics Assistant",
    description="Endpoints leverage your trimmed_file.xlsx with columns like full_name, age, league, season, current_club, etc."
)

@app.get("/")
def root():
    return {
        "message": "Ready. Use /compare, /visualize, /top, /filter, /plot, /play, /player_profile, /compare_models, /trajectory_predict, /clustering."
    }

@app.get("/players")
def list_players():
    return {"players": sorted(all_data["player"].dropna().unique())}

@app.get("/stats")
def list_stats():
    return {"available_stats": all_data.select_dtypes("number").columns.tolist()}

@app.get("/visualize")
def visualize_stat(player1: str, player2: str, stat: str):
    df1 = all_data[all_data["player"].str.lower() == player1.lower()]
    df2 = all_data[all_data["player"].str.lower() == player2.lower()]
    if df1.empty or df2.empty:
        return JSONResponse(404, {"error": "Player(s) not found"})
    p1 = df1.groupby("season")[stat].mean().reset_index()
    p2 = df2.groupby("season")[stat].mean().reset_index()

    plt.figure()
    plt.plot(p1["season"], p1[stat], marker="o", label=player1)
    plt.plot(p2["season"], p2[stat], marker="x", label=player2)
    plt.title(f"{stat.title()} Comparison: {player1} vs {player2}")
    plt.xlabel("Season"); plt.ylabel(stat.title()); plt.grid(); plt.legend()

    buf = BytesIO()
    plt.savefig(buf, format="png"); plt.close(); buf.seek(0)
    img_b64 = base64.b64encode(buf.read()).decode()
    return {"image": f"data:image/png;base64,{img_b64}"}

@app.get("/compare")
def compare_players(player1: str, player2: str):
    df1 = all_data[all_data["player"].str.lower() == player1.lower()]
    df2 = all_data[all_data["player"].str.lower() == player2.lower()]
    if df1.empty or df2.empty:
        return JSONResponse(404, {
            "error": "One or both players not found.",
            "player1_found": not df1.empty,
            "player2_found": not df2.empty
        })
    stats = ["goals_overall", "assists_overall", "shots_per_90_overall", "passes_per_90_overall"]
    valid = [s for s in stats if s in df1.columns and s in df2.columns]
    return {
        player1: df1[valid].mean(numeric_only=True).round(2).to_dict(),
        player2: df2[valid].mean(numeric_only=True).round(2).to_dict()
    }

@app.get("/top")
def top_players(stat: str = Query(...), season: str = Query(None), top_n: int = 5):
    df = all_data[df["season"] == season] if season else all_data
    df = df[["player", stat]].dropna()
    topn = df.groupby("player")[stat].mean().nlargest(top_n)
    return topn.reset_index().to_dict("records")

@app.get("/filter")
def filter_stats(team: str = None, season: str = None, min_goals: int = 0):
    df = all_data
    if team:
        df = df[df["current_club"].str.lower() == team.lower()]
    if season:
        df = df[df["season"] == season]
    df = df[df["goals_overall"] >= min_goals]
    return df[
        ["player", "current_club", "season", "goals_overall", "assists_overall"]
    ].sort_values("goals_overall", ascending=False).to_dict("records")

@app.get("/plot")
def plot_stat(player: str, stat: str):
    df = all_data[all_data["player"].str.lower() == player.lower()]
    if df.empty:
        return JSONResponse(404, {"error": "Player not found."})
    series = df.groupby("season")[stat].mean().reset_index()
    plt.figure()
    plt.plot(series["season"], series[stat], marker="o")
    plt.title(f"{player} – {stat.title()} by Season")
    plt.xlabel("Season"); plt.ylabel(stat.title()); plt.grid()
    buf = BytesIO(); plt.savefig(buf, format="png"); plt.close(); buf.seek(0)
    return {"image": f"data:image/png;base64,{base64.b64encode(buf.read()).decode()}"}

@app.get("/play")
def play_summary(player: str, season: str):
    samples = {
        "thomas müller": ["Low cross…", "Backheel assist…"],
        "jamal musiala": ["Solo run…", "Nutmeg assist…"]
    }
    return {"player": player, "season": season, "highlight_reel": samples.get(player.lower(), ["No data"])}

@app.get("/player_profile")
def player_profile(player: str, season: str):
    df_seas = all_data[all_data["season"] == season]
    feats = [c for c in ["goals_per_90_overall","assists_per_90_overall","shots_per_90_overall","passes_per_90_overall","dribbles_per_90_overall"] if c in df_seas]
    if df_seas.empty or not feats:
        return JSONResponse(404, {"error": "Season or features not found."})
    scaled = StandardScaler().fit_transform(df_seas[feats])
    df_s = pd.DataFrame(scaled, columns=feats, index=df_seas.index)
    idx = df_s.index[df_seas["player"].str.lower() == player.lower()]
    if idx.empty:
        return JSONResponse(404, {"error": "Player not in season."})
    return {"player": player, "season": season, "radar_stats": df_s.loc[idx[0]].round(2).to_dict()}

@app.get("/compare_models")
def compare_models(player1: str, player2: str, season: str = None):
    df = all_data[all_data["season"] == season] if season else all_data
    def summ(p):
        sub = df[df["player"].str.lower()==p.lower()]
        return None if sub.empty else {
            "xg_proxy": round(sub["xg_proxy"].mean(),2),
            "xa_proxy": round(sub["xa_proxy"].mean(),2),
            "attacking_value_index": round(0.6*sub["xg_proxy"].mean()+0.4*sub["xa_proxy"].mean(),2)
        }
    r1, r2 = summ(player1), summ(player2)
    if not r1 or not r2:
        return JSONResponse(404, {"error": "Player(s) not found."})
    return {player1: r1, player2: r2}

@app.get("/trajectory_predict")
def trajectory_predict(player: str, stat: str):
    dfp = all_data[all_data["player"].str.lower()==player.lower()].dropna(subset=[stat]).copy()
    if len(dfp)<3:
        return JSONResponse(400, {"error":"Need ≥3 seasons of data."})
    dfp["season_year"] = dfp["season"].str.split("-").str[0].astype(int)
    recent = dfp.sort_values("season_year").tail(3)
    X, y = recent[["season_year"]], recent[stat]
    model = LinearRegression().fit(X, y)
    next_year = recent["season_year"].max()+1
    return {
        "player": player, "stat": stat,
        "predicted_season": f"{next_year}-{next_year+1}",
        "predicted_value": round(model.predict([[next_year]])[0],2)
    }

@app.get("/clustering")
def clustering():
    return {
        "clusters": {
            int(l): all_data[all_data["player_role_cluster"]==l]["player"].unique().tolist()
            for l in sorted(all_data["player_role_cluster"].dropna().unique())
        }
    }

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=10000, reload=True)
