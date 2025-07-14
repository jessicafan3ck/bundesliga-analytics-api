from fastapi import FastAPI, Query
from fastapi.responses import JSONResponse
import pandas as pd
import matplotlib.pyplot as plt
import base64
from io import BytesIO
import uvicorn
import os

# -------------------------
# Load and preprocess data
# -------------------------
dataframes = []
data_folder = "./bundesliga_data"

for filename in os.listdir(data_folder):
    if filename.endswith(".csv"):
        df = pd.read_csv(os.path.join(data_folder, filename))
        df['season'] = filename.split("-")[2][:4]  # crude season tag
        dataframes.append(df)

all_data = pd.concat(dataframes, ignore_index=True)

# Normalize column names
all_data.columns = [c.strip().lower().replace(" ", "_") for c in all_data.columns]

# -------------------------
# FastAPI app setup
# -------------------------
app = FastAPI(title="Bundesliga Analytics Assistant")

@app.get("/")
def root():
    return {"message": "Bundesliga Analytics Assistant ready. Use /compare, /top, /filter, /plot, /play."}

@app.get("/compare")
def compare_players(player1: str, player2: str):
    df1 = all_data[all_data['player'] == player1]
    df2 = all_data[all_data['player'] == player2]

    if df1.empty or df2.empty:
        return JSONResponse(status_code=404, content={"error": "Player not found."})

    stats = ['goals', 'assists', 'shots_per_game', 'passes_per_game']
    summary = {
        player1: df1[stats].mean().round(2).to_dict(),
        player2: df2[stats].mean().round(2).to_dict()
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
        df = df[df['team'] == team]
    if season:
        df = df[df['season'] == season]
    df = df[df['goals'] >= min_goals]
    return df[['player', 'team', 'season', 'goals', 'assists']].sort_values(by="goals", ascending=False).to_dict(orient="records")

@app.get("/plot")
def plot_stat(player: str, stat: str):
    df = all_data[all_data['player'] == player]
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
    # mock plays
    samples = {
        "thomas muller": ["Low cross from the right, first-time flick into the net.", "Backheel assist under pressure."],
        "erling haaland": ["Bulldozes through defense, powers it home.", "Counterattack finish from 30 yards."]
    }
    key = player.lower()
    plays = samples.get(key, ["No highlight data available for this player."])
    return {"player": player, "season": season, "highlight_reel": plays}

# -------------------------
# Main (optional)
# -------------------------
if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
