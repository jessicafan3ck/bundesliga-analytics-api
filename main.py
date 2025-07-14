from fastapi import FastAPI, Query
from fastapi.responses import JSONResponse
import pandas as pd
import matplotlib.pyplot as plt
import base64
from io import BytesIO
import uvicorn

# -------------------------
# Load and normalize Bundesliga data
# -------------------------
all_data = pd.read_csv("./merged_bundesliga_stats.csv")
all_data.columns = [c.strip().lower().replace(" ", "_") for c in all_data.columns]

# Ensure consistent player column
if 'full_name' in all_data.columns:
    all_data.rename(columns={"full_name": "player"}, inplace=True)

# -------------------------
# FastAPI app setup
# -------------------------
app = FastAPI(title="Bundesliga Analytics Assistant")

@app.get("/")
def root():
    return {"message": "Bundesliga Analytics Assistant ready. Use /compare, /top, /filter, /plot, /play."}

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
    return df[['player', 'current_club', 'season', 'goals_overall', 'assists_overall']].sort_values(by="goals_overall", ascending=False).to_dict(orient="records")

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
        "jamal musiala": ["Solo run into the box, tight dribble and goal.", "Nutmeg assist through a crowd."]
    }
    key = player.lower()
    plays = samples.get(key, ["No highlight data available for this player."])
    return {"player": player, "season": season, "highlight_reel": plays}

# -------------------------
# Run locally for testing
# -------------------------
if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
