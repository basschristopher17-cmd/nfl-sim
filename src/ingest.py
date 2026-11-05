"""Pull nflverse data once and cache it as parquet in data/."""
import nflreadpy as nfl
from config import DATA, seasons

PBP_COLS = [
    "game_id", "season", "week", "season_type", "posteam", "defteam", "home_team", "away_team",
    "play_type", "yards_gained", "sack", "interception", "complete_pass", "qb_scramble",
    "fumble_lost", "touchdown", "return_touchdown", "wp", "qtr", "down", "ydstogo", "yardline_100",
    "passer_player_id", "receiver_player_id", "rusher_player_id",
    "field_goal_attempt", "field_goal_result", "kick_distance",
]
PS_COLS = [
    "player_id", "player_display_name", "position", "team", "season", "week", "season_type",
    "completions", "attempts", "passing_yards", "passing_tds", "passing_interceptions",
    "carries", "rushing_yards", "rushing_tds", "targets", "receptions", "receiving_yards", "receiving_tds",
]


def main():
    DATA.mkdir(exist_ok=True)
    S = seasons()
    print("seasons:", S)
    pbp = nfl.load_pbp(S).to_pandas()
    pbp[[c for c in PBP_COLS if c in pbp.columns]].to_parquet(DATA / "pbp.parquet", index=False)
    ps = nfl.load_player_stats(S).to_pandas()
    ps[[c for c in PS_COLS if c in ps.columns]].to_parquet(DATA / "player_stats.parquet", index=False)
    nfl.load_schedules(S).to_pandas().to_parquet(DATA / "schedule.parquet", index=False)
    nfl.load_injuries(S).to_pandas().to_parquet(DATA / "injuries.parquet", index=False)
    # --- rosters: weekly roster (status per week) + the newest depth-chart snapshot (updated daily in season) ---
    ro = nfl.load_rosters_weekly(S).to_pandas()
    keep = ["season", "week", "team", "position", "depth_chart_position", "status", "full_name", "gsis_id", "game_type"]
    ro[[c for c in keep if c in ro.columns]].to_parquet(DATA / "rosters.parquet", index=False)
    dp = nfl.load_depth_charts([S[-1]]).to_pandas()
    dp = dp[dp.dt == dp.dt.max()][["dt", "team", "player_name", "gsis_id", "pos_abb", "pos_rank"]]
    dp.to_parquet(DATA / "depth.parquet", index=False)
    print("rosters:", len(ro), "rows | depth chart snapshot:", dp.dt.iloc[0], len(dp), "rows")
    print("cached ->", DATA)


if __name__ == "__main__":
    main()
