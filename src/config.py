"""Shared settings + paths. Tune the MODEL dict after backtesting."""
from datetime import date
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
OUT = ROOT / "output"
DOCS = ROOT / "docs"

# ---- model knobs (backtest.py reports how each one moves accuracy) ----
MODEL = dict(
    decay=0.90,        # weight per game ago when building ratings (0.90 = last ~10 games matter)
    n_sims=2000,       # simulations per game
    hfa=0.012,         # home-field edge as a % boost to yards per play (home +, away -)
    conc_tgt=40.0,     # Dirichlet concentration, target shares (lower = more game-to-game volume swing)
    conc_car=30.0,     # same for carries
    rz_boost=1.20,     # yards multiplier inside the 12 (calibrates red-zone scoring)
    form_sd=0.05,      # game-to-game 'team form' swing in yards/play (adds realistic spread)
    q_play_prob=0.80,  # chance a 'Questionable' player is active
    time_scale=1.05,   # global clock multiplier; raise = fewer plays per game
    eff=1.00,          # global yards-per-play multiplier (calibration; keep close to 1)
)


def current_season(today=None):
    t = today or date.today()
    return t.year if t.month >= 3 else t.year - 1


def seasons(today=None):
    c = current_season(today)
    return [c - 2, c - 1, c]
