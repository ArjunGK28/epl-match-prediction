"""Build the F1, F2 and F3 feature sets from the raw rosters, lineups and match results.

Run from the repository root:  python src/build_features.py
Writes data/processed/matches.csv with one row per match.

Only the starting eleven is used (lineup slots 1 to 11). Substitutes and minutes played are
ignored because they are not known before kick-off.
"""
import csv
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
RAW = ROOT / "data" / "raw"
OUT = ROOT / "data" / "processed" / "matches.csv"

SEASONS = ["0910", "1011", "1112", "1213", "1314", "1415", "1516", "1617", "1718", "1819"]
N_STARTERS = 11
POSITIONS = {"Goalkeeper": "gk", "Defender": "def", "Midfielder": "mid", "Forward": "fwd"}
# F3 statistic name -> column index in the roster files (PS% and Drb appear twice, so use indices)
PLAYER_STATS = {"rating": 12, "pass_success": 9, "passes": 27, "shots": 8, "key_passes": 21,
                "blocks": 19, "interceptions": 14}

F1_COLUMNS = ["h_rating", "a_rating"]
F2_COLUMNS = F1_COLUMNS + [f"{side}_{pos}" for side in "ha" for pos in POSITIONS.values()]
F3_COLUMNS = [f"{side}_p{i}_{stat}" for side in "ha" for i in range(1, N_STARTERS + 1) for stat in PLAYER_STATS]
META_COLUMNS = ["season", "match_id", "home_team", "away_team", "home_goals", "away_goals", "label"]


def read_rows(path):
    with open(path, encoding="utf-8") as f:
        return list(csv.reader(f))


def load_roster(season):
    """Season statistics per player, plus each team's mean as a fallback for unlisted players."""
    rows = read_rows(RAW / "rosters" / f"rosters_with_stats_20{season[:2]}-20{season[2:]}.csv")[1:]
    players, by_team = {}, {}
    for row in rows:
        stats = np.array([float(row[i]) for i in PLAYER_STATS.values()])
        players[row[2]] = stats
        by_team.setdefault(row[0], []).append(stats)
    team_mean = {team: np.mean(stats, axis=0) for team, stats in by_team.items()}
    return players, team_mean


def team_features(row, header, side, players, team_mean):
    team_id = row[header.index(f"{side}_id")]
    stats, positions = [], []
    for i in range(1, N_STARTERS + 1):
        j = header.index(f"{side}_p{i}_id")
        stats.append(players.get(row[j], team_mean[team_id]))
        positions.append(POSITIONS[row[j + 2]])
    stats = np.array(stats)
    ratings = stats[:, 0]
    f1 = ratings.mean()
    # a formation with no player in some position falls back to the eleven's average
    f2 = [ratings[[p == pos for p in positions]].mean() if pos in positions else f1 for pos in POSITIONS.values()]
    return f1, f2, stats.ravel()


def build():
    matches = []
    for season in SEASONS:
        players, team_mean = load_roster(season)
        results = {r[0].strip(): (int(r[1]), int(r[2]))
                   for r in read_rows(RAW / "match_results" / f"match_results-{season}.csv")[1:]}
        lineups = read_rows(RAW / "lineups" / f"match_lineup_{season}.csv")
        header = lineups[0]
        for row in lineups[1:]:
            if row[0] not in results:
                continue
            home_goals, away_goals = results[row[0]]
            label = int(np.sign(home_goals - away_goals))  # 1 home win, 0 draw, -1 home loss
            h1, h2, h3 = team_features(row, header, "h", players, team_mean)
            a1, a2, a3 = team_features(row, header, "a", players, team_mean)
            meta = [season, row[0], row[2], row[4], home_goals, away_goals, label]
            matches.append(meta + [h1, a1] + h2 + a2 + list(h3) + list(a3))
    return matches


def main():
    matches = build()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(META_COLUMNS + F2_COLUMNS + F3_COLUMNS)
        for m in matches:
            writer.writerow(m[:len(META_COLUMNS)] + [f"{v:.4f}" for v in m[len(META_COLUMNS):]])
    labels = np.array([m[6] for m in matches])
    print(f"{len(matches)} matches written to {OUT.relative_to(ROOT)}")
    print(f"home win {(labels == 1).mean():.4f}  draw {(labels == 0).mean():.4f}  home loss {(labels == -1).mean():.4f}")
    print(f"features: F1 {len(F1_COLUMNS)}, F2 {len(F2_COLUMNS)}, F3 {len(F3_COLUMNS)}")


if __name__ == "__main__":
    main()
