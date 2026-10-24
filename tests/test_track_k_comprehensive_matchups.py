"""Network-free invariant tests for Track K, never touching production."""
import unittest

import numpy as np
import pandas as pd
import polars as pl

from track_k_comprehensive_matchups import (
    ADJUSTED_METRICS, EXTRA_SITUATIONS, aggregate_game_situations,
    build_prior_team_features, matchup_interactions, compare_paired,
)


class TrackKTests(unittest.TestCase):
    def test_aggregate_situations(self):
        frame = pl.DataFrame([
            {"game_id": "G1", "posteam": "AAA", "defteam": "BBB", "epa": 0.4,
             "success": 1.0, "pass": 1, "rush": 0, "down": 1, "yardline_100": 50.0},
            {"game_id": "G1", "posteam": "AAA", "defteam": "BBB", "epa": -0.2,
             "success": 0.0, "pass": 0, "rush": 1, "down": 3, "yardline_100": 15.0},
            {"game_id": "G1", "posteam": "BBB", "defteam": "AAA", "epa": 0.1,
             "success": 1.0, "pass": 1, "rush": 0, "down": 2, "yardline_100": 65.0},
        ])
        out = aggregate_game_situations(frame).set_index("team")
        self.assertAlmostEqual(float(out.loc["AAA", "off_early_epa"]), .4)
        self.assertAlmostEqual(float(out.loc["AAA", "off_third_epa"]), -.2)
        self.assertAlmostEqual(float(out.loc["AAA", "off_redzone_epa"]), -.2)
        self.assertAlmostEqual(float(out.loc["BBB", "def_early_epa"]), .4)

    def make_history(self):
        rows, per = [], []
        for week in range(1, 5):
            r = {"game_id": f"G{week}", "gameday": pd.Timestamp(2019, 9, week),
                 "season": 2019, "week": week, "home_team": "AAA", "away_team": "BBB",
                 "away_def_epa_allowed_r8": .1, "away_off_epa_r8": .2,
                 "home_def_epa_allowed_r8": .3, "home_off_epa_r8": .4}
            rows.append(r)
            for team in ("AAA", "BBB"):
                vals = {"game_id": r["game_id"], "team": team}
                for side in ("off", "def"):
                    for metric in EXTRA_SITUATIONS:
                        vals[f"{side}_{metric}"] = float(week if team == "AAA" else -week)
                vals["off_all_epa"] = float(week if team == "AAA" else -week)
                vals["def_all_epa"] = float(week if team == "AAA" else -week)
                per.append(vals)
        return pd.DataFrame(rows), pd.DataFrame(per)

    def test_rolling_uses_prior_games_only(self):
        games, actual = self.make_history()
        original, sit, adj = build_prior_team_features(games, actual)
        self.assertIn("diff_off_early_epa_r4", sit)
        self.assertIn("diff_off_opponent_relative_epa_r4", adj)
        self.assertTrue(pd.isna(original.loc[original.week.eq(1), "home_off_early_epa_r4"].iloc[0]))
        self.assertAlmostEqual(
            float(original.loc[original.week.eq(3), "home_off_early_epa_r4"].iloc[0]), 1.5
        )
        changed = actual.copy()
        changed.loc[changed.game_id.eq("G3"), "off_early_epa"] = 999
        after, _, _ = build_prior_team_features(games, changed)
        self.assertAlmostEqual(
            float(original.loc[original.week.eq(3), "home_off_early_epa_r4"].iloc[0]),
            float(after.loc[after.week.eq(3), "home_off_early_epa_r4"].iloc[0]),
        )
        self.assertGreater(
            float(after.loc[after.week.eq(4), "home_off_early_epa_r4"].iloc[0]),
            float(original.loc[original.week.eq(4), "home_off_early_epa_r4"].iloc[0]),
        )
        self.assertEqual(len(ADJUSTED_METRICS), 2)

    def test_matchup_products_reverse_on_team_swap(self):
        r = {}
        sources = ("off_pass_epa", "def_pass_epa_allowed", "off_rush_epa",
                   "def_rush_epa_allowed", "off_early_epa", "def_early_epa",
                   "off_third_epa", "def_third_epa", "off_redzone_epa",
                   "def_redzone_epa", "off_sack_rate", "def_sack_rate")
        for window in (4, 8):
            for source in sources:
                r[f"home_{source}_r{window}"] = .7
                r[f"away_{source}_r{window}"] = -.3
        source = pd.DataFrame([r])
        orig, cols = matchup_interactions(source)
        self.assertEqual(len(cols), 12)
        swapped = source.copy()
        for w in (4, 8):
            for stem in sources:
                h, a = f"home_{stem}_r{w}", f"away_{stem}_r{w}"
                swapped[h], swapped[a] = source[a], source[h]
        other, _ = matchup_interactions(swapped)
        for col in cols:
            self.assertAlmostEqual(float(orig[col].iloc[0]), -float(other[col].iloc[0]))

    def test_paired_no_disagreements(self):
        f = pd.DataFrame({
            "season": [2022] * 4, "home_win": [0, 1, 0, 1],
            "p_a": [0.4, 0.7, 0.2, 0.8], "p_b": [0.4, 0.7, 0.2, 0.8],
        })
        c = compare_paired(f, "a", "b")
        self.assertEqual(c["net_correct"], 0)
        self.assertEqual(c["disagreement_wins"], 0)
        self.assertEqual(c["mcnemar_exact_p"], 1.0)


if __name__ == "__main__":
    unittest.main()
