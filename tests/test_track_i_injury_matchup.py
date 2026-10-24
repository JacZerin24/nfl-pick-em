"""Deterministic, network-free checks for Track I research features."""
import unittest

import pandas as pd

from track_i_injury_matchup_interactions import (
    GROUPS, MATCHUPS, add_injury_matchup_features, paired_compare,
)


class InjuryMatchupTests(unittest.TestCase):
    def setup_frames(self):
        game = {"game_id": "2019_01_BBB_AAA", "season": 2019, "week": 1,
                "home_team": "AAA", "away_team": "BBB", "home_win": 1}
        for s in ("home", "away"):
            for stat in ("def_sack_rate", "def_takeaway_rate", "def_pass_epa_allowed",
                         "off_rush_epa", "off_pass_epa"):
                game[f"{s}_{stat}_r8"] = .3 if s == "home" else .1
        h = {"season": 2019, "week": 1, "team": "AAA"}
        a = {"season": 2019, "week": 1, "team": "BBB"}
        for group in GROUPS:
            h[f"{group}_severe_backup_gap"] = 0.0
            a[f"{group}_severe_backup_gap"] = 1.0
        return pd.DataFrame([game]), pd.DataFrame([h, a])

    def test_missing_opponent_is_not_invented(self):
        games, injury = self.setup_frames()
        out, _, _ = add_injury_matchup_features(games, injury.query("team == 'AAA'"))
        self.assertEqual(int(out["both_reports_present"].iloc[0]), 0)
        self.assertEqual(float(out["injury_opponent_ol_pressure"].iloc[0]), 0.0)

    def test_opponent_strength_x_missing_unit(self):
        games, injury = self.setup_frames()
        out, pos, ints = add_injury_matchup_features(games, injury)
        self.assertEqual(len(pos), 5)
        self.assertEqual(len(ints), len(MATCHUPS))
        self.assertEqual(int(out["both_reports_present"].iloc[0]), 1)
        self.assertAlmostEqual(float(out["injury_opponent_ol_pressure"].iloc[0]), .3)
        self.assertAlmostEqual(float(out["injury_opponent_db_passing"].iloc[0]), .3)
        self.assertAlmostEqual(float(out["injury_opponent_skill_pass_defense"].iloc[0]), -.3)
        self.assertAlmostEqual(float(out["injury_gap_diff_ol"].iloc[0]), 1.0)

    def test_swapping_sides_reverses_interaction_sign(self):
        games, injury = self.setup_frames()
        original, _, ints = add_injury_matchup_features(games, injury)
        flipped = games.copy()
        for stem in ("team", "def_sack_rate_r8", "def_takeaway_rate_r8", "def_pass_epa_allowed_r8", "off_rush_epa_r8", "off_pass_epa_r8"):
            flipped[f"home_{stem}"], flipped[f"away_{stem}"] = games[f"away_{stem}"], games[f"home_{stem}"]
        # Injuries follow the actual teams after swapping sides.
        changed, _, _ = add_injury_matchup_features(flipped, injury)
        for c in ints:
            self.assertAlmostEqual(float(original[c].iloc[0]), -float(changed[c].iloc[0]))

    def test_equal_prediction_has_zero_disagreement(self):
        df = pd.DataFrame({"season": [2019] * 4, "home_win": [1, 0, 1, 0],
                           "p_a": [.6, .3, .2, .8], "p_b": [.6, .3, .2, .8]})
        r = paired_compare(df, "a", "b")
        self.assertEqual(r["net_correct"], 0)
        self.assertEqual(r["flip_wins"], 0)
        self.assertEqual(r["flip_losses"], 0)
        self.assertEqual(r["mcnemar_exact_p"], 1.0)


if __name__ == "__main__":
    unittest.main()
