"""Synthetic, network-free checks for Track L FTN-charted features."""
import unittest

import numpy as np
import pandas as pd

from track_l_ftn_scheme_matchups import (
    add_underdog_scheme, join_ftn_pbp, mean_if, rolling_scheme,
    summarize_unit, team_game_scheme, RATE_NAMES, EFFECT_NAMES, PRODUCTS,
)


class TrackLSchemeTests(unittest.TestCase):
    def rows(self):
        ftn, pbp = [], []
        for game, team1, team2 in (("G1", "AAA", "BBB"), ("G2", "AAA", "BBB")):
            for play in range(1, 25):
                offense = team1 if play % 2 else team2
                defense = team2 if play % 2 else team1
                ftn.append({
                    "nflverse_game_id": game, "nflverse_play_id": play,
                    "is_motion": play % 3 == 0, "is_play_action": play % 4 == 0,
                    "is_screen_pass": play % 5 == 0, "is_rpo": play % 6 == 0,
                    "qb_location": "S" if play % 2 else "U",
                    "n_blitzers": 1 if play % 4 == 0 else 0,
                })
                pbp.append({
                    "game_id": game, "play_id": float(play), "posteam": offense,
                    "defteam": defense, "epa": .05*play, "pass": 1,
                    "rush": 0,
                })
        return pd.DataFrame(ftn), pd.DataFrame(pbp)

    def test_join_and_use_rate_are_explicit(self):
        chart, pbp = self.rows()
        combined, cov = join_ftn_pbp(chart, pbp)
        self.assertEqual(len(combined), 48)
        self.assertEqual(cov["matched_share"], 1.0)
        team_games = team_game_scheme(combined)
        self.assertEqual(len(team_games), 4)
        self.assertTrue(team_games["off_motion_rate"].between(0, 1).all())
        self.assertTrue(team_games["def_blitz_rate"].between(0, 1).all())

    def test_duplicate_chart_play_is_rejected(self):
        ftn, pbp = self.rows()
        bad = pd.concat([ftn, ftn.head(1).assign(is_motion=False)], ignore_index=True)
        with self.assertRaisesRegex(ValueError, "duplicate FTN"):
            join_ftn_pbp(bad, pbp)

    def test_low_sample_epa_does_not_become_zero(self):
        g = pd.DataFrame({"epa": [1., 2., 3., 4.],
                          "pass_play": [True]*4,
                          "is_motion": [True, False, False, False],
                          "is_play_action": [True, False, False, False],
                          "is_screen_pass": [False]*4, "is_rpo": [False]*4,
                          "blitz": [True, False, False, False],
                          "shotgun": [False]*4})
        self.assertTrue(np.isnan(mean_if(g, g["is_motion"])))
        out = summarize_unit(g, defense=False)
        self.assertTrue(np.isnan(out["off_motion_epa"]))
        self.assertEqual(out["off_motion_rate"], .25)

    def test_prior_only_shift_and_future_sensitivity(self):
        schedule=[]
        data=[]
        for i in range(1, 5):
            game=f"G{i}"
            schedule.append({
                "game_id":game, "gameday":pd.Timestamp(2022,9,i),
                "season":2022, "week":i, "home_team":"AAA", "away_team":"BBB",
            })
            for team in ("AAA","BBB"):
                data.append({"game_id":game,"team":team,
                             **{col:float(i if team=="AAA" else -i) for col in
                                (*RATE_NAMES,*EFFECT_NAMES)},
                             "off_charted_plays":20,"def_charted_plays":20})
        base=pd.DataFrame(schedule)
        pg=pd.DataFrame(data)
        orig, cov = rolling_scheme(base,pg)
        self.assertTrue(pd.isna(orig.loc[orig.week.eq(1),"home_off_motion_rate_r4"].iloc[0]))
        self.assertEqual(float(orig.loc[orig.week.eq(3),"home_off_motion_rate_r4"].iloc[0]),1.5)
        pg2=pg.copy()
        pg2.loc[pg2.game_id.eq("G3"),"off_motion_rate"]=999.
        later,_=rolling_scheme(base,pg2)
        self.assertEqual(float(later.loc[later.week.eq(3),"home_off_motion_rate_r4"].iloc[0]),1.5)
        self.assertGreater(float(later.loc[later.week.eq(4),"home_off_motion_rate_r4"].iloc[0]),
                           float(orig.loc[orig.week.eq(4),"home_off_motion_rate_r4"].iloc[0]))
        self.assertEqual(cov["charted_team_game_fraction"],1.)

    def test_interactions_reverse_on_swap(self):
        row={"dog_is_home":True}
        for w in (4,8):
            for name in (*RATE_NAMES,*EFFECT_NAMES):
                row[f"home_{name}_r{w}"]=.8
                row[f"away_{name}_r{w}"]=.2
        x=pd.DataFrame([row])
        z, groups = add_underdog_scheme(x)
        self.assertEqual(len(groups["matchups"]),2*len(PRODUCTS))
        self.assertEqual(len(groups["rates"]),2*len(RATE_NAMES))
        # Keep actual teams, flip dog orientation: sign reverses in every feature.
        y=x.copy()
        y["dog_is_home"]=False
        other, _ = add_underdog_scheme(y)
        for family in groups.values():
            for col in family:
                self.assertAlmostEqual(float(z[col].iloc[0]),-float(other[col].iloc[0]))


if __name__ == "__main__":
    unittest.main()
