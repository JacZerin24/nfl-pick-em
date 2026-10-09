"""Network-free source and pregame-safety invariants for Track M."""
import unittest
import numpy as np
import pandas as pd

from track_m_participation_matchups import (
    COVERAGE_METRICS, MATCHUPS, add_coverage_matchups,
    game_summary, join_participation, roll_teams, compare_consensus,
)


class TrackMTests(unittest.TestCase):
    def fixture(self):
        part, pbp=[],[]
        for week in range(1,5):
            for play in range(1,21):
                team="AAA" if play%2 else "BBB"
                part.append({
                    "nflverse_game_id":f"G{week}","play_id":play,
                    "defense_man_zone_type":"MAN" if play%4<2 else "ZONE",
                    "defense_coverage_type":"COVER_1" if play%4<2 else "COVER_2",
                    "was_pressure":bool(play%3==0),
                    "route":"GO" if play%5==0 else "SLANT",
                    "offense_players":"player001;player002",
                    "defense_players":"player003;player004",
                })
                pbp.append({
                    "game_id":f"G{week}","play_id":play,
                    "posteam":team,"defteam":"BBB" if team=="AAA" else "AAA",
                    "epa":.1*(play%7),"pass":1,"rush":0,
                })
        return pd.DataFrame(part),pd.DataFrame(pbp)

    def test_join_rates_and_identity_fields(self):
        p,b=self.fixture()
        played,stats=join_participation(p,b)
        self.assertEqual(stats["join_share"],1.)
        self.assertEqual(len(played),80)
        self.assertTrue(played["valid_coverage"].all())
        s=game_summary(played)
        self.assertEqual(len(s),8)
        self.assertTrue(s.def_man_rate.between(0,1).all())
        self.assertTrue(s.def_pressure_rate.between(0,1).all())

    def test_falsified_duplicate_is_rejected(self):
        p,b=self.fixture()
        extra=p.head(1).assign(defense_man_zone_type="DIFFERENT")
        with self.assertRaisesRegex(ValueError,"Conflicting"):
            join_participation(pd.concat([p,extra],ignore_index=True),b)

    def test_rolling_never_uses_current_game(self):
        schedule=pd.DataFrame([
            {"game_id":f"G{i}","gameday":pd.Timestamp(2023,9,i),
             "home_team":"AAA","away_team":"BBB"} for i in range(1,5)
        ])
        perf=[]
        for i in range(1,5):
            for t in ("AAA","BBB"):
                perf.append({"game_id":f"G{i}","team":t,
                             **{metric:float(i if t=="AAA" else -i)
                                for metric in COVERAGE_METRICS}})
        perf=pd.DataFrame(perf)
        pre=roll_teams(schedule,perf)
        self.assertTrue(np.isnan(pre.loc[pre.game_id.eq("G1"),"home_def_man_rate_r4"].iloc[0]))
        self.assertEqual(float(pre.loc[pre.game_id.eq("G3"),"home_def_man_rate_r4"].iloc[0]),1.5)
        changed=perf.copy()
        changed.loc[changed.game_id.eq("G3"),"def_man_rate"]=999.
        second=roll_teams(schedule,changed)
        self.assertEqual(float(second.loc[second.game_id.eq("G3"),"home_def_man_rate_r4"].iloc[0]),1.5)
        self.assertGreater(float(second.loc[second.game_id.eq("G4"),"home_def_man_rate_r4"].iloc[0]),
                           float(pre.loc[pre.game_id.eq("G4"),"home_def_man_rate_r4"].iloc[0]))

    def test_dog_home_symmetry(self):
        row={"dog_is_home":True}
        for w in (4,8):
            for metric in COVERAGE_METRICS:
                row[f"home_{metric}_r{w}"]=.7
                row[f"away_{metric}_r{w}"]=.2
        x=pd.DataFrame([row])
        d, groups=add_coverage_matchups(x)
        swapped=x.copy()
        swapped["dog_is_home"]=False
        o,_=add_coverage_matchups(swapped)
        self.assertEqual(len(groups["interactions"]),2*len(MATCHUPS))
        for family in groups.values():
            for col in family:
                self.assertAlmostEqual(float(d[col].iloc[0]),-float(o[col].iloc[0]))

    def test_equal_picks_have_no_consensus_advantage(self):
        x=pd.DataFrame({"dog_win":[0,1,0,1],"p_a":[.1,.8,.3,.6],
                        "p_b":[.1,.8,.3,.6],"p_variance":[.9]*4})
        result=compare_consensus(x,"a","b")
        self.assertEqual(result["net_correct"],0)
        self.assertEqual(result["flip_wins"],0)
        self.assertEqual(result["exact_p"],1.)


if __name__=="__main__":
    unittest.main()
