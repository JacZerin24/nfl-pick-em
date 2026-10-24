"""Track J kickoff-slot identification and pregame residual invariants."""
import unittest

import pandas as pd

from track_j_primetime_slots import slot_labels, add_prior_team_slot_history, paired


class PrimetimeTests(unittest.TestCase):
    def test_night_and_day_slots(self):
        rows = [
            ("2019-09-08", "20:20", "SUN_NIGHT_PROXY"),
            ("2019-09-09", "20:15", "MON_NIGHT_PROXY"),
            ("2019-09-12", "20:20", "THU_NIGHT_PROXY"),
            ("2019-09-08", "13:00", "SUN_EARLY"),
            ("2019-09-08", "16:25", "SUN_LATE"),
            ("2019-09-08", "09:30", "SUN_MORNING"),
            ("2019-09-08", "unknown", "OTHER"),
        ]
        df = pd.DataFrame([{"gameday":d, "gametime":t} for d,t,_ in rows])
        result = slot_labels(df)
        self.assertEqual(result["slot"].tolist(), [s for _,_,s in rows])
        self.assertEqual(int(result["slot_is_missing"].iloc[-1]), 1)

    def example(self):
        rows=[]
        slots = (("2019-09-12","20:20"),("2019-09-15","13:00"),
                 ("2019-09-19","20:20"),("2019-09-26","20:20"))
        for i, (day, clock) in enumerate(slots,1):
            rows.append({"game_id":f"G{i}","season":2019,"week":i,
                         "gameday":day,"gametime":clock,
                         "home_team":"AAA","away_team":"BBB",
                         "home_win":float(i%2==1),"market_home_prob":.65})
        return slot_labels(pd.DataFrame(rows))

    def test_current_outcome_never_changes_own_history(self):
        g=self.example()
        a=add_prior_team_slot_history(g)
        g2=g.copy()
        g2.loc[g2.game_id.eq("G3"),"home_win"]=0.
        b=add_prior_team_slot_history(g2)
        for col in ("prior_slot_surprise_diff","prior_slot_experience_diff"):
            self.assertAlmostEqual(float(a.loc[a.game_id.eq("G3"),col].iloc[0]),
                                   float(b.loc[b.game_id.eq("G3"),col].iloc[0]))
        self.assertNotAlmostEqual(
            float(a.loc[a.game_id.eq("G4"),"prior_slot_surprise_diff"].iloc[0]),
            float(b.loc[b.game_id.eq("G4"),"prior_slot_surprise_diff"].iloc[0])
        )

    def test_same_slot_prior_count(self):
        x=add_prior_team_slot_history(self.example())
        self.assertAlmostEqual(float(x.loc[x.game_id.eq("G1"),"home_prior_slot_experience"].iloc[0]),0)
        self.assertGreater(float(x.loc[x.game_id.eq("G3"),"home_prior_slot_experience"].iloc[0]),0)

    def test_equal_picks_no_disagreement(self):
        x=pd.DataFrame({"season":[2022]*4,"home_win":[0,1,0,1],
                        "p_one":[.3,.8,.9,.7],"p_two":[.3,.8,.9,.7]})
        out=paired(x,"one","two")
        self.assertEqual(out["net_correct"],0)
        self.assertEqual(out["flip_wins"],0)
        self.assertEqual(out["mcnemar_p"],1.0)


if __name__=="__main__":
    unittest.main()
