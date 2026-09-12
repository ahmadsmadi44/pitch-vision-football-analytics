import sys
import unittest
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from speed_and_distance_estimator.speed_and_distance_estimator import SpeedAndDistanceEstimator
from player_stats import build_player_stats
from player_rating.player_rating import PlayerRatingModel
from ball_gaps import fill_ball_gaps

class AnalyticsTests(unittest.TestCase):
    def tracks(self, xs):
        return {'players': [{1:{'team':1, 'position_transformed':[x,0]}} if x is not None else {} for x in xs]}

    def test_constant_motion_and_fps(self):
        tracks=self.tracks([i*.2 for i in range(26)])
        SpeedAndDistanceEstimator(fps=25).add_speed_and_distance(tracks)
        speeds=[p[1]['speed'] for p in tracks['players']]
        self.assertAlmostEqual(speeds[10],18)
        self.assertAlmostEqual(tracks['players'][-1][1]['distance'],4.6)
        stats=build_player_stats(tracks,25)['1']
        self.assertAlmostEqual(stats['observed_seconds'],1.04)
        self.assertLessEqual(stats['peak_speed_kmh'],45)

    def test_rejects_jump_and_clears_stale_values(self):
        tracks=self.tracks([i*10 for i in range(11)])
        tracks['players'][0][1]['speed']=300
        estimator=SpeedAndDistanceEstimator(fps=25)
        estimator.add_speed_and_distance(tracks)
        self.assertEqual(estimator.rejected_windows,2)
        self.assertTrue(all('speed' not in f[1] for f in tracks['players']))
        self.assertIsNone(build_player_stats(tracks,25)['1']['peak_speed_kmh'])

    def test_no_motion_across_missing_track(self):
        tracks=self.tracks([0,0,None,0,0,1])
        SpeedAndDistanceEstimator(fps=25).add_speed_and_distance(tracks)
        self.assertFalse(any('speed' in p for f in tracks['players'] for p in f.values()))

    def test_ball_gaps_are_bounded(self):
        known=lambda x:{1:{'bbox':[x,0,x+1,1]}}
        result=fill_ball_gaps([{},known(0),{},known(2),{},{},{},known(6),{}],1)
        self.assertEqual(result[2][1]['bbox'][0],1)
        for i in [0,4,5,6,8]: self.assertEqual(result[i],{})

    def test_possession_seconds_not_touches_and_unknown_not_recovery(self):
        tracks=self.tracks([0]*5)
        for i in [0,1,4]: tracks['players'][i][1]['has_ball']=True
        stats=build_player_stats(tracks,25)['1']
        self.assertAlmostEqual(stats['possession_seconds'],.12)
        self.assertEqual(stats['possession_sequences'],2)
        raw=PlayerRatingModel().compute_raw_stats(tracks,[1,1,0,2,1])
        self.assertEqual(raw[1]['ball_recoveries'],1)

if __name__=='__main__': unittest.main()
