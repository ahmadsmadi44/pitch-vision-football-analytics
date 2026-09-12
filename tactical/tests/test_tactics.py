import sys,unittest
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from analysis import analyze,shape,edges,lanes
from ball_path import select_ball_path

class TacticsTests(unittest.TestCase):
 def players(self,team=1,x=25):return [{'id':f'{team}-{i}','team':team,'x':x+(i%3)*2,'y':15+i*4,'vx':0,'vy':0} for i in range(9)]
 def test_block_is_relative_to_defended_goal(self):
  self.assertEqual(shape(self.players(x=20),1,1,None)['block'],'low')
  self.assertEqual(shape(self.players(x=50),1,1,None)['block'],'mid')
  self.assertEqual(shape(self.players(x=75),1,1,None)['block'],'high')
  self.assertEqual(shape(self.players(team=2,x=80),2,-1,None)['block'],'low')
 def test_missing_players_gates_shape(self):self.assertEqual(shape(self.players()[:3],1,1,None)['block'],'unknown')
 def test_proximity_alone_is_not_pressing(self):
  ps=self.players();ball=[29,15]
  self.assertFalse(shape(ps,1,1,ball)['pressers'])
  ps[0]['vx']=3
  self.assertEqual(shape(ps,1,1,ball)['pressers'][0]['id'],'1-0')
 def test_edges_do_not_cross_teams(self):
  for a,b in edges(self.players()+self.players(team=2)):self.assertEqual(a.split('-')[0],b.split('-')[0])
 def test_blocked_lane(self):
  ps=[{'id':'target','team':1,'x':20,'y':10},{'id':'defender','team':2,'x':10,'y':10}]
  self.assertFalse(lanes(ps,1,[0,10])[0]['open'])
 def test_missing_ball_means_unknown_no_tactical_claim(self):
  frames=[{'time':i/10,'players':self.players()+self.players(team=2,x=70),'ball':None} for i in range(5)]
  result=analyze({'frames':frames,'duration':.5})
  self.assertEqual(result['summary']['possessionCoverage'],0)
  self.assertTrue(all(f['phases']['1']=='unknown' for f in frames))
  self.assertEqual(result['events'],[])
 def test_turnover_requires_sustained_assignment(self):
  frames=[]
  for i in range(16):
   ps=self.players()+self.players(team=2,x=70)
   frames.append({'time':i/10,'players':ps,'ball':[25,15] if i<8 else [70,15]})
  result=analyze({'frames':frames,'duration':1.6})
  changes=[e for e in result['events'] if e['type']=='turnover']
  self.assertEqual(len(changes),1);self.assertEqual(changes[0]['team'],2)
  self.assertEqual(frames[-1]['phases']['1'],'defensive transition')
 def test_static_ball_distractor_is_rejected(self):
  frames=[{'time':i/10,'candidates':[{'position':[5,5],'confidence':.95,'method':'full-frame'},{'position':[30+i*.3,30],'confidence':.5,'method':'full-frame'}]} for i in range(12)]
  select_ball_path(frames)
  self.assertTrue(all(f['ball'][0]>20 for f in frames))
if __name__=='__main__':unittest.main()

class AssociationTests(unittest.TestCase):
 def test_motion_association_survives_source_id_changes_and_team_crossing(self):
  from association import reassociate
  frames=[]
  for i in range(12):
   frames.append({'time':i/10,'players':[{'id':str(i*2+1),'team':1,'teamConfidence':.9,'x':20+i*.2,'y':30},{'id':str(i*2+2),'team':2,'teamConfidence':.9,'x':22-i*.2,'y':30}]})
  identities=reassociate(frames,{})
  self.assertEqual(len(identities),2)
  self.assertEqual(len({f['players'][0]['id'] for f in frames}),1)
  self.assertEqual(len({f['players'][1]['id'] for f in frames}),1)
