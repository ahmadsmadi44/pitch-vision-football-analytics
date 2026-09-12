import json,cv2,pandas as pd,numpy as np
from pathlib import Path
from scipy.optimize import linear_sum_assignment
import argparse
parser=argparse.ArgumentParser();parser.add_argument('--run',default='validation/tactical');parser.add_argument('--annotations',default='archive/top_view/annotations/D_20220220_1_0000_0030.csv');args=parser.parse_args()
root=Path(args.run);raw=json.loads((root/'detections.json').read_text());data=json.loads((root/'tactical.json').read_text());df=pd.read_csv(args.annotations,header=[0,1,2],index_col=0)
H=cv2.getPerspectiveTransform(np.float32(data['calibration']['nativeCorners']),np.float32([[0,0],[105,0],[105,68],[0,68]]))
correct=0;matched=0;assigned=0;total_gt=0;total_pred=0;ballerrors=[];ballcandidates=0;gtposes=[];idtruth={}
for rec,frame in zip(raw['frames'],data['frames']):
 row=df.loc[rec['frame']+1];gt=[];labels=[]
 for team in ['0','1']:
  for pid in map(str,range(11)):
   a=[row[(team,pid,k)] for k in ['bb_left','bb_top','bb_width','bb_height']]
   if np.isfinite(a).all():gt.append([a[0]+a[2]/2,a[1]+a[3]/2]);labels.append((int(team)+1,pid))
 ps=rec['players'];total_gt+=len(gt);total_pred+=len(ps);centres=np.array([[(p['bbox'][0]+p['bbox'][2])/2,(p['bbox'][1]+p['bbox'][3])/2] for p in ps]);dist=np.linalg.norm(np.array(gt)[:,None]-centres[None],axis=2);rr,cc=linear_sum_assignment(dist)
 for r,c in zip(rr,cc):
  if dist[r,c]>35:continue
  matched+=1;pred=next((p['team'] for p in frame['players'] if p.get('sourceId',p['id'])==ps[c]['id']),None);idtruth.setdefault(ps[c]['id'],[]).append(labels[r][0])
  if pred is not None:assigned+=1;correct+=pred==labels[r][0]
 a=[row[('BALL','BALL',k)] for k in ['bb_left','bb_top','bb_width','bb_height']]
 if np.isfinite(a).all():
  center=np.float32([[[a[0]+a[2]/2,a[1]+a[3]/2]]]);reg=np.array(rec['registration']);stabilized=cv2.perspectiveTransform(center/2,reg)*2;pos=cv2.perspectiveTransform(stabilized,H)[0,0];gtposes.append((rec['time'],pos.tolist()))
  if frame['ball']:ballerrors.append(float(np.linalg.norm(pos-frame['ball'])))
  ballcandidates+=any(np.linalg.norm(center[0,0]-np.array([(b['bbox'][0]+b['bbox'][2])/2,(b['bbox'][1]+b['bbox'][3])/2]))<20 for b in rec['ballCandidates'])
print('matches',matched,'assigned',assigned,'correct',correct,'acc',correct/assigned,'ball candidate recall',ballcandidates/len(data['frames']),'ballerrors median',np.median(ballerrors),'within1m',np.mean(np.array(ballerrors)<1))
print('Median ball error and team assignments evaluated against source annotations, not used as model inputs.')
(root/'quality-audit.json').write_text(json.dumps({'matched':matched,'groundTruthPlayers':total_gt,'predictedPlayers':total_pred,'playerRecall35px':matched/total_gt,'playerPrecision35px':matched/total_pred,'ballRetainedFrames':len(ballerrors),'ballRetainedCoverage':len(ballerrors)/len(data['frames']),'assigned':assigned,'correct':correct,'teamAccuracy':correct/assigned,'ballCandidateRecall':ballcandidates/len(data['frames']),'ballMedianError':float(np.median(ballerrors)),'ballWithin1m':float(np.mean(np.array(ballerrors)<1))},indent=2))
