"""Evaluate saved detectors against source annotations; annotations never feed inference."""
import json,sys
from pathlib import Path
import cv2,numpy as np,pandas as pd
from scipy.optimize import linear_sum_assignment
from ultralytics import YOLO
import torch
ROOT=Path(__file__).resolve().parents[1]
out=ROOT/'validation/tactical';out.mkdir(exist_ok=True)
df=pd.read_csv(ROOT/'archive/top_view/annotations/D_20220220_1_0000_0030.csv',header=[0,1,2],index_col=0)
cap=cv2.VideoCapture(str(ROOT/'input_videos/clip_01_0-12s.mp4'))
frames=[];indices=list(range(0,360,30))
for i in indices:
 cap.set(0 if False else cv2.CAP_PROP_POS_FRAMES,i);ok,f=cap.read();frames.append(f)
cap.release()
print('CUDA',torch.cuda.is_available(),flush=True)
results={}
for name in ['best_topview.pt','best_topdown_noball.pt']:
 model=YOLO(ROOT/'models'/name); print(name,model.names,flush=True)
 predictions=[];matched=0;gt_count=0;pred_count=0;ball_hits=0;ball_gt=0
 for fi,frame in zip(indices,frames):
  result=model.predict(frame,imgsz=1280,conf=.15,verbose=False)[0]
  boxes=result.boxes.xyxy.cpu().numpy();classes=result.boxes.cls.cpu().numpy();scores=result.boxes.conf.cpu().numpy()
  entries=[{'bbox':b.tolist(),'class':model.names[int(c)],'confidence':float(s)} for b,c,s in zip(boxes,classes,scores)]
  predictions.append({'frame':fi,'detections':entries})
  gt=[]
  row=df.loc[fi+1]
  for team in ['0','1']:
   for pid in map(str,range(11)):
    a=[row[(team,pid,k)] for k in ['bb_left','bb_top','bb_width','bb_height']]
    if np.isfinite(a).all():gt.append([a[0]+a[2]/2,a[1]+a[3]/2])
  pred=[[(d['bbox'][0]+d['bbox'][2])/2,(d['bbox'][1]+d['bbox'][3])/2] for d in entries if d['class']=='player']
  if pred and gt:
   distances=np.linalg.norm(np.array(gt)[:,None]-np.array(pred)[None],axis=2);rr,cc=linear_sum_assignment(distances);matched+=int((distances[rr,cc]<=35).sum())
  pred_count+=len(pred);gt_count+=len(gt)
  a=[row[('BALL','BALL',k)] for k in ['bb_left','bb_top','bb_width','bb_height']]
  if np.isfinite(a).all():
   ball_gt+=1;centre=np.array([a[0]+a[2]/2,a[1]+a[3]/2]);ball_hits+=int(any(np.linalg.norm(centre-np.array([(d['bbox'][0]+d['bbox'][2])/2,(d['bbox'][1]+d['bbox'][3])/2]))<=20 for d in entries if d['class']=='ball'))
 results[name]={'classes':model.names,'frames':len(frames),'matched':matched,'gt':gt_count,'predicted':pred_count,'player_precision_35px':matched/pred_count if pred_count else 0,'player_recall_35px':matched/gt_count,'ball_recall_20px':ball_hits/ball_gt if ball_gt else None,'predictions':predictions}
 print(name,{k:v for k,v in results[name].items() if k!='predictions'},flush=True)
(out/'detector-evaluation.json').write_text(json.dumps(results,indent=2))
