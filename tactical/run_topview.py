"""Top-view detection, stable appearance teams and calibrated tactical export."""
import argparse,json,sys,math
from pathlib import Path
import cv2,numpy as np
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'stage4_local'))
from teams import descriptor,assign_teams

def main():
 p=argparse.ArgumentParser();p.add_argument('--clip',default='input_videos/clip_01_0-12s.mp4');p.add_argument('--out',default='validation/tactical');p.add_argument('--reuse',action='store_true');p.add_argument('--stride',type=int,default=3);p.add_argument('--team-overrides');args=p.parse_args()
 out=ROOT/args.out;out.mkdir(parents=True,exist_ok=True)
 cap=cv2.VideoCapture(str(ROOT/args.clip));fps=cap.get(cv2.CAP_PROP_FPS);n=int(cap.get(cv2.CAP_PROP_FRAME_COUNT));w=cap.get(3);h=cap.get(4)
 if not cap.isOpened():raise ValueError('Cannot open clip')
 ok,first=cap.read();cap.set(cv2.CAP_PROP_POS_FRAMES,0)
 # Manually checked against full-frame field markings, native 3840x2160.
 corners=np.float32([[465,147],[3408,103],[3396,2020],[498,2013]])*np.float32([w/3840,h/2160])
 H=cv2.getPerspectiveTransform(corners,np.float32([[0,0],[105,0],[105,68],[0,68]]))
 orb=cv2.ORB_create(nfeatures=2500);small=cv2.resize(first,(1920,1080));gray=cv2.cvtColor(small,cv2.COLOR_BGR2GRAY);kp0,des0=orb.detectAndCompute(gray,None)
 if args.reuse:
  raw=json.loads((out/'detections.json').read_text())
  if raw['clip']!=args.clip or raw['frameCount']!=n:raise ValueError('Cache clip mismatch')
 else:
  from ultralytics import YOLO
  import supervision as sv
  model=YOLO(ROOT/'models/best_topview.pt');tracker=sv.ByteTrack(frame_rate=round(fps/args.stride),lost_track_buffer=45,track_activation_threshold=.25)
  raw={'clip':args.clip,'fps':fps,'frameCount':n,'stride':args.stride,'frames':[]};matcher=cv2.BFMatcher(cv2.NORM_HAMMING)
  for fi in range(n):
   ok,frame=cap.read()
   if not ok:break
   if fi%args.stride:continue
   current=cv2.resize(frame,(1920,1080));kp,des=orb.detectAndCompute(cv2.cvtColor(current,cv2.COLOR_BGR2GRAY),None)
   pairs=matcher.knnMatch(des,des0,k=2) if des is not None else []
   matches=[a for pair in pairs if len(pair)==2 for a,b in [pair] if a.distance<.7*b.distance]
   reg=None
   if len(matches)>=12:
    src=np.float32([kp[m.queryIdx].pt for m in matches]);dst=np.float32([kp0[m.trainIdx].pt for m in matches]);reg,inliers=cv2.findHomography(src,dst,cv2.RANSAC,2.5)
    if inliers is None or inliers.sum()<12:reg=None
   if fi==0:reg=np.eye(3)
   result=model.predict(frame,imgsz=1280,conf=.15,verbose=False)[0]
   detection=sv.Detections.from_ultralytics(result);players=detection[detection.class_id==0]
   # Remove off-field detections before association, retaining original pixel boxes.
   centers=(players.xyxy[:,:2]+players.xyxy[:,2:])/2
   keep=np.array([cv2.pointPolygonTest(corners,tuple(map(float,c)),False)>=0 for c in centers],dtype=bool)
   players=tracker.update_with_detections(players[keep]);entries=[]
   for box,confidence,tid in zip(players.xyxy,players.confidence,players.tracker_id):entries.append({'id':str(int(tid)),'bbox':box.tolist(),'confidence':float(confidence)})
   balls=[{'bbox':b.tolist(),'confidence':float(s),'method':'full-frame'} for b,s in zip(detection.xyxy[detection.class_id==1],detection.confidence[detection.class_id==1])]
   # Small-ball rescue: overlapping native-resolution tiles, still the same YOLO model.
   if not balls:
    for x,y in [(0,0),(int(w*.4),0),(0,int(h*.35)),(int(w*.4),int(h*.35))]:
     tile=frame[y:min(int(h),y+int(h*.65)),x:min(int(w),x+int(w*.6))]
     r=model.predict(tile,imgsz=1280,conf=.15,verbose=False)[0]
     for b,c,s in zip(r.boxes.xyxy.cpu().numpy(),r.boxes.cls.cpu().numpy(),r.boxes.conf.cpu().numpy()):
      if int(c)==1:balls.append({'bbox':(b+np.array([x,y,x,y])).tolist(),'confidence':float(s),'method':'tiled'})
   raw['frames'].append({'frame':fi,'time':fi/fps,'players':entries,'ballCandidates':balls,'registration':reg.tolist() if reg is not None else None})
   if fi%30==0:print('Detected',fi,'/',n,'players',len(entries),'balls',len(balls),flush=True)
  (out/'detections.json').write_text(json.dumps(raw))
 cap.release()
 # Sample appearance at full resolution, not the downsampled detector image.
 samples={};cap=cv2.VideoCapture(str(ROOT/args.clip))
 for record in raw['frames'][::3]:
  cap.set(cv2.CAP_PROP_POS_FRAMES,record['frame']);ok,frame=cap.read()
  if not ok:continue
  for pl in record['players']:
   f=descriptor(frame,pl['bbox'])
   if f is not None:samples.setdefault(pl['id'],[]).append(f)
 cap.release();teams,quality=assign_teams(samples)
 if args.team_overrides:
  overrides=json.loads((ROOT/args.team_overrides).read_text())
  for tid,team in overrides.items():
   if team not in [1,2,None] or tid not in teams:raise ValueError('Invalid override')
   teams[tid].update(team=team,method='reviewed goalkeeper override',role='goalkeeper')
 frames=[];last_ball=None;last_ball_time=-10
 for rec in raw['frames']:
  reg=np.array(rec['registration']) if rec['registration'] is not None else None
  def project(box):
   if reg is None:return None
   center=np.float32([[(box[0]+box[2])/2,(box[1]+box[3])/2]])
   center*=1920/w
   stabilized=cv2.perspectiveTransform(center[None],reg)[0]*(w/1920)
   pos=cv2.perspectiveTransform(stabilized[None],H)[0,0]
   if not np.isfinite(pos).all() or not (0<=pos[0]<=105 and 0<=pos[1]<=68):return None
   return [round(float(v),3) for v in pos]
  players=[]
  for pl in rec['players']:
   pos=project(pl['bbox']);assignment=teams.get(pl['id'],{})
   if pos is not None:players.append({'id':pl['id'],'team':assignment.get('team'),'teamConfidence':assignment.get('confidence',0),'role':assignment.get('role','outfield'),'x':pos[0],'y':pos[1]})
  # Ball detections inside a player's central body region are common false positives.
  def body_overlap(candidate):
   b=candidate['bbox'];x,y=(b[0]+b[2])/2,(b[1]+b[3])/2
   return any(a['bbox'][0]+.1*(a['bbox'][2]-a['bbox'][0])<x<a['bbox'][2]-.1*(a['bbox'][2]-a['bbox'][0]) and a['bbox'][1]+.1*(a['bbox'][3]-a['bbox'][1])<y<a['bbox'][3]-.1*(a['bbox'][3]-a['bbox'][1]) for a in rec['players'])
  rec_candidates=[b for b in rec['ballCandidates'] if not body_overlap(b)]
  candidates=[(project(b['bbox']),b) for b in rec_candidates];candidates=[(pos,b) for pos,b in candidates if pos is not None]
  if last_ball is not None and rec['time']-last_ball_time<.5:
   candidates=[(pos,b) for pos,b in candidates if np.linalg.norm(np.array(pos)-last_ball)<=35*(rec['time']-last_ball_time)+1]
  chosen=max(candidates,key=lambda pair:pair[1]['confidence']) if candidates else None
  ball=chosen[0] if chosen else None
  if ball is not None:last_ball=np.array(ball);last_ball_time=rec['time']
  frames.append({'frame':rec['frame'],'time':rec['time'],'players':players,'ball':ball,'ballSource':chosen[1]['method'] if chosen else 'missing','registrationValid':reg is not None,'candidates':[{'position':pos,'confidence':b['confidence'],'method':b['method']} for pos,b in [(project(x['bbox']),x) for x in rec_candidates] if pos is not None]})
 # Reject static paint/spare-ball hypotheses: require a coherent moving segment.
 from ball_path import select_ball_path
 select_ball_path(frames)
 # Bounded interpolation only, capped at 0.3 s and plausible travel distance.
 known=[i for i,f in enumerate(frames) if f['ball'] is not None]
 for a,b in zip(known,known[1:]):
  dt=frames[b]['time']-frames[a]['time']
  if b-a>1 and dt<=.31 and np.linalg.norm(np.array(frames[b]['ball'])-frames[a]['ball'])/dt<=35:
   for i in range(a+1,b):
    frac=(frames[i]['time']-frames[a]['time'])/dt;frames[i]['ball']=(np.array(frames[a]['ball'])*(1-frac)+np.array(frames[b]['ball'])*frac).tolist();frames[i]['ballSource']='interpolated'
 from association import reassociate
 teams=reassociate(frames,teams)
 # One-second velocity baseline avoids interpreting detector jitter as pressing.
 for i,f in enumerate(frames):
  prior=frames[max(0,i-round(fps/args.stride))];dt=f['time']-prior['time'];old={p['id']:p for p in prior['players']}
  for pl in f['players']:
   before=old.get(pl['id']);v=np.array([pl['x']-before['x'],pl['y']-before['y']])/dt if before and dt>=.5 else None
   pl['vx']=round(float(v[0]),3) if v is not None and np.linalg.norm(v)<12.5 else None;pl['vy']=round(float(v[1]),3) if v is not None and np.linalg.norm(v)<12.5 else None
 from analysis import analyze
 payload={'schemaVersion':1,'clip':args.clip,'fps':fps,'stride':args.stride,'duration':n/fps,'pitch':{'length':105,'width':68,'dimensionsVerified':False},'teams':[{'id':1,'name':'White','color':'#efeee1','attacks':1},{'id':2,'name':'Blue','color':'#53a9dc','attacks':-1}],'teamAssignments':teams,'teamMethod':quality,'frames':frames,'source':'model','calibration':{'nativeCorners':corners.tolist(),'method':'frame-to-reference ORB homography, then pitch homography','dimensions':'assumed 105 × 68 m'}}
 payload=analyze(payload);(out/'tactical.json').write_text(json.dumps(payload,allow_nan=False));print('Exported',out/'tactical.json',flush=True)
if __name__=='__main__':main()
