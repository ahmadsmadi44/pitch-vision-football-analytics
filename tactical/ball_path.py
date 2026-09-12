"""Offline ball association. Avoid selecting penalty spots just by confidence."""
import numpy as np

def select_ball_path(frames):
    tracks=[]
    for i,f in enumerate(frames):
        used=set()
        for c in sorted(f['candidates'],key=lambda c:-c['confidence']):
            point=np.array(c['position']);choices=[]
            for k,t in enumerate(tracks):
                if k in used:continue
                last=t[-1];dt=f['time']-last['time']
                if not 0<dt<=.5:continue
                velocity=np.zeros(2)
                if len(t)>1:
                    before=t[-2];velocity=(np.array(last['point'])-before['point'])/(last['time']-before['time'])
                    if np.linalg.norm(velocity)>35:continue
                residual=np.linalg.norm(point-(last['point']+velocity*dt))
                speed=np.linalg.norm(point-last['point'])/dt
                if speed<=35 and residual<=(max(1.2,dt*8) if len(t)>1 else 35*dt+.6):choices.append((residual,k))
            if choices:k=min(choices)[1]
            else:k=len(tracks);tracks.append([])
            tracks[k].append({'i':i,'time':f['time'],'point':point,'candidate':c});used.add(k)
    # A stationary mark has negligible spatial extent. Actual stationary ball can
    # remain in a confirmed moving trajectory, but isn't invented when isolated.
    valid=[]
    for t in tracks:
        pts=np.array([p['point'] for p in t]);extent=float(np.linalg.norm(np.ptp(pts,axis=0)))
        if len(t)>=6 and extent>=.8:valid.append(t)
    winners={}
    for t in sorted(valid,key=lambda t:len(t)):
        for p in t:
            if p['i'] not in winners or len(t)>winners[p['i']][0]:winners[p['i']]=(len(t),p)
    for i,f in enumerate(frames):
        w=winners.get(i);f['ball']=w[1]['point'].tolist() if w else None;f['ballSource']=w[1]['candidate']['method'] if w else 'missing'
        f.pop('candidates',None)
