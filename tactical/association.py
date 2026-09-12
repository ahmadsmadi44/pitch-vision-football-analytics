"""Associate top-view centres, using velocity and team evidence rather than box overlap."""
import numpy as np
from scipy.optimize import linear_sum_assignment

def reassociate(frames,assignments,max_gap=1.0):
    states={};next_id=1
    for frame in frames:
        now=frame['time'];players=frame['players'];active=[key for key,s in states.items() if now-s['time']<=max_gap]
        costs=np.full((len(active),len(players)),1e6)
        for a,key in enumerate(active):
            s=states[key];dt=now-s['time'];pred=s['position']+s['velocity']*dt
            for b,p in enumerate(players):
                if s['team'] and p['team'] and s['team']!=p['team']:continue
                d=np.linalg.norm(np.array([p['x'],p['y']])-pred)
                if d<=1.2+8*dt:costs[a,b]=d
        matched={}
        if costs.size:
            rr,cc=linear_sum_assignment(costs)
            matched={c:active[r] for r,c in zip(rr,cc) if costs[r,c]<1e5}
        for b,p in enumerate(players):
            source=p['id'];key=matched.get(b)
            if key is None:
                key=str(next_id);next_id+=1
                states[key]={'position':np.array([p['x'],p['y']]),'velocity':np.zeros(2),'time':now,'team':p['team'],'sources':set(),'confidence':[],'role':p.get('role'),'reviewed':False}
            s=states[key];position=np.array([p['x'],p['y']]);dt=now-s['time']
            if dt>0:
                v=(position-s['position'])/dt
                if np.linalg.norm(v)<12.5:s['velocity']=.5*s['velocity']+.5*v
            if p['team'] is not None:s['team']=p['team']
            if p.get('role')=='goalkeeper':s['role']='goalkeeper'
            if assignments.get(source,{}).get('method')=='reviewed goalkeeper override':s['reviewed']=True
            s['position']=position;s['time']=now;s['sources'].add(source);s['confidence'].append(p.get('teamConfidence',0))
            p['sourceId']=source;p['id']=key;p['team']=s['team'];p['role']=s['role']
    result={key:{'team':s['team'],'confidence':round(float(np.mean(s['confidence'])),3),'sourceTrackIds':sorted(s['sources'],key=int),'method':'reviewed goalkeeper override' if s['reviewed'] else 'appearance and motion association','role':s['role']} for key,s in states.items()}
    return result
