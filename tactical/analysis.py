"""Transparent tactical heuristics over measured coordinates; no learned win model."""
import math
from collections import Counter
import numpy as np

def distance(a,b):return math.hypot(a['x']-b[0],a['y']-b[1])
def edges(players):
    found=set()
    for a in players:
        same=sorted([p for p in players if p['team']==a['team'] and p['id']!=a['id']],key=lambda p:distance(p,(a['x'],a['y'])))
        for b in same[:2]:found.add(tuple(sorted([a['id'],b['id']])))
    return [list(edge) for edge in sorted(found)]

def zone(ball):
    if ball is None:return 'unknown'
    return ('left' if ball[1]<68/3 else 'right' if ball[1]>68*2/3 else 'central')

def shape(players,team,attack,ball):
    ps=[p for p in players if p['team']==team and p.get('role')!='goalkeeper']
    if len(ps)<7:return {'count':len(ps),'block':'unknown','reason':'Fewer than seven assigned players','pressers':[]}
    xy=np.array([[p['x'],p['y']] for p in ps]);own=xy[:,0] if attack==1 else 105-xy[:,0]
    # Exclude the deepest isolated player as a goalkeeper candidate only for block depth.
    order=np.argsort(own);outfield=order[1:] if own[order[1]]-own[order[0]]>12 else order
    height=float(np.median(own[outfield]));width=float(np.quantile(xy[outfield,1],.9)-np.quantile(xy[outfield,1],.1));depth=float(np.quantile(own[outfield],.9)-np.quantile(own[outfield],.1))
    label='low' if height<105*.36 else 'high' if height>105*.62 else 'mid'
    pressers=[];near=0
    if ball is not None:
        for p in ps:
            d=distance(p,ball)
            if d<=8:near+=1
            if p.get('vx') is None or d==0:continue
            closing=(p['vx']*(ball[0]-p['x'])+p['vy']*(ball[1]-p['y']))/d
            if d<12 and closing>1.5:pressers.append({'id':p['id'],'distance':round(d,1),'closingSpeed':round(closing,1)})
    return {'count':len(ps),'block':label,'height':round(height,1),'width':round(width,1),'depth':round(depth,1),'center':[round(float(v),2) for v in xy[outfield].mean(axis=0)],'pressers':pressers,'nearBall':near}

def lanes(players,owner,ball):
    if owner is None or ball is None:return []
    attackers=[p for p in players if p['team']==owner];defenders=[p for p in players if p['team'] not in [None,owner]]
    result=[]
    for p in attackers:
        end=np.array([p['x'],p['y']]);start=np.array(ball);delta=end-start;length=float(np.linalg.norm(delta))
        if length<3:continue
        clearances=[]
        for d in defenders:
            point=np.array([d['x'],d['y']]);t=float(np.dot(point-start,delta)/(length*length))
            if 0<t<1:clearances.append(float(np.linalg.norm(point-(start+t*delta))))
        clearance=min(clearances) if clearances else 30
        result.append({'id':p['id'],'x':p['x'],'y':p['y'],'clearance':round(clearance,1),'length':round(length,1),'open':clearance>=2})
    return sorted(result,key=lambda p:-p['clearance'])

def space_grid(players):
    assigned=[p for p in players if p['team'] in [1,2]]
    if not assigned:return []
    grid=[]
    for y in np.linspace(2.125,65.875,16):
        row=[]
        for x in np.linspace(2.1875,102.8125,24):
            nearest=min(assigned,key=lambda p:distance(p,(x,y)));row.append(nearest['team'])
        grid.append(row)
    return grid

def opportunities(players,owner,ball,attack):
    if owner is None or ball is None:return None
    defenders=[p for p in players if p['team'] in [1,2] and p['team']!=owner]
    if len(defenders)<7:return None
    candidates=[]
    for x in np.linspace(8,97,16):
        for y in [10,22,34,46,58]:
            if 5<(x-ball[0])*attack<30:
                clearance=min(distance(p,(x,y)) for p in defenders)
                candidates.append({'x':round(float(x),1),'y':y,'clearance':round(clearance,1)})
    return max(candidates,key=lambda p:p['clearance']) if candidates else None

def advice(frame,owner,defending,last_turnover):
    ball=frame['ball'];s=frame['shapes'].get(str(defending),{}) if defending else {};block=s.get('block','unknown')
    if owner is None:return [{'kind':'quality','title':'Possession is unassigned','evidence':'No sustained, unambiguous ball proximity in this sample.','suggestion':'Review the source before interpreting pressing or attacking options.'}]
    if ball is None:return [{'kind':'quality','title':'Ball position is temporarily unavailable','evidence':'Team possession is carried briefly from the last observed touch, but no ball coordinate is available in this frame.','suggestion':'Use the phase label for continuity; wait for a detected ball before interpreting press distance or passing lanes.'}]
    if block=='unknown':return [{'kind':'quality','title':'The defensive shape is incomplete','evidence':s.get('reason','Insufficient evidence'),'suggestion':'Use the source view to inspect missing or uncertain team assignments.'}]
    rules={
     'low':('Stretch the low block', 'Use width and a quick return pass to move the block before attacking a gap. Look for cutback access after drawing a wide defender out.'),
     'mid':('Move the mid block before playing through it','Circulate outside the compact shape, then look for a receiving option between lines when a midfielder steps out.'),
     'high':('Explore space behind the high block','Look for a supported third-player release or a run behind the advanced line. Check timing and offside before choosing a direct pass.')}
    title,suggestion=rules[block]
    result=[{'kind':'attack','title':title,'evidence':f"Defending unit median: {s['height']} m from its own goal; width {s['width']} m; depth {s['depth']} m.",'suggestion':suggestion}]
    if s['pressers']:
        ids=', '.join(p['id'] for p in s['pressers'])
        best=next((p for p in frame['lanes'] if p['open']),None)
        result.append({'kind':'attack','title':'A closing-down movement is visible','evidence':f"Track(s) {ids} are within 12 m and moving toward the ball faster than 1.5 m/s.",'suggestion':f"Consider a release toward track {best['id']} ({best['clearance']} m geometric lane clearance), then move beyond the presser." if best else 'Use a supporting return pass or a third-player combination; no clear geometric outlet was found.'})
    near_attack=sum(p['team']==owner and distance(p,ball)<=8 for p in frame['players']);near_def=s.get('nearBall',0)
    if last_turnover is not None and frame['time']-last_turnover<3 and near_def>=near_attack and near_def>=2:
        result.append({'kind':'defend','title':'A local counter-press opportunity','evidence':f"Within 8 m of the ball: {near_def} defenders and {near_attack} attackers, immediately after a turnover.",'suggestion':'The closest player can pressure the ball while support covers the short exits. This is a candidate action, not a predicted recovery.'})
    else:
        result.append({'kind':'defend','title':'Protect the centre and coordinate the press','evidence':f"Within 8 m: {near_def} defenders versus {near_attack} attackers; {len(s['pressers'])} closing-down candidates.",'suggestion':'Keep covering support behind the first presser. If support is outnumbered, recover the block rather than chasing alone; use the touchline to restrict exits when the ball goes wide.'})
    return result

def analyze(payload):
    frames=payload['frames'];attack_by={team['id']:team.get('attacks',1 if team['id']==1 else -1) for team in payload.get('teams',[])};candidate=None;candidate_since=0;owner=None;previous_owner=None;previous_known=-10;turnover_time=None;events=[];last_switch=-10
    for i,f in enumerate(frames):
        assigned=[p for p in f['players'] if p['team'] in [1,2]];ball=f['ball'];near=sorted(assigned,key=lambda p:distance(p,ball)) if ball else []
        raw=None
        if near and distance(near[0],ball)<=4.0:
            other=next((p for p in near if p['team']!=near[0]['team']),None)
            if other is None or distance(other,ball)-distance(near[0],ball)>=.75:raw=near[0]['team']
        if raw!=candidate:candidate=raw;candidate_since=f['time']
        direct_owner=raw if raw is not None and f['time']-candidate_since>=.2 else None
        if direct_owner is not None:
            owner=direct_owner
            if previous_owner is not None and owner!=previous_owner and f['time']-previous_known<=1.8:
                turnover_time=f['time'];events.append({'type':'turnover','time':f['time'],'frameIndex':i,'team':owner,'title':f'Team {owner} gains sustained proximity'})
            previous_owner=owner;previous_known=f['time']
            f['possessionSource']='sustained player proximity'
        elif previous_owner is not None and f['time']-previous_known<=1.2:
            # Preserve the last directly observed team briefly while a pass is
            # travelling or the tiny ball is missed. Longer gaps stay unknown.
            owner=previous_owner
            f['possessionSource']='bounded carry from last observed touch'
        else:
            owner=None
            f['possessionSource']='unassigned'
        f['possessionTeam']=owner
        transition=owner is not None and turnover_time is not None and f['time']-turnover_time<2
        f['phases']={str(t):'unknown' if owner is None else ('attacking transition' if t==owner else 'defensive transition') if transition else ('in possession' if t==owner else 'out of possession') for t in [1,2]}
        f['shapes']={str(t):shape(assigned,t,attack_by.get(t,1 if t==1 else -1),ball) for t in [1,2]}
        # A block label describes out-of-possession geometry only.
        for t in [1,2]:
            if owner is None or owner==t:f['shapes'][str(t)]['block']='not defending' if owner==t else 'unknown'
        f['edges']=edges(assigned);f['lanes']=lanes(assigned,owner,ball);f['space']=space_grid(assigned);f['openSpace']=opportunities(assigned,owner,ball,attack_by.get(owner,1 if owner==1 else -1))
        f['advice']=advice(f,owner,3-owner if owner else None,turnover_time)
        f['zone']=zone(ball);f['reaction']=None
        # Only describe observed ball movement, with uninterrupted assigned possession.
        if i and owner and ball:
            candidates=[(j,old) for j,old in enumerate(frames[:i]) if 1<=f['time']-old['time']<=2 and old.get('possessionTeam')==owner and old['ball']]
            if candidates:
                j,old=candidates[0];same=all(x.get('possessionTeam')==owner and x['ball'] for x in frames[j:i+1])
                dy=ball[1]-old['ball'][1];defending=str(3-owner);before=old['shapes'][defending].get('center');now=f['shapes'][defending].get('center')
                if same and abs(dy)>=8 and before and now:
                    shift=now[1]-before[1];f['reaction']={'ballShift':round(dy,1),'blockShift':round(shift,1),'duration':round(f['time']-old['time'],1),'direction':'right' if dy>0 else 'left','ratio':round(shift/dy,2)}
                    if f['time']-last_switch>2:
                        last_switch=f['time'];events.append({'type':'lateral move','time':f['time'],'frameIndex':i,'team':owner,'title':f"Ball moves {abs(dy):.1f} m {f['reaction']['direction']}; block shifts {abs(shift):.1f} m"})
        f['reactions']={}
        if ball and i:
            previous=[old for old in frames[:i] if 1<=f['time']-old['time']<=2 and old['ball']]
            if previous:
                old=previous[0];dy=ball[1]-old['ball'][1]
                if abs(dy)>=5:
                    for t in [1,2]:
                        before=old['shapes'][str(t)].get('center');now=f['shapes'][str(t)].get('center')
                        if before and now:
                            shift=now[1]-before[1]
                            f['reactions'][str(t)]={'ballShift':round(dy,1),'blockShift':round(shift,1),'duration':round(f['time']-old['time'],1),'direction':'right' if dy>0 else 'left','ratio':round(shift/dy,2)}
                    if f['reactions'] and f['time']-last_switch>2:
                        last_switch=f['time'];events.append({'type':'lateral move','time':f['time'],'frameIndex':i,'team':owner,'title':f"Ball moves {abs(dy):.1f} m {'right' if dy>0 else 'left'}; inspect each team's response"})
        if any(s['pressers'] for t,s in f['shapes'].items() if int(t)!=owner) and owner and (i==0 or not frames[i-1].get('pressActive')):
            events.append({'type':'press candidate','time':f['time'],'frameIndex':i,'team':3-owner,'title':'Closing-down movement starts'})
        f['pressActive']=bool(owner and f['shapes'][str(3-owner)]['pressers'])
    # Contiguous segments, explicit unknown spans. Endpoints are seconds, not frame counts.
    segments=[]
    for i,f in enumerate(frames):
        label='unknown' if f['possessionTeam'] is None else f"Team {f['possessionTeam']} possession"
        if any('transition' in phase for phase in f['phases'].values()):label=f"Team {f['possessionTeam']} transition"
        end=frames[i+1]['time'] if i+1<len(frames) else payload['duration']
        if segments and segments[-1]['label']==label:segments[-1]['end']=end
        else:segments.append({'start':f['time'],'end':end,'label':label,'frameIndex':i})
    known=sum(f['ball'] is not None for f in frames);poss=sum(f['possessionTeam'] is not None for f in frames)
    payload['events']=events;payload['segments']=segments
    payload['summary']={'frames':len(frames),'ballCoverage':known/len(frames) if frames else 0,'possessionCoverage':poss/len(frames) if frames else 0,'teamCounts':dict(Counter(str(v['team']) for v in payload.get('teamAssignments',{}).values())),'eventCount':len(events)}
    directions=', '.join(f"{team['name']} attacks {'+x' if attack_by.get(team['id'],1)>0 else '-x'}" for team in payload.get('teams',[]))
    payload['methodology']={'type':'rules-based descriptive analysis','blockThresholds':'Median outfield distance from own goal: low <36% of length, high >62%, otherwise mid; at least 7 assigned tracks. Not FIFA official classifications.','pressure':'Within 12 m, observed velocity toward current ball >1.5 m/s. Not inferred from distance alone.','possession':'Nearest team within 4.0 m, at least 0.75 m closer than an opponent, sustained 0.2 s. The last observed team may carry for at most 1.2 s across a pass or detector miss; longer gaps remain unknown.','space':'Nearest-player regions on a 24×16 grid, not a calibrated pitch-control probability.','advice':'Coaching hypotheses triggered by measurements, not a learned strategy or guaranteed outcome. Pass lanes omit ball flight, reaction time and offside.','reactions':'Lateral displacement over 1-2 s compares observed endpoints; it does not establish causation or predict a future response.', 'directions':directions or 'Directions must be set per source/half.'}
    return payload
