"""Native-pixel appearance descriptors and temporally aggregated team assignment."""
import cv2,numpy as np
from sklearn.cluster import KMeans

def descriptor(frame,bbox):
    x1,y1,x2,y2=map(int,bbox);h,w=frame.shape[:2]
    crop=frame[max(0,y1):min(h,y2),max(0,x1):min(w,x2)]
    if not crop.size:return None
    hsv=cv2.cvtColor(crop,cv2.COLOR_BGR2HSV);lab=cv2.cvtColor(crop,cv2.COLOR_BGR2LAB)
    hue,sat,val=cv2.split(hsv)
    # Remove turf and almost-black shadows/hair. Do not fall back to grass.
    mask=~((hue>=25)&(hue<=95)&(sat>35)) & (val>45)
    if mask.sum()<6:return None
    pixels=lab[mask].astype(float)/255
    # Luminance quantiles preserve white kits, chroma quantiles preserve blue kits.
    return np.concatenate([np.quantile(pixels,[.25,.5,.75],axis=0).ravel(),
                           [np.mean(sat[mask]<55),np.mean((hue[mask]>95)&(hue[mask]<135)),np.mean(((hue[mask]<15)|(hue[mask]>165))&(sat[mask]>80)),np.mean((hue[mask]>=15)&(hue[mask]<35)&(sat[mask]>80))]])

def assign_teams(samples):
    features={k:np.median(v,axis=0) for k,v in samples.items() if len(v)>=3}
    if len(features)<4:return {},{}
    special={k for k,f in features.items() if max(f[-2:])>.3}
    keys=[k for k in features if k not in special];x=np.array([features[k] for k in keys])
    if len(keys)<4:return {},{}
    # Scale feature dimensions, then trim severe appearance outliers before refitting.
    scale=np.maximum(np.std(x,axis=0),.06);z=x/scale
    km=KMeans(n_clusters=2,n_init=20,random_state=12).fit(z)
    distances=np.min(km.transform(z),axis=1);limit=np.median(distances)+3*max(np.median(abs(distances-np.median(distances))),.2)
    keep=distances<=limit
    if keep.sum()>=4:km.fit(z[keep])
    # Stable semantic labels: high white-pixel fraction is team 1, blue team 2.
    white_label=int(np.argmax(km.cluster_centers_[:,-4]*scale[-4]))
    result={}
    for key,point in zip(keys,z):
        ds=km.transform([point])[0];label=int(ds.argmin());margin=float((max(ds)-min(ds))/max(max(ds),1e-6))
        votes=[]
        for f in samples[key]: votes.append(int(km.predict([f/scale])[0])==label)
        agreement=float(np.mean(votes))
        confidence=margin*agreement
        result[key]={'team':(1 if label==white_label else 2) if confidence>=.35 and min(ds)<=max(limit,2.8) else None,
                     'confidence':round(confidence,3),'samples':len(samples[key]),'agreement':round(agreement,3)}
    for key in special:result[key]={'team':None,'confidence':0,'samples':len(samples[key]),'agreement':0,'reason':'Distinct red/yellow kit, requires role review'}
    return result,{'method':'native non-grass Lab quantiles + track median + KMeans + temporal agreement','minimumConfidence':.35,'tracks':len(keys)}
