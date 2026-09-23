"""Align whole-body juggling cels and explicit looking directions to source frames."""
import json
from pathlib import Path
import cv2,numpy as np
from track_juggling import find_bow_tie
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'work/juggle-v2'
cv2.setNumThreads(4)
refs=[2472,2478,2484,2487,2490,2496,2502,2505,2508,2514,2517,2520,2523]
cap=cv2.VideoCapture(str(ROOT/'trick-heart-original.mp4'));cap.set(1,2472)
frames={}
for n in range(2472,2785):
 ok,f=cap.read();assert ok;frames[n]=f
anchors={n:find_bow_tie(frames[n],(1350,430,1750,620))[:2] for n in refs}
# Canonical coordinates are relative to the moving collar. Matching includes
# arms and face, excludes border/credits, and normalizes the camera magnification.
def canonical(f,anchor,scale):
 ax,ay=anchor;mat=np.array([[1/scale,0,380-ax/scale],[0,1/scale,500-ay/scale]],np.float32)
 im=cv2.warpAffine(f,mat,(760,1080));valid=cv2.warpAffine(np.ones(f.shape[:2],np.uint8),mat,(760,1080),flags=cv2.INTER_NEAREST)
 # Only visible character area, up to the cropped waist for zoomed shots.
 im=cv2.resize(cv2.GaussianBlur(im,(5,5),0),(152,216)).astype(float)
 valid=cv2.resize(valid,(152,216),interpolation=cv2.INTER_NEAREST)>0
 valid[:18]=False;valid[180:]=False;valid[:,:12]=False;valid[:,140:]=False
 return im,valid
features={n:canonical(frames[n],anchors[n],1) for n in refs}
rows=[];lastanchor=None
for n,f in frames.items():
 scale=1 if n<2628 else 1.59
 window=(1350,430,1750,620) if n<2628 else ((40,530,560,800) if n<2702 else (1400,490,1900,850))
 if n<2769:anchor=find_bow_tie(f,window)[:2];lastanchor=anchor
 else:anchor=lastanchor # already closing; drawing hidden progressively
 cur,valid=canonical(f,anchor,scale)
 scores=[]
 for r in refs:
  im,v=features[r];mask=valid&v
  # Bright cream border and curtains are not pose evidence.
  mask &= (cur[:,:,2]>100)
  scores.append(float(np.abs(cur-im)[mask].mean()) if mask.any() else 999)
 ref=refs[int(np.argmin(scores))] if n<2769 else rows[-1]['reference']
 ax,ay=anchor;rx,ry=anchors[ref]
 rows.append(dict(frame=n,reference=ref,matrix=[[scale,0,ax-scale*rx],[0,scale,ay-scale*ry]],error=min(scores)))
(OUT/'body-tracks.json').write_text(json.dumps(rows,indent=2))
old=json.loads((OUT/'timeline-before.json').read_text())
patch=[dict(start=2472/24,end=2785/24,source='work/juggle-v2/f2496.png',art='assets/lynn/juggle-v2-2496.png',regions=[[0,0,1920,1080]],full_patch=True,static=True,whole_body_tracks='work/juggle-v2/body-tracks.json')]
# Match only the head and duck, never sofa/background particles: three directions.
cap=cv2.VideoCapture(str(ROOT/'trick-heart-original.mp4'));lookrefs=[3360,3364,3372];lf={}
for n in lookrefs:
 im=cv2.imread(str(OUT/f'f{n}.png'));lf[n]=cv2.resize(im[340:565,920:1130],(84,90)).astype(float)
labels=[]
for n in range(3356,3397):
 cap.set(1,n);ok,f=cap.read();assert ok
 im=cv2.resize(f[340:565,920:1130],(84,90)).astype(float)
 labels.append(lookrefs[int(np.argmin([np.abs(im-lf[r]).mean() for r in lookrefs]))])
a=3356;last=labels[0]
def addlook(a,b,r):
 patch.append(dict(start=a/24,end=b/24,source=f'work/juggle-v2/f{r}.png',art=f'assets/lynn/look-v2-{r}.png',regions=[[420,475,1640,1080],[590,245,1345,475]],full_patch=True,static=True,source_gold_particles=True))
for n,r in zip(range(3357,3397),labels[1:]):
 if r!=last:addlook(a,n,last);a=n;last=r
addlook(a,3397,last)
new=[s for s in old if not (s.get('anatomical_head') and s['start']>=103 and s['start']<116) and not (s['start']>=3356/24-.00001 and s['end']<=3397/24+.00001)]+patch
new.sort(key=lambda s:s['start'])
(OUT/'patch.json').write_text(json.dumps(patch,indent=2));(ROOT/'work/timeline.json').write_text(json.dumps(new,indent=2)+'\n')
print('Body references:',{r:sum(x['reference']==r for x in rows) for r in refs})
print('Look holds:',[(round(s['start']*24),round(s['end']*24),s['art']) for s in patch[1:]])
