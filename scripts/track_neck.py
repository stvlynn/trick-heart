"""Track the collar/chin junction, which stays visible when hands obscure the hair."""
import cv2,json,numpy as np
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
cv2.setNumThreads(4)
REF=(1488,435,1600,503)

def track(frame,ref):
 x1,y1,x2,y2=REF
 patch=cv2.cvtColor(ref[y1:y2,x1:x2],cv2.COLOR_BGR2GRAY)
 target=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
 small=cv2.resize(target,(960,540))
 # The collar is always below the chin, in this part of the frame.
 small[:180]=0;small[440:]=0
 best=(-1,None)
 for scale in np.arange(.85,1.751,.025):
  templ=cv2.resize(patch,None,fx=scale*.5,fy=scale*.5,interpolation=cv2.INTER_AREA)
  corr=cv2.matchTemplate(small,templ,cv2.TM_CCOEFF_NORMED)
  _,score,_,xy=cv2.minMaxLoc(corr)
  if score>best[0]:best=(score,(scale,xy[0]*2,xy[1]*2))
 scale,x,y=best[1];best=(-1,None)
 left=max(0,x-9);top=max(0,y-9);right=min(1920,x+int(patch.shape[1]*(scale+.03))+12);bottom=min(1080,y+int(patch.shape[0]*(scale+.03))+12)
 search=target[top:bottom,left:right]
 for s in np.arange(scale-.024,scale+.0241,.002):
  templ=cv2.resize(patch,None,fx=s,fy=s,interpolation=cv2.INTER_LINEAR)
  corr=cv2.matchTemplate(search,templ,cv2.TM_CCOEFF_NORMED)
  _,score,_,xy=cv2.minMaxLoc(corr)
  if score>best[0]:best=(score,(s,xy[0]+left,xy[1]+top))
 s,x,y=best[1]
 return np.array([[s,0,x-x1*s],[0,s,y-y1*s]],np.float64),best[0]

if __name__=='__main__':
 ref=cv2.imread(str(ROOT/'work/frames/s067-103.500.png'))
 cap=cv2.VideoCapture(str(ROOT/'trick-heart-original.mp4'));cap.set(cv2.CAP_PROP_POS_FRAMES,2472)
 rows=[]
 for n in range(2472,2702):
  ok,frame=cap.read();assert ok
  mat,score=track(frame,ref);rows.append(dict(frame=n,matrix=mat.tolist(),score=score))
  if n%24==0:print(n/24,score,flush=True)
 (ROOT/'work/shoulder-fix/neck-tracks.json').write_text(json.dumps(rows,indent=2))
 print('min',min(r['score'] for r in rows))
