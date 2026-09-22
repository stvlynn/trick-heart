"""Anchor the replacement to the visible neck tip, independent of hair and hand occlusion."""
import cv2,json,numpy as np
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
cv2.setNumThreads(4)

def landmarks(im):
 b,g,r=cv2.split(im.astype(np.float32))
 skin=((r>225)&(g>100)&(b>110)&(r>g+8)&(r>b+5)&(np.abs(b-g)<80)).astype(np.uint8)
 n,labels,stats,cents=cv2.connectedComponentsWithStats(skin)
 faces=[i for i in range(1,n) if stats[i,4]>3000 and 180<cents[i,1]<750 and stats[i,1]<600]
 face=max(faces,key=lambda i:stats[i,4]);x,y,w,h,_=stats[face]
 ys,xs=np.where(labels==face);bottom=ys.max();chin=np.array([np.median(xs[ys>=bottom-2]),float(bottom)])
 necks=[i for i in range(1,n) if stats[i,4]>100 and abs(cents[i,0]-chin[0])<w*.4 and chin[1]-5<cents[i,1]<chin[1]+w*.4]
 if not necks:raise ValueError('Neck not visible; this frame requires inspection')
 neck=max(necks,key=lambda i:stats[i,4]);ys,xs=np.where(labels==neck);bottom=ys.max();tip=np.array([float(np.median(xs[ys>=bottom-1])),float(bottom)])
 return tip,float(w),dict(face_bounds=stats[face].tolist(),neck_bounds=stats[neck].tolist())

def track(frame,ref):
 tip,w,_=landmarks(frame);rt,rw,_=landmarks(ref);scale=w/rw;delta=tip-rt*scale
 return np.array([[scale,0,delta[0]],[0,scale,delta[1]]],np.float64),1.0

if __name__=='__main__':
 ref=cv2.imread(str(ROOT/'work/frames/s067-103.500.png'));rt,rw,_=landmarks(ref)
 cap=cv2.VideoCapture(str(ROOT/'trick-heart-original.mp4'));cap.set(cv2.CAP_PROP_POS_FRAMES,2472);rows=[]
 for n in range(2472,2702):
  ok,frame=cap.read();assert ok
  tip,w,detail=landmarks(frame);scale=w/rw;delta=tip-rt*scale
  mat=np.array([[scale,0,delta[0]],[0,scale,delta[1]]]);rows.append(dict(frame=n,matrix=mat.tolist(),score=1.0,neck_tip=tip.tolist(),face_width=w,**detail))
 (ROOT/'work/shoulder-fix/neck-tracks.json').write_text(json.dumps(rows,indent=2))
 print('Tracked neck junction in every frame:',len(rows),'reference tip',rt.tolist())
