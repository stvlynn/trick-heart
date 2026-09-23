"""Whole-body cel replacement; keep source credits, border and closing curtains."""
import json
import cv2,numpy as np
W,H=1920,1080
class BodyAnimation:
 def __init__(self,root,path):
  self.rows={r['frame']:r for r in json.loads((root/path).read_text())};self.arts={}
  for n in sorted({r['reference'] for r in self.rows.values()}):
   im=cv2.resize(cv2.imread(str(root/f'assets/lynn/juggle-v2-{n}.png')),(W,H))
   b,g,r=cv2.split(im.astype(float));cream=((r>195)&(g>185)&(b>160)&(r-b<45)).astype(np.uint8)
   count,lab,stats,_=cv2.connectedComponentsWithStats(cream)
   keep=[i for i in range(1,count) if stats[i,4]>1000 and (stats[i,0]==0 or stats[i,1]==0 or stats[i,0]+stats[i,2]>=W or stats[i,1]+stats[i,3]>=H)]
   edge=cv2.dilate(np.isin(lab,keep).astype(np.uint8),np.ones((13,13),np.uint8))>0
   im[edge]=(56,48,132)
   self.arts[n]=im
 def apply(self,frame,n,red_background,background_color):
  row=self.rows[n];ref=row['reference'];mat=np.array(row['matrix'],np.float32);bgmask=red_background(frame)&(frame[:,:,2]>110)
  bg=np.median(frame[bgmask],axis=0).astype(np.uint8) if bgmask.any() else np.array([56,47,134],np.uint8)
  art=self.arts[ref].copy();art[red_background(art)]=bg
  warped=cv2.warpAffine(art,mat,(W,H),borderValue=tuple(map(int,bg)))
  out=frame.copy();x1,x2=(1120,1920) if n<2628 else ((0,700) if n<2702 else (1190,1920))
  out[:,x1:x2]=warped[:,x1:x2]
  b,g,r=cv2.split(frame.astype(float))
  cream=((r>195)&(g>185)&(b>160)&(r-b<45)).astype(np.uint8)
  count,lab,stats,_=cv2.connectedComponentsWithStats(cream)
  keep=[i for i in range(1,count) if stats[i,4]>1000 and (stats[i,0]==0 or stats[i,1]==0 or stats[i,0]+stats[i,2]>=W or stats[i,1]+stats[i,3]>=H)]
  border=cv2.dilate(np.isin(lab,keep).astype(np.uint8),np.ones((7,7),np.uint8))>0
  out[border]=frame[border]
  # This cel omitted the airborne hat ball; restore that isolated source prop.
  if ref==2490:
   ball=((r>200)&(g>165)&(b<90)).astype(np.uint8);ball[300:]=0
   mask=cv2.dilate(ball,np.ones((3,3),np.uint8))>0;out[mask]=frame[mask]
  if n>=2769:
   # Curtains span the full frame vertically. Include their ink edges and every
   # pixel outside their opening, rather than a colour-only mask with pinholes.
   curtain=(r<105)&(g<40)&(b<55)&(r>25)
   columns=np.mean(curtain,axis=0)>.80
   center=W//2
   left=np.flatnonzero(columns[:center]);right=np.flatnonzero(columns[center:])+center
   if left.size:out[:,:min(W,int(left.max())+4)]=frame[:,:min(W,int(left.max())+4)]
   if right.size:out[:,max(0,int(right.min())-3):]=frame[:,max(0,int(right.min())-3):]
  return out
