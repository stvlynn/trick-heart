"""Sample the actual compositor at all shot changes and anchors for visual QC."""
import json,sys
from pathlib import Path
import cv2
import numpy as np
from PIL import Image,ImageDraw
from composite import Plate,finish_visible_cuffs,ROOT,FPS
specs=json.loads((ROOT/'work/timeline.json').read_text())
times=set(json.loads((ROOT/'work/anchors.json').read_text()))
for s in specs:times.update([round(s['start']*24)/24,round(s['end']*24-1)/24,(s['start']+s['end'])/2])
cap=cv2.VideoCapture(str(ROOT/'trick-heart-original.mp4'))
detector=cv2.SIFT_create(nfeatures=2500,contrastThreshold=.025)
plates={};report=[];thumbs=[]
folder=ROOT/'work/qc';folder.mkdir(exist_ok=True)
for i,t in enumerate(sorted(times)):
 cap.set(cv2.CAP_PROP_POS_FRAMES,round(t*FPS));ok,frame=cap.read()
 if not ok:continue
 missing=[]
 active=[(k,s) for k,s in enumerate(specs) if round(s['start']*FPS)<=round(t*FPS)<round(s['end']*FPS)]
 if active:
  gray=cv2.resize(cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY),(960,540));features=detector.detectAndCompute(gray,None)
  for k,s in active:
   if not (ROOT/s['art']).exists():missing.append(s['art']);continue
   if k not in plates:plates[k]=Plate(s)
   frame,confidence=plates[k].apply(frame,gray,features)
 frame=finish_visible_cuffs(frame,t)
 b,g,r=cv2.split(frame.astype(np.int16));magenta=(r>150)&(g<110)&(b>65)&(b<r*.8)&(r>g*2)
 residual=int(magenta.sum())
 report.append(dict(time=t,magenta=residual,missing=missing))
 thumb=Image.fromarray(cv2.cvtColor(cv2.resize(frame,(384,216)),cv2.COLOR_BGR2RGB));d=ImageDraw.Draw(thumb);d.rectangle((0,0,200,20),fill='black');d.text((4,3),f'{t:.3f}s red={residual}'+(' MISSING' if missing else ''),fill='white');thumbs.append(thumb)
 if i%40==0:print(i,len(times),flush=True)
for page in range((len(thumbs)+19)//20):
 sheet=Image.new('RGB',(1536,1080))
 for j,im in enumerate(thumbs[page*20:(page+1)*20]):sheet.paste(im,((j%4)*384,(j//4)*216))
 sheet.save(folder/f'page-{page:02d}.jpg',quality=92)
(folder/'report.json').write_text(json.dumps(report,indent=2))
print('QC samples',len(thumbs))
