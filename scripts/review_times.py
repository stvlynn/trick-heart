import json,sys
import cv2
from PIL import Image,ImageDraw
from composite import Plate,finish_visible_cuffs,ROOT,FPS
specs=json.loads((ROOT/'work/timeline.json').read_text()); plates={};cap=cv2.VideoCapture(str(ROOT/'trick-heart-original.mp4'));det=cv2.SIFT_create(nfeatures=2500,contrastThreshold=.025)
ts=list(map(float,sys.argv[1:]))
out=Image.new('RGB',(1440,270*((len(ts)+2)//3)))
for i,t in enumerate(ts):
 cap.set(cv2.CAP_PROP_POS_FRAMES,round(t*FPS));ok,f=cap.read()
 gray=cv2.resize(cv2.cvtColor(f,cv2.COLOR_BGR2GRAY),(960,540));features=det.detectAndCompute(gray,None)
 for k,s in enumerate(specs):
  if round(s['start']*FPS)<=round(t*FPS)<round(s['end']*FPS):
   if k not in plates:plates[k]=Plate(s)
   f,_=plates[k].apply(f,gray,features,round(t*FPS))
 f=finish_visible_cuffs(f,t);cv2.imwrite(str(ROOT/f'work/qc/detail-{t:.3f}.jpg'),f)
 im=Image.fromarray(cv2.cvtColor(cv2.resize(f,(480,270)),cv2.COLOR_BGR2RGB));ImageDraw.Draw(im).text((5,5),str(t),fill='white');out.paste(im,(i%3*480,i//3*270))
out.save(str(ROOT/'work/qc/fixes.jpg'))
