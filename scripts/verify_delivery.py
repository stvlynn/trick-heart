"""Decode every delivery frame, compare original audio packets, and make a contact sheet."""
import hashlib,json,subprocess,sys
from pathlib import Path
import cv2
from PIL import Image,ImageDraw
from composite import ROOT
video=Path(sys.argv[1]);folder=ROOT/'work/delivery-qc';folder.mkdir(exist_ok=True)
def probe(p):return json.loads(subprocess.check_output(['ffprobe','-v','error','-show_streams','-show_format','-of','json',str(p)]))
def audio_hash(p):
 data=subprocess.check_output(['ffmpeg','-v','error','-i',str(p),'-map','0:a:0','-c','copy','-f','adts','pipe:1'])
 return hashlib.sha256(data).hexdigest()
meta=probe(video);source=probe(ROOT/'trick-heart-original.mp4');cap=cv2.VideoCapture(str(video));n=0;thumbs=[];prev=None;jumps=[]
while True:
 ok,f=cap.read()
 if not ok:break
 if n%24==0:
  im=Image.fromarray(cv2.cvtColor(cv2.resize(f,(384,216)),cv2.COLOR_BGR2RGB));ImageDraw.Draw(im).text((5,5),f'{n/24:.3f}s',fill='white');thumbs.append(im)
 n+=1
cap.release()
for page in range((len(thumbs)+19)//20):
 sheet=Image.new('RGB',(1536,1080))
 for j,im in enumerate(thumbs[page*20:(page+1)*20]):sheet.paste(im,(j%4*384,j//4*216))
 sheet.save(folder/f'page-{page:02d}.jpg',quality=92)
result=dict(decoded_frames=n,expected_frames=3768,metadata=meta,audio_sha256=audio_hash(video),source_audio_sha256=audio_hash(ROOT/'trick-heart-original.mp4'))
result['audio_identical']=result['audio_sha256']==result['source_audio_sha256'];result['passed']=n==3768 and result['audio_identical']
(folder/'verification.json').write_text(json.dumps(result,indent=2));print(json.dumps({k:v for k,v in result.items() if k!='metadata'},indent=2));assert result['passed']
