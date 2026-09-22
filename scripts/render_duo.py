"""Composite generated Lynn sprites into source video, retaining source audio/text."""
from pathlib import Path
import subprocess
import numpy as np
from PIL import Image, ImageFilter

ROOT = Path(__file__).resolve().parents[1]
W, H, FPS = 1920, 1080, 24
START, FRAMES = 10, 85
sprite = np.asarray(Image.open(ROOT / 'assets/lynn/duo-010.png').convert('RGB').resize((W,H), Image.Resampling.LANCZOS)).astype(np.float32)
reference = np.asarray(Image.open(ROOT / 'work/frames/duo-010.png').convert('RGB')).astype(np.float32)
regions = np.zeros((H,W), bool)
regions[145:1055,450:880] = True
regions[500:1055,1030:1565] = True
regions[230:335,440:485] = False  # animated sparkle stays from source
r,g,b = sprite.transpose(2,0,1)
background = (r > g*1.55) & (r < g*3.6) & (np.abs(g-b)<23) & (g<115) & (r<200)
alpha = (~background & regions).astype(np.float32)[...,None]
bg = np.median(reference[:100,:400],axis=(0,1))
old_ink = (np.max(np.abs(reference-bg),axis=2)>24) & regions
old_ink = np.asarray(Image.fromarray(old_ink.astype(np.uint8)*255).filter(ImageFilter.MaxFilter(11)))>0
reader = subprocess.Popen(['ffmpeg','-v','error','-ss',str(START),'-i',str(ROOT/'trick-heart-original.mp4'),'-frames:v',str(FRAMES),'-f','rawvideo','-pix_fmt','rgb24','pipe:1'],stdout=subprocess.PIPE)
writer = subprocess.Popen(['ffmpeg','-v','error','-y','-f','rawvideo','-pix_fmt','rgb24','-s',f'{W}x{H}','-r',str(FPS),'-i','pipe:0','-ss',str(START),'-i',str(ROOT/'trick-heart-original.mp4'),'-map','0:v','-map','1:a','-t',str(FRAMES/FPS),'-c:v','libx264','-crf','17','-preset','fast','-pix_fmt','yuv420p','-c:a','aac','-movflags','+faststart',str(ROOT/'output/lynn-duo-preview.mp4')],stdin=subprocess.PIPE)
for _ in range(FRAMES):
    raw = reader.stdout.read(W*H*3)
    if len(raw) != W*H*3: break
    frame = np.frombuffer(raw,np.uint8).reshape(H,W,3).astype(np.float32)
    frame[old_ink] = np.median(frame[:100,:400],axis=(0,1))
    frame = frame*(1-alpha)+sprite*alpha
    writer.stdin.write(np.clip(frame,0,255).astype(np.uint8).tobytes())
writer.stdin.close()
assert writer.wait()==0
assert reader.wait()==0
print(ROOT/'output/lynn-duo-preview.mp4')
