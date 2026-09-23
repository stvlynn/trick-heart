"""Build frame-accurate repair holds from source drawing references."""
import json,copy
from pathlib import Path
import cv2,numpy as np
ROOT=Path(__file__).resolve().parents[1]
cv2.setNumThreads(4)
OUT=ROOT/'work/motion-audit'
old=json.loads((OUT/'timeline-before.json').read_text())
cap=cv2.VideoCapture(str(ROOT/'trick-heart-original.mp4'));frames=[]
while True:
    ok,f=cap.read()
    if not ok:break
    frames.append(cv2.resize(f,(320,180)))
patch=[];removed=set();mapping=[]

def base(i,ref=None,art=None,**flags):
    s=copy.deepcopy(old[i]);s.update(flags)
    for k in ['art_transform','background','only_when_magenta']:s.pop(k,None)
    s['static']=True
    if ref is not None:s['source']=f'work/motion-audit/details/f{ref}.png'
    if art is not None:s['art']=art
    return s

def repair(i,n,**flags):return base(i,n,f'assets/lynn/repair-{n}.png',**flags)

def append(a,b,s):
    s=copy.deepcopy(s);s.update(start=a/24,end=b/24);patch.append(s)

def passthrough(a,b):
    append(a,b,dict(source='work/frames/s008-016.500.png',art='assets/lynn/s008-016.500.png',regions=[[0,0,1920,1080]],source_passthrough=True,static=True))

def split(a,b,refs,roi):
    """Nearest source drawing in a fixed character ROI; no delivery pixels used."""
    x1,y1,x2,y2=[round(v/6) for v in roi]
    def feature(f):return cv2.GaussianBlur(f[y1:y2,x1:x2],(3,3),0).astype(float)
    features=[feature(frames[n]) for n,s in refs]
    labels=[]
    for n in range(a,b):
        q=feature(frames[n]);ds=[float(np.abs(q-r).mean()) for r in features];k=int(np.argmin(ds));labels.append(k);mapping.append(dict(frame=n,reference=refs[k][0],error=ds[k]))
    first=a;last=labels[0]
    for n,k in zip(range(a+1,b),labels[1:]):
        if k!=last:append(first,n,refs[last][1]);first=n;last=k
    append(first,b,refs[last][1])

# Rabbit emerging from hat was overwritten by an early face plate.
removed.add(24);passthrough(849,862)
for a,b,ref in [(862,868,864),(868,874,870),(874,880,876),(880,886,882),(886,892,864),(892,898,870),(898,901,876)]:
    s=base(24,ref,f'assets/lynn/frame-{ref}.png',source_particles=True,local_motion=True)
    append(a,b,s)

# Rabbit interaction: smile, wink, blink, then the actual rabbit closeup cut.
removed.update([26,27,28])
refs=[(936,base(26,local_motion=True)),(939,repair(26,939,local_motion=True)),(948,repair(26,948,local_motion=True)),(960,base(27,local_motion=True)),(964,repair(27,964,local_motion=True)),(989,base(28,local_motion=True))]
for _,s in refs:s['regions']=[[700,0,1920,1080]]
split(930,1018,refs,[850,0,1920,1080])

# Rapid montage. Large pose/camera changes have dedicated drawings. Local
# redraw differences are transferred by dense flow only within the same pose.
removed.update(range(29,37))
refs=[(1092,base(29,local_motion=True,source_border=True,source_white_foreground=True)),(1097,repair(29,1097,local_motion=True,source_border=True)),(1100,repair(30,1100,local_motion=True,source_border=True)),(1105,base(30,local_motion=True,source_border=True)),(1114,repair(30,1114,local_motion=True,source_border=True))]
split(1091,1118,refs,[400,60,1880,1030])
for i in [31,32,33,36]:append(round(old[i]['start']*24),round(old[i]['end']*24),base(i,local_motion=True,source_border=True))
passthrough(1208,1210)
append(1210,1224,base(34,local_motion=True,source_border=True,max_flow=45))
split(1224,1244,[(1224,base(35,1224,local_motion=True,source_border=True)),(1233,repair(35,1233,local_motion=True,source_border=True))],[400,50,1600,1030])

# Four apparently still cup shots contain distinct in-place drawings.
for i,n in [(39,1425),(40,1479),(41,1567),(42,1609)]:
    removed.add(i);a=round(old[i]['start']*24);b=round(old[i]['end']*24)
    ref=round(float(Path(old[i]['source']).stem.rsplit('-',1)[1])*24)
    s0=base(i,local_motion=True);s1=repair(i,n,local_motion=True)
    roi=[min(r[0] for r in s0['regions']),0,max(r[2] for r in s0['regions']),1080]
    split(a,b,[(ref,s0),(n,s1)],roi)

# Profile entrance, open eyes, gust reaction, closed eyes, and exit.
removed.add(49);passthrough(2127,2138)
append(2138,2144,repair(49,2139,local_motion=True))
append(2144,2156,repair(49,2148,local_motion=True))
append(2156,2169,base(49,local_motion=True))
passthrough(2169,2197)

# Flower growth, spiral reaction, burst and white flash; then card expressions.
removed.update([54,55])
append(2401,2410,base(54,local_motion=True))
append(2410,2419,repair(54,2410,local_motion=True))
append(2419,2427,repair(54,2419,local_motion=True))
append(2427,2432,repair(54,2428,local_motion=True,source_white_foreground=True))
passthrough(2432,2437)
append(2437,2440,repair(55,2437,local_motion=True,source_white_foreground=True))
append(2440,2448,base(55,local_motion=True))
append(2448,2455,repair(55,2449,local_motion=True))

# Framed reprise of juggling: retain actual hands, balls and body animation.
removed.add(57)
append(2702,2769,dict(source='work/frames/s067-103.500.png',art='assets/lynn/s067-103.500.png',regions=[[1170,0,1810,485]],full_patch=True,anatomical_head=True,head_tracks='work/motion-audit/credits-head-tracks.json'))

# Wind must composite above the characters, including inside the replaced ROI.
removed.add(58);append(round(old[58]['start']*24),round(old[58]['end']*24),base(58,local_motion=True,restore_wind=True))

# Small inset hand shot: both outfits need actual cuffs, not a tint of old sleeves.
for a,b,n in [(3003,3011,3005),(3011,3038,3012)]:
    append(a,b,dict(source=f'work/motion-audit/details/f{n}.png',art=f'assets/lynn/repair-{n}.png',regions=[[500,200,1550,860]],full_patch=True,static=True,local_motion=True,max_flow=55))

# Duck -> rabbit-on-sleeve -> closed-eye inset -> open-eye closeup. The old
# timeline pasted a large closeup over the rabbit panel and exposed its bottom.
removed.update([62,63,64,65])
passthrough(3265,3281)
append(3281,3305,dict(source='work/motion-audit/details/f3284.png',art='assets/lynn/repair-3284.png',regions=[[500,260,1450,860]],full_patch=True,static=True,local_motion=True))
append(3305,3321,repair(63,3305,regions=[[500,260,1450,860]],local_motion=True))
append(3321,3328,repair(64,3321,local_motion=True))
append(3328,3356,repair(64,3335,local_motion=True))
split(3356,3397,[(3360,base(65,local_motion=True)),(3372,repair(65,3372,local_motion=True)),(3378,repair(65,3378,local_motion=True))],[850,230,1300,1030])

# Ensure every newly referenced source exists, extracted exactly once.
refs={s['source'] for s in patch if s['source'].startswith('work/motion-audit/details/')}
cap=cv2.VideoCapture(str(ROOT/'trick-heart-original.mp4'))
for path in sorted(refs):
    p=ROOT/path
    if not p.exists():
        n=int(p.stem[1:]);cap.set(1,n);ok,f=cap.read();assert ok;cv2.imwrite(str(p),f)

timeline=[s for i,s in enumerate(old) if i not in removed]+patch
timeline.sort(key=lambda s:s['start'])
(OUT/'patch.json').write_text(json.dumps(patch,indent=2))
(OUT/'mapping.json').write_text(json.dumps(mapping,indent=2))
(ROOT/'work/timeline.json').write_text(json.dumps(timeline,ensure_ascii=False,indent=2)+'\n')
print(f'{len(patch)} repair holds; removed {len(removed)} old holds')
