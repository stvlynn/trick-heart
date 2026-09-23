"""Independent cels for the five rapid shots at 18.5–19.667 seconds."""
import json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'work/pose19-fix'
old=json.loads((OUT/'timeline-before.json').read_text());patch=[]
for a,b,n in [(449,454,450),(454,459,456),(459,464,460),(464,472,466)]:
 s=dict(start=a/24,end=b/24,source=f'work/pose19-fix/f{n}.png',art=f'assets/lynn/pose19-{n}.png',regions=[[0,0,1920,1080]],full_patch=True,static=True)
 if a==444:s['protect']=[[0,420,230,1080],[1700,0,1920,400]]
 patch.append(s)
new=[s for s in old if not (round(s['start']*24)>=449 and round(s['end']*24)<=472)]+patch
new.sort(key=lambda s:s['start'])
assert sum(round(s['end']*24)-round(s['start']*24) for s in patch)==23
for n in range(449,472):assert sum(round(s['start']*24)<=n<round(s['end']*24) for s in new)==1
(OUT/'patch.json').write_text(json.dumps(patch,indent=2));(ROOT/'work/timeline.json').write_text(json.dumps(new,indent=2)+'\n')
print('Four independent shots, 23 frames, no overlapping layers or gaps.')
