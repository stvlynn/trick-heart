"""Rank possible lost drawings/cuts by comparing source and delivery animation.

This is a candidate detector, not a semantic proof of missing frames. It removes
red backdrop, small lyric/particle components and normalizes the foreground box
to reduce camera translation/scale. Per-shot drawing clusters and consecutive
frame differences then reveal source changes suppressed in the delivery.
"""
import argparse, json
from pathlib import Path
import cv2
import numpy as np
from PIL import Image, ImageDraw

ROOT=Path(__file__).resolve().parents[1]
cv2.setNumThreads(4)

def descriptor(frame, spec):
    im=cv2.resize(frame,(320,180),interpolation=cv2.INTER_AREA)
    roi=np.zeros((180,320),np.uint8)
    for x1,y1,x2,y2 in spec['regions']:
        roi[round(y1/6):round(y2/6),round(x1/6):round(x2/6)]=1
    b,g,r=cv2.split(im.astype(float))
    red=(r>g*1.5)&(r-g>20)&(np.abs(g-b)<30)&(g<130)
    # Opening removes narrow text strokes; retaining only substantial islands
    # suppresses floating particles. The same method is used for both videos.
    mask=cv2.morphologyEx(((~red)&(roi>0)).astype(np.uint8),cv2.MORPH_OPEN,np.ones((3,3),np.uint8))
    n,labels,stats,_=cv2.connectedComponentsWithStats(mask)
    keep=[i for i in range(1,n) if stats[i,4]>=100]
    mask=np.isin(labels,keep).astype(np.uint8)
    if not mask.any():return np.zeros((64,64,3),np.uint8)
    mask=cv2.dilate(mask,np.ones((5,5),np.uint8))
    ys,xs=np.where(mask);x1,x2=xs.min(),xs.max()+1;y1,y2=ys.min(),ys.max()+1
    im[mask==0]=0
    return cv2.resize(cv2.GaussianBlur(im[y1:y2,x1:x2],(3,3),0),(64,64))

def cluster(arr, threshold):
    reps=[];ids=[];repindices=[]
    for i,a in enumerate(arr):
        ds=[float(np.abs(a.astype(float)-b).mean()) for b in reps]
        if ds and min(ds)<threshold:ids.append(int(np.argmin(ds)))
        else:ids.append(len(reps));reps.append(a.astype(float));repindices.append(i)
    return ids,repindices

def run(video,out,threshold=5):
    specs=json.loads((ROOT/'work/timeline.json').read_text())
    out.mkdir(parents=True,exist_ok=True)
    source=cv2.VideoCapture(str(ROOT/'trick-heart-original.mp4'))
    delivery=cv2.VideoCapture(str(video));data={};events=[];previous=None;n=0
    while True:
        ok,a=source.read();ok2,b=delivery.read()
        if not ok or not ok2:break
        smalla=cv2.resize(a,(160,90));smallb=cv2.resize(b,(160,90))
        if previous is not None:
            da=float(np.abs(smalla.astype(float)-previous[0]).mean());db=float(np.abs(smallb.astype(float)-previous[1]).mean())
            if da>12 and db<da*.3:events.append(dict(frame=n,time=n/24,source_change=da,delivery_change=db))
        previous=(smalla.astype(float),smallb.astype(float))
        for i,s in enumerate(specs):
            if round(s['start']*24)<=n<round(s['end']*24):
                row=data.setdefault(i,dict(frames=[],source=[],delivery=[]))
                row['frames'].append(n);row['source'].append(descriptor(a,s));row['delivery'].append(descriptor(b,s))
        n+=1
    results=[]
    for i,d in data.items():
        if len(d['frames'])<9:continue
        si,sr=cluster(d['source'],threshold);di,dr=cluster(d['delivery'],threshold)
        sm=np.array([np.abs(a.astype(float)-b).mean() for a,b in zip(d['source'][1:],d['source'])]);dm=np.array([np.abs(a.astype(float)-b).mean() for a,b in zip(d['delivery'][1:],d['delivery'])])
        lost=np.flatnonzero((sm>3)&(dm<sm*.3))+1
        score=len(lost)*max(1,len(sr)/max(1,len(dr)))
        results.append(dict(plate=i,start=specs[i]['start'],end=specs[i]['end'],art=specs[i]['art'],source_clusters=len(sr),delivery_clusters=len(dr),suppressed_events=[d['frames'][j] for j in lost],score=score,representative_frames=[d['frames'][j] for j in sr],candidate=bool(len(lost)>1 or len(sr)>=len(dr)*2 and len(sr)>=3)))
    results.sort(key=lambda r:r['score'],reverse=True)
    report=dict(video=str(video),decoded_frames=n,cluster_threshold=threshold,method='foreground component filtering, bounding-box normalization, drawing clustering, temporal change ratio',limitations='Heuristic candidates: hair redraws, camera changes, occlusion, lyrics and particles can cause false positives. Manual review required.',suppressed_large_changes=events,shots=results)
    (out/'report.json').write_text(json.dumps(report,ensure_ascii=False,indent=2))
    # Compare up to four source representatives against their delivered frames.
    candidates=[r for r in results if r['candidate']]
    for page in range((len(candidates)+5)//6):
        sheet=Image.new('RGB',(1280,6*390));draw=ImageDraw.Draw(sheet)
        for j,r in enumerate(candidates[page*6:(page+1)*6]):
            frames=r['representative_frames'];chosen=[frames[k] for k in np.linspace(0,len(frames)-1,min(4,len(frames))).astype(int)]
            y=j*390;draw.text((5,y),f"{r['start']:.3f}-{r['end']:.3f}s S:{r['source_clusters']} D:{r['delivery_clusters']} lost:{len(r['suppressed_events'])}",fill='white')
            for col,f in enumerate(chosen):
                for row,cap in enumerate([source,delivery]):
                    cap.set(1,f);ok,im=cap.read();assert ok
                    thumb=Image.fromarray(cv2.cvtColor(cv2.resize(im,(320,180)),cv2.COLOR_BGR2RGB));sheet.paste(thumb,(col*320,y+25+row*180))
                draw.text((col*320+5,y+25),str(f),fill='white')
        sheet.save(out/f'candidates-{page:02d}.jpg',quality=90)
    print(json.dumps(dict(frames=n,candidates=len(candidates),top=candidates[:15],large_changes=events),indent=2))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--video',default=str(ROOT/'output/Trick-Heart-Lynn-PV.mp4'));p.add_argument('--out',default=str(ROOT/'work/motion-audit/before'));p.add_argument('--threshold',type=float,default=5);a=p.parse_args();run(Path(a.video),Path(a.out),a.threshold)
