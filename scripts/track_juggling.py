"""Per-drawing head placement for the juggling shot (103.0s to 112.583s).

The source is a boiling limited-animation loop: every few frames the whole
upper body is redrawn, so feature tracking against a single reference frame
jitters and slips whenever the hands cover the hair or the collar. Instead,
frames are grouped into identical drawings, the black bow tie is located on
each drawing, and the Lynn head plate is anchored to the collar just above it.
That keeps the plate's neck sitting on the source shoulders in every frame.
"""
import json
import cv2
import numpy as np
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
cv2.setNumThreads(4)
FIRST,LAST=2472,2702          # frame range of the juggling shot (exclusive end)
CUT=2628                      # the shot cuts to a closer framing here
REF_FRAME=2484                # frame the Lynn plate s067-103.500 was drawn against
WIDE_WINDOW=(1350,430,1750,620)
ZOOM_WINDOW=(40,530,560,800)
OUT=ROOT/'work/shoulder-fix/head-tracks.json'

def read_frames():
    cap=cv2.VideoCapture(str(ROOT/'trick-heart-original.mp4'))
    cap.set(cv2.CAP_PROP_POS_FRAMES,FIRST)
    frames={}
    for n in range(FIRST,LAST):
        ok,f=cap.read();assert ok,n
        frames[n]=f
    return frames

def group_drawings(frames):
    """Map each frame to a drawing id; identical redraws share an id."""
    reps=[];ids={}
    for n in sorted(frames):
        small=cv2.resize(frames[n],(240,135)).astype(np.float32)
        best=None
        for i,rep in enumerate(reps):
            d=float(np.abs(small-rep).mean())
            if d<1.5 and (best is None or d<best[0]):best=(d,i)
        if best is None:
            reps.append(small);ids[n]=len(reps)-1
        else:ids[n]=best[1]
    return ids

def find_bow_tie(frame,window):
    """Return (anchor_x, anchor_y, width, area) of the black bow tie inside window.

    The anchor is the top centre of the tie, i.e. where the collar meets the neck.
    Thin ink outlines are removed with a morphological opening so only the
    solid tie lobes remain.
    """
    x1,y1,x2,y2=window
    roi=frame[y1:y2,x1:x2]
    dark=(roi.max(axis=2)<75).astype(np.uint8)
    dark=cv2.morphologyEx(dark,cv2.MORPH_OPEN,cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(9,9)))
    n,labels,stats,cents=cv2.connectedComponentsWithStats(dark)
    parts=[i for i in range(1,n) if stats[i,4]>500]
    if not parts:return None
    largest=max(parts,key=lambda i:stats[i,4])
    cx,cy=cents[largest]
    # Keep lobes belonging to the same tie as the largest blob.
    lobes=[i for i in parts if abs(cents[i][1]-cy)<45 and abs(cents[i][0]-cx)<150]
    xs=[stats[i,0] for i in lobes];xe=[stats[i,0]+stats[i,2] for i in lobes];ys=[stats[i,1] for i in lobes]
    left,right,top=min(xs),max(xe),min(ys)
    return (float(x1+(left+right)/2),float(y1+top),int(right-left),int(sum(stats[i,4] for i in lobes)))

def estimate_zoom_scale(ref,zoomed):
    """Scale of the close framing relative to the wide framing, from the face."""
    g0=cv2.cvtColor(ref,cv2.COLOR_BGR2GRAY);g1=cv2.cvtColor(zoomed,cv2.COLOR_BGR2GRAY)
    templ=g0[300:445,1400:1700]           # eyes, mouth and chin of the reference drawing
    best=(-1,1.0)
    for s in np.arange(1.35,1.80,.005):
        t=cv2.resize(templ,None,fx=s,fy=s,interpolation=cv2.INTER_LINEAR)
        corr=cv2.matchTemplate(g1,t,cv2.TM_CCOEFF_NORMED)
        _,score,_,_=cv2.minMaxLoc(corr)
        if score>best[0]:best=(score,float(s))
    return best[1],best[0]

def main():
    frames=read_frames()
    ids=group_drawings(frames)
    drawings={}
    for n,i in ids.items():drawings.setdefault(i,[]).append(n)
    ref=frames[REF_FRAME]
    ref_anchor=find_bow_tie(ref,WIDE_WINDOW)
    zoom_scale,zoom_score=estimate_zoom_scale(ref,frames[CUT])
    print(f'{len(drawings)} drawings; reference anchor {ref_anchor}; zoom scale {zoom_scale:.3f} (score {zoom_score:.3f})')
    rows=[];qc=[]
    for i,members in sorted(drawings.items()):
        first=members[0]
        zoom=first>=CUT
        scale=zoom_scale if zoom else 1.0
        tie=find_bow_tie(frames[first],ZOOM_WINDOW if zoom else WIDE_WINDOW)
        assert tie is not None,f'no bow tie found for drawing {i} (frame {first})'
        ax,ay,width,area=tie
        mat=[[scale,0.0,ax-scale*ref_anchor[0]],[0.0,scale,ay-scale*ref_anchor[1]]]
        qc.append(dict(drawing=i,frame=first,anchor=[ax,ay],width=width,area=area,scale=scale,members=len(members)))
        for n in members:
            rows.append(dict(frame=n,drawing=i,matrix=mat,score=1.0,anchor=[ax,ay]))
    rows.sort(key=lambda r:r['frame'])
    OUT.write_text(json.dumps(dict(reference_anchor=ref_anchor,zoom_scale=zoom_scale,drawings=qc,frames=rows),indent=1))
    for q in qc:print(q)
    write_qc_sheet(frames,qc)

def write_qc_sheet(frames,qc,size=(300,300)):
    """One crop per drawing with the detected anchor marked, for eyeballing."""
    cols=8;w,h=size
    sheet=np.zeros((h*((len(qc)+cols-1)//cols),w*cols,3),np.uint8)
    for k,q in enumerate(qc):
        f=frames[q['frame']].copy();ax,ay=map(int,q['anchor']);s=q['scale']
        cv2.drawMarker(f,(ax,ay),(0,255,0),cv2.MARKER_CROSS,40,2)
        half=int(200*s)
        crop=f[max(0,ay-half):ay+half,max(0,ax-half):ax+half]
        crop=cv2.resize(crop,(w,h))
        cv2.putText(crop,f"d{q['drawing']} f{q['frame']}",(4,16),cv2.FONT_HERSHEY_SIMPLEX,.5,(255,255,255),1)
        sheet[k//cols*h:(k//cols+1)*h,k%cols*w:(k%cols+1)*w]=crop
    cv2.imwrite(str(OUT.with_name('anchor-qc.jpg')),sheet,[cv2.IMWRITE_JPEG_QUALITY,90])

if __name__=='__main__':
    main()
