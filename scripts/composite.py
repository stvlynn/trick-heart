"""Source-tracked video compositing of AI-created Lynn replacement plates.

All character art comes from image_gen. This script only composites that art
into source video frames, tracks source camera transforms and muxes source audio.
"""
import argparse
import json
import subprocess
from pathlib import Path
import cv2
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
W,H,FPS=1920,1080,24
cv2.setNumThreads(4)

def finish_visible_cuffs(frame,t,original=None):
    """Match isolated source hand shots to the locked cyan/white costume cuffs."""
    edits=[]
    if t<10:edits.append(('cyan',550,550,1400,850))
    if 16.375<=t<18.5:
        edits.extend([('cyan',0,0,1000,680),('white',700,500,1920,1080)])
    if 22.75<=t<25.875:edits.extend([('cyan',0,700,1920,1080),('navy',0,0,1920,1080)])
    if 42.4166667<=t<43.1666667:edits.append(('beret',800,695,1920,1080))
    if 92.3333333<=t<93.125:edits.extend([('cyan',0,600,1250,1080),('navy',0,600,1250,1080)])
    if t>=154.0833333:edits.append(('jacket',150,310,950,780))
    if 135.3<=t<136.0416667:edits.append(('beret',570,500,1380,810))
    if 42.4166667<=t<45.4583333:edits.append(('cyan',0,650,1100,1080))
    if 53.375<=t<53.9583333:edits.append(('hat',0,0,1920,1080))
    if 77.625<=t<80.4166667:edits.append(('cyan',0,850,1920,1080))
    if 74<=t<80.4166667:edits.extend([('cyan',0,0,1920,1080),('navy',0,0,1920,1080)])
    if 102.2916667<=t<103:edits.append(('white',0,600,1920,1080))
    if 125.125<=t<126.5833333:edits.append(('white',650,300,1450,800))
    if 144.9166667<=t<146.2083333:edits.append(('white',500,250,1500,850))
    if 103<=t<112.583333333:
        edits.extend([('navy',0,480,1920,1080),('cyan',0,480,1920,1080)])
    for kind,x1,y1,x2,y2 in edits:
        roi=frame[y1:y2,x1:x2]
        b,g,r=cv2.split(roi.astype(np.float32))
        if kind=='cyan':
            mask=(r>180)&(g>70)&(g<205)&(b<90)&(g>r*.40)&(g<r*.78)
            target=np.array([213,174,76],np.float32)
            light=np.clip(r/245,.7,1.1)
        elif kind in ('hat','navy','beret'):
            mask=(r>130)&(r>g*2)&(b>g*1.4)&(b<r*.8)
            target=np.array([213,174,76] if kind=='hat' else ([42,42,42] if kind=='beret' else [73,54,32]),np.float32)
            light=np.clip(r/195,.7,1.1)
        elif kind=='jacket':
            mask=(r>210)&(g>150)&(g<235)&(b>100)&(b<190)&(r>g*1.10)
            target=np.array([73,54,32],np.float32);light=np.clip(r/245,.7,1.1)
        else:
            mx=np.maximum.reduce([b,g,r]);mn=np.minimum.reduce([b,g,r])
            mask=(mx-mn<55)&(mn>85)&(mx<210)&(b>=r)&(b>g+5)
            target=np.array([244,240,235],np.float32)
            light=np.clip((mx-150)*.002+1,.75,1.03)
        colors=np.clip(target[None,None,:]*light[...,None],0,255).astype(np.uint8)
        roi[mask]=colors[mask]
    if original is not None and 103<=t<112.583333333:
        low=original.min(axis=2);high=original.max(axis=2)
        mask=((low>15)&(high<70)&(high-low<15)).astype(np.uint8)
        n,labels,stats,cents=cv2.connectedComponentsWithStats(mask)
        for i in range(1,n):
            x,y,w,h,area=stats[i];cx,cy=cents[i]
            if 1500<area<20000 and 450<cy<800 and 45<w<230 and 35<h<170 and .8<w/h<2.8:
                frame[labels==i]=[213,174,76]
    return frame

def red_background(im):
    b,g,r=cv2.split(im.astype(np.float32))
    return (r>g*1.55)&(r-g>20)&(np.abs(g-b)<25)&(g<120)&(r<205)

def background_color(im):
    sampled=im[::6,::6]
    candidates=red_background(sampled)
    flat=sampled[candidates] if candidates.sum()>100 else sampled.reshape(-1,3)
    q=flat//4
    cols,counts=np.unique(q,axis=0,return_counts=True)
    value=cols[counts.argmax()]*4
    near=np.max(np.abs(flat.astype(np.int16)-value),axis=1)<6
    return np.median(flat[near],axis=0).astype(np.uint8)

class Plate:
    def __init__(self,spec):
        self.spec=spec
        self.ref=cv2.imread(str(ROOT/spec['source']))
        self.art=cv2.resize(cv2.imread(str(ROOT/spec['art'])),(W,H),interpolation=cv2.INTER_LANCZOS4)
        self.bg=background_color(self.ref)
        if spec.get('follow_magenta_head'):
            b,g,r=cv2.split(self.art.astype(np.float32))
            yy,xx=np.ogrid[:H,:W]
            cream=(r>190)&(g>180)&(b>160)&(r>b+4)&((yy<70)|(yy>1010)|(xx<75)|(xx>1840))
            self.art[cream]=self.bg
        if spec.get('art_transform'):
            self.art=cv2.warpAffine(self.art,np.array(spec['art_transform'],np.float64),(W,H),borderValue=tuple(map(int,self.bg)))
        self.roi=np.zeros((H,W),np.uint8)
        for x1,y1,x2,y2 in spec['regions']:
            self.roi[y1:y2,x1:x2]=255
        for x1,y1,x2,y2 in spec.get('exclude',[]):self.roi[y1:y2,x1:x2]=0
        if spec.get('full_patch'):
            self.alpha=self.roi.copy()
        else:
            old=(~red_background(self.ref)).astype(np.uint8)*255
            old=cv2.dilate(old,np.ones((11,11),np.uint8))
            new=(~red_background(self.art)).astype(np.uint8)*255
            self.alpha=cv2.bitwise_and(cv2.bitwise_or(old,new),self.roi)
        self.art[red_background(self.art)]=self.bg
        if spec.get('anatomical_head'):
            from juggling import build_head_masks
            self.head_erase,self.head_alpha=build_head_masks(self.art,red_background)
            self.neck_tracks={r['frame']:r for r in json.loads((ROOT/spec['neck_tracks']).read_text())}
        self.sift=cv2.SIFT_create(nfeatures=1800,contrastThreshold=.025)
        gray=cv2.resize(cv2.cvtColor(self.ref,cv2.COLOR_BGR2GRAY),(960,540))
        roi=cv2.resize(self.roi,(960,540),interpolation=cv2.INTER_NEAREST)
        self.keys,self.desc=self.sift.detectAndCompute(gray,roi)
        self.matcher=cv2.BFMatcher()
        self.last=np.eye(2,3,dtype=np.float64)
        self.failures=0

    def transform(self,gray,features):
        if self.spec.get('static'):return np.eye(2,3,dtype=np.float64),999
        keys,desc=features
        if self.desc is None or desc is None:return self.last,0
        pairs=self.matcher.knnMatch(self.desc,desc,k=2)
        good=[p[0] for p in pairs if len(p)==2 and p[0].distance<.70*p[1].distance]
        if len(good)<5:self.failures+=1;return self.last,0
        a=np.float32([self.keys[m.queryIdx].pt for m in good])*2
        b=np.float32([keys[m.trainIdx].pt for m in good])*2
        mat,inliers=cv2.estimateAffinePartial2D(a,b,method=cv2.RANSAC,ransacReprojThreshold=3,maxIters=1500)
        if mat is None or inliers.sum()<5:self.failures+=1;return self.last,0
        scale=np.hypot(mat[0,0],mat[1,0])
        if scale<.1 or scale>10:self.failures+=1;return self.last,0
        self.last=mat
        return mat,int(inliers.sum())

    def apply(self,frame,gray,features,frame_index=None):
        if self.spec.get('anatomical_head'):
            from juggling import composite_head
            if frame_index in self.neck_tracks:
                row=self.neck_tracks[frame_index];mat=np.array(row['matrix']);score=row['score']
            else:
                from track_neck import track
                mat,score=track(frame,self.ref)
            return composite_head(frame,self.art,self.head_erase,self.head_alpha,mat,background_color(frame)),int(score*100)
        if self.spec.get('only_when_magenta'):
            b,g,r=cv2.split(frame.astype(np.float32))
            if np.count_nonzero((r>155)&(g<95)&(b>g*1.5)&(b<r*.8))<500:return frame,999
        mat,confidence=self.transform(gray,features)
        if self.spec.get('follow_magenta_head'):
            def center(im):
                b,g,r=cv2.split(im[:650].astype(np.float32))
                mask=((r>155)&(g<95)&(b>g*1.5)&(b<r*.8)).astype(np.uint8)
                n,lab,stats,cents=cv2.connectedComponentsWithStats(mask)
                choices=[i for i in range(1,n) if stats[i,4]>500 and cents[i,1]<480]
                if not choices:return None
                i=max(choices,key=lambda i:stats[i,4]);return cents[i],stats[i,4]
            x=center(frame);rx=center(self.ref)
            if x is not None and rx is not None:
                scale=np.sqrt(x[1]/rx[1]);delta=x[0]-rx[0]*scale
                mat=np.array([[scale,0,delta[0]],[0,scale,delta[1]]],np.float64);confidence=999
        art=cv2.warpAffine(self.art,mat,(W,H),flags=cv2.INTER_LINEAR,borderValue=tuple(map(int,self.bg)))
        art[red_background(art)]=self.spec.get('background',background_color(frame))
        alpha=cv2.warpAffine(self.alpha,mat,(W,H),flags=cv2.INTER_LINEAR).astype(np.float32)[...,None]/255
        if self.spec.get('dynamic_erase'):
            old=(~red_background(frame)).astype(np.uint8)*255
            old=cv2.dilate(old,np.ones((7,7),np.uint8)).astype(np.float32)[...,None]/255
            alpha=np.maximum(alpha,old)
        output=(frame.astype(np.float32)*(1-alpha)+art.astype(np.float32)*alpha).astype(np.uint8)
        for x1,y1,x2,y2 in self.spec.get('protect',[]):
            output[y1:y2,x1:x2]=frame[y1:y2,x1:x2]
        for polygon in self.spec.get('preserve_polygons',[]):
            mask=np.zeros((H,W),np.uint8)
            points=cv2.transform(np.array([polygon],np.float32),mat)[0].astype(np.int32)
            cv2.fillPoly(mask,[points],255)
            b,g,r=cv2.split(frame.astype(np.float32))
            animal=((b>220)&(g>220)&(r>220))|((r>155)&(g<100)&(b>g*1.5))
            animal=cv2.dilate(cv2.morphologyEx(animal.astype(np.uint8),cv2.MORPH_CLOSE,np.ones((3,3),np.uint8)),np.ones((3,3),np.uint8))
            mask[animal==0]=0;output[mask>0]=frame[mask>0]
        if self.spec.get('preserve_curtains') and frame[20,W//2,2]>110:
            b,g,r=cv2.split(frame)
            curtain=(r<95)&(r>35)&(g<25)&(b<35)&(r.astype(float)>b.astype(float)*3)&(r.astype(float)>g.astype(float)*3)
            n,labels,stats,_=cv2.connectedComponentsWithStats(curtain.astype(np.uint8))
            keep=[i for i in range(1,n) if stats[i,4]>50000 and stats[i,3]>1000]
            mask=np.isin(labels,keep);output[mask]=frame[mask]
        if self.spec.get('follow_magenta_head'):
            b,g,r=cv2.split(frame.astype(np.float32))
            cream=((r>190)&(g>180)&(b>160)&(r>b+4)).astype(np.uint8)
            n,labels,stats,_=cv2.connectedComponentsWithStats(cream)
            keep=[i for i in range(1,n) if stats[i,4]>1000 and (stats[i,0]==0 or stats[i,1]==0 or stats[i,0]+stats[i,2]>=W or stats[i,1]+stats[i,3]>=H)]
            mask=cv2.dilate(np.isin(labels,keep).astype(np.uint8),np.ones((7,7),np.uint8));output[mask>0]=frame[mask>0]
        if self.spec.get('preserve_balls'):
            b,g,r=cv2.split(frame)
            mask=((r>200)&(g>165)&(b<70)).astype(np.uint8)
            mask[480:]=0
            count,labels,stats,_=cv2.connectedComponentsWithStats(mask)
            for i in range(1,count):
                x,y,w,h,area=stats[i]
                if area>100:
                    cx=x+w//2;cy=y+h//2;rad=max(w,h)//2+5
                    yy,xx=np.ogrid[:H,:W]
                    ball=(xx-cx)**2+(yy-cy)**2<=rad**2
                    output[ball]=frame[ball]
        return output,confidence

def render(manifest,start,end,destination,base=None):
    specs=json.loads(Path(manifest).read_text())
    specs=[s for s in specs if s['end']>start and s['start']<end]
    plates=[Plate(s) for s in specs]
    cap=cv2.VideoCapture(str(ROOT/'trick-heart-original.mp4'))
    basecap=cv2.VideoCapture(str(base)) if base else None
    first=round(start*FPS);last=round(end*FPS)
    cap.set(cv2.CAP_PROP_POS_FRAMES,first)
    if basecap:basecap.set(cv2.CAP_PROP_POS_FRAMES,first)
    writer=subprocess.Popen(['ffmpeg','-v','error','-y','-f','rawvideo','-pix_fmt','bgr24','-s',f'{W}x{H}','-r',str(FPS),'-i','pipe:0','-ss',str(start),'-i',str(ROOT/'trick-heart-original.mp4'),'-map','0:v','-map','1:a','-t',str((last-first)/FPS),'-c:v','libx264','-crf','17','-preset','fast','-pix_fmt','yuv420p','-c:a','aac','-b:a','192k','-movflags','+faststart',str(destination)],stdin=subprocess.PIPE)
    detector=cv2.SIFT_create(nfeatures=2500,contrastThreshold=.025)
    report=[]
    for n in range(first,last):
        ok,frame=cap.read()
        if not ok:break
        original=frame
        if basecap:
            baseok,baseframe=basecap.read();assert baseok
        active=[p for p in plates if round(p.spec['start']*FPS)<=n<round(p.spec['end']*FPS)]
        if active:
            gray=cv2.resize(cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY),(960,540))
            features=detector.detectAndCompute(gray,None)
            for p in active:
                frame,confidence=p.apply(frame,gray,features,n)
                if confidence<8:report.append({'frame':n,'plate':p.spec['art'],'inliers':confidence})
        elif basecap:frame=baseframe
        frame=finish_visible_cuffs(frame,n/FPS,original)
        writer.stdin.write(frame.tobytes())
        if n%120==0:print(f'{n/FPS:.2f}s / {end:.2f}s',flush=True)
    writer.stdin.close()
    code=writer.wait();cap.release()
    Path(str(destination)+'.tracking.json').write_text(json.dumps(report,indent=2))
    assert code==0
    print(f'Wrote {destination}; {len(report)} low-confidence plate frames',flush=True)

if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--manifest',default=str(ROOT/'work/timeline.json'))
    parser.add_argument('--start',type=float,default=0)
    parser.add_argument('--end',type=float,default=157)
    parser.add_argument('--out',default=str(ROOT/'output/trick-heart-lynn.mp4'))
    parser.add_argument('--base')
    args=parser.parse_args()
    render(args.manifest,args.start,args.end,Path(args.out),args.base)
