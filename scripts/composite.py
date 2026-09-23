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

def finish_visible_cuffs(frame,t):
    """Match isolated source hand shots to the locked cyan/white costume cuffs."""
    edits=[]
    if t<10:edits.append(('cyan',550,550,1400,850))
    if 16.375<=t<18.5:
        edits.extend([('cyan',0,0,1000,680),('white',700,500,1920,1080)])
    if 22.75<=t<25.875:edits.extend([('cyan',0,700,1920,1080),('navy',0,0,1920,1080)])
    if 42.4166667<=t<43.1666667:edits.append(('beret',800,695,1920,1080))
    if 92.3333333<=t<93.125:edits.extend([('cyan',0,600,1250,1080),('navy',0,600,1250,1080)])
    if t>=154.0833333:edits.append(('jacket',150,310,950,780))
    if 135.3<=t<3281/24:edits.append(('beret',570,300,1380,810))
    if 42.4166667<=t<45.4583333:edits.append(('cyan',0,650,1100,1080))
    if 53.375<=t<53.9583333:edits.append(('hat',0,0,1920,1080))
    if 849/24<=t<862/24:edits.append(('hat',0,650,1920,1080))
    if 77.625<=t<80.4166667:edits.append(('cyan',0,850,1920,1080))
    if 74<=t<80.4166667:edits.extend([('cyan',0,0,1920,1080),('navy',0,0,1920,1080)])
    if 102.2916667<=t<103:edits.append(('white',0,600,1920,1080))
    if 125.125<=t<126.5833333:edits.append(('white',650,300,1450,800))
    if 144.9166667<=t<146.2083333:edits.append(('white',500,250,1500,850))
    if 103<=t<115.375:
        # Whole frame: the source jacket shoulders reach up to the collar line.
        edits.extend([('navy',0,0,1920,1080),('cyan',0,0,1920,1080)])
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
    return frame

def red_background(im):
    b,g,r=cv2.split(im.astype(np.float32))
    return (r>g*1.55)&(r-g>20)&(np.abs(g-b)<25)&(g<120)&(r<205)

def band_edges(im,x1,x2,y_lo=200,y_hi=900):
    """Per-column row of the letterbox band's top and bottom ink outline.

    The band's outline boils from frame to frame, so a plate cannot carry its
    own copy of the edge. The outline is the first dark run below the red
    backdrop; where the source character's hair spills over the outline the
    edge is interpolated from neighbouring columns. Loose peach petals drifting
    over the backdrop are ignored so they do not pull the edge outwards.
    Returns (top, bottom, solid) with top/bottom as float arrays over x.
    """
    b,g,r=cv2.split(im.astype(np.float32))
    petal=cv2.dilate(((r>225)&(g>175)&(g<225)&(b>120)&(b<200)).astype(np.uint8),np.ones((7,7),np.uint8))>0
    solid=(~red_background(im))&(~petal)
    solid[:y_lo]=False;solid[y_hi:]=False;solid[:,:x1]=False;solid[:,x2:]=False
    dark=(im.max(axis=2)<70)&solid
    ys=np.arange(H)[:,None]
    first=np.where(solid,ys,H).min(axis=0);last=np.where(solid,ys,-1).max(axis=0)
    first_dark=np.where(dark,ys,H).min(axis=0);last_dark=np.where(dark,ys,-1).max(axis=0)
    # Outline is trusted only where it sits right at the band boundary.
    top_ok=(first<H)&(first_dark-first<=3);bottom_ok=(last>=0)&(last-last_dark<=3)
    cols=np.arange(W)
    def fill(values,ok):
        ok=ok.copy();ok[:x1]=False;ok[x2:]=False
        if ok.sum()<2:return values.astype(np.float32)
        out=np.interp(cols,cols[ok],values[ok]).astype(np.float32)
        return cv2.medianBlur(out.reshape(1,-1),5).ravel()
    return fill(first_dark,top_ok),fill(last_dark,bottom_ok),solid

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
        if spec.get('whole_body_tracks'):
            from juggle_body import BodyAnimation
            self.body_animation=BodyAnimation(ROOT,spec['whole_body_tracks'])
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
            tracks=json.loads((ROOT/spec['head_tracks']).read_text())
            self.head_tracks={r['frame']:r for r in tracks['frames']}
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
        if self.spec.get('source_passthrough'):
            return frame,999
        if self.spec.get('whole_body_tracks'):
            return self.body_animation.apply(frame,frame_index,red_background,background_color),999
        if self.spec.get('anatomical_head'):
            from juggling import composite_head
            # Placement is precomputed per source drawing by scripts/track_juggling.py.
            row=self.head_tracks[frame_index];mat=np.array(row['matrix'],np.float64)
            return composite_head(frame,self.art,self.head_erase,self.head_alpha,mat,background_color(frame),row['anchor']),999
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
        if self.spec.get('local_motion'):
            # Transfer local source redraw motion, not a whole-image oscillation.
            # Large semantic changes use separately drawn plates; this is only
            # for small within-pose hair/eyes/hands/clothing deformations.
            current=cv2.resize(cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY),(480,270))
            reference=cv2.resize(cv2.cvtColor(self.ref,cv2.COLOR_BGR2GRAY),(480,270))
            flow=cv2.calcOpticalFlowFarneback(current,reference,None,.5,3,25,5,7,1.5,0)
            flow=cv2.resize(cv2.GaussianBlur(flow,(9,9),0),(W,H))*4
            flow=np.clip(flow,-self.spec.get('max_flow',28),self.spec.get('max_flow',28))
            yy,xx=np.mgrid[:H,:W].astype(np.float32)
            art=cv2.remap(art,xx+flow[:,:,0],yy+flow[:,:,1],cv2.INTER_LINEAR,borderMode=cv2.BORDER_REPLICATE)
        art[red_background(art)]=self.spec.get('background',background_color(frame))
        if self.spec.get('source_gold_particles'):
            b,g,r=cv2.split(art.astype(float))
            gold=((r>215)&(g>150)&(g<235)&(b>100)&(b<205)&(r-b>30)).astype(np.uint8);gold[475:]=0
            gold=cv2.dilate(gold,np.ones((3,3),np.uint8))>0
            art[gold]=background_color(frame)
        if self.spec.get('restore_overlays'):
            art=self.strip_plate_overlays(art,frame)
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
        if self.spec.get('follow_band'):
            output=self.follow_band(frame,art,output)
        if self.spec.get('restore_overlays'):
            output=self.restore_overlays(frame,art,output)
        if self.spec.get('source_border'):
            b,g,r=cv2.split(frame.astype(float))
            cream=((r>195)&(g>195)&(b>175)&(r-b>3)&(r-b<45)).astype(np.uint8)
            n,lab,stats,_=cv2.connectedComponentsWithStats(cream)
            keep=[i for i in range(1,n) if stats[i,4]>1000 and (stats[i,0]==0 or stats[i,1]==0 or stats[i,0]+stats[i,2]>=W or stats[i,1]+stats[i,3]>=H)]
            edge=cv2.dilate(np.isin(lab,keep).astype(np.uint8),np.ones((9,9),np.uint8))>0
            output[edge]=frame[edge]
        if self.spec.get('source_gold_particles'):
            b,g,r=cv2.split(frame.astype(float))
            gold=((r>215)&(g>150)&(g<235)&(b>100)&(b<205)&(r-b>30)).astype(np.uint8);gold[475:]=0
            mask=(cv2.dilate(gold,np.ones((3,3),np.uint8))>0)&red_background(art)
            output[mask]=frame[mask]
        if self.spec.get('source_particles'):
            output=self.restore_particles(frame,output)
        if self.spec.get('restore_wind'):
            output=self.restore_wind(frame,output)
        if self.spec.get('source_white_foreground'):
            b,g,r=cv2.split(frame.astype(float))
            white=((np.minimum.reduce([b,g,r])>225)&(np.maximum.reduce([b,g,r])-np.minimum.reduce([b,g,r])<18)).astype(np.uint8)
            n,lab,stats,_=cv2.connectedComponentsWithStats(white)
            mask=cv2.dilate(np.isin(lab,[i for i in range(1,n) if stats[i,4]>18000]).astype(np.uint8),np.ones((3,3),np.uint8))>0
            output[mask]=frame[mask]
        return output,confidence

    def restore_particles(self,frame,output):
        b,g,r=cv2.split(frame.astype(float))
        white=((np.minimum.reduce([b,g,r])>225)&(np.maximum.reduce([b,g,r])-np.minimum.reduce([b,g,r])<22)).astype(np.uint8)
        skin=((r>225)&(b>180)&(g>180)&(r-b>15)&(r-b<70)&(np.abs(g-b)<20)).astype(np.uint8)
        face=np.zeros((H,W),np.uint8)
        count,labels,stats,_=cv2.connectedComponentsWithStats(skin)
        if count>1:
            k=1+np.argmax(stats[1:,4]);points=cv2.findNonZero((labels==k).astype(np.uint8))
            if points is not None:cv2.fillConvexPoly(face,cv2.convexHull(points),1)
        n,lab,stats,_=cv2.connectedComponentsWithStats(white);mask=np.zeros((H,W),np.uint8)
        for i in range(1,n):
            if 100<stats[i,4]<12000 and max(stats[i,2:4])<160 and np.mean(face[lab==i])<.2:mask[lab==i]=1
        mask=cv2.dilate(mask,np.ones((5,5),np.uint8))
        text=((r>215)&(g>150)&(g<235)&(b>100)&(b<205)&(r-b>30)).astype(np.uint8);text[190:]=0
        output[mask>0]=frame[mask>0]
        # Keep glyph cores without copying the old red hair around their edges.
        output[text>0]=frame[text>0]
        return output

    def restore_wind(self,frame,output):
        b,g,r=cv2.split(frame.astype(float))
        white=((np.minimum.reduce([b,g,r])>200)&(np.maximum.reduce([b,g,r])-np.minimum.reduce([b,g,r])<28)).astype(np.uint8)
        thick=cv2.dilate((cv2.distanceTransform(white,cv2.DIST_L2,5)>9).astype(np.uint8),np.ones((19,19),np.uint8))
        thin=white.copy();thin[thick>0]=0
        n,lab,stats,_=cv2.connectedComponentsWithStats(thin)
        keep=[i for i in range(1,n) if max(stats[i,2:4])>85 and stats[i,4]>100]
        mask=cv2.dilate(np.isin(lab,keep).astype(np.uint8),np.ones((3,3),np.uint8))>0
        output[mask]=frame[mask]
        return output

    def strip_plate_overlays(self,art,frame):
        """Remove the plate's own copy of backdrop overlays (the telephone string,
        stray specks). The source versions are put back by `restore_overlays`."""
        bg=background_color(frame)
        fg=((~red_background(art))&(self.roi>0)).astype(np.uint8)
        # Thin horizontal structures near where the source string runs: the plate's
        # own string, which never lines up with the boiling source one.
        kept=cv2.morphologyEx(fg,cv2.MORPH_OPEN,np.ones((15,1),np.uint8))
        near_string=cv2.dilate(self.source_overlay(frame).astype(np.uint8),np.ones((51,51),np.uint8))>0
        thin=(cv2.dilate(fg-kept,np.ones((5,5),np.uint8))>0)&near_string
        # Small islands detached from the character: hallucinated marks and glyph
        # pieces. Anything touching the character (hat feather, ribbon) stays.
        n,labels,stats,_=cv2.connectedComponentsWithStats(fg)
        big=cv2.dilate(np.isin(labels,[i for i in range(1,n) if stats[i,4]>=2500]).astype(np.uint8),np.ones((21,21),np.uint8))>0
        island=np.zeros((H,W),bool)
        for i in range(1,n):
            if stats[i,4]<2500 and not big[labels==i].any():island[labels==i]=True
        island=cv2.dilate(island.astype(np.uint8),np.ones((5,5),np.uint8))>0
        art=art.copy();art[(thin|island)&(self.roi>0)]=bg
        return art

    def source_overlay(self,frame):
        """Source pixels that belong to backdrop overlays: the string and lyric text.

        The string is a thin white line with an ink edge; anything thick and white
        (gloves, the source's hat ribbon) and the ink hugging it is excluded, so
        only long thin structures count as string. Lyric text has its own colour.
        """
        b,g,r=cv2.split(frame.astype(np.float32))
        whitish=(np.minimum.reduce([b,g,r])>165)&(np.maximum.reduce([b,g,r])-np.minimum.reduce([b,g,r])<45)
        ink=(np.maximum.reduce([b,g,r])<110)&(cv2.dilate(whitish.astype(np.uint8),np.ones((9,9),np.uint8))>0)
        line=((whitish|ink)&(~red_background(frame))).astype(np.uint8)
        dist=cv2.distanceTransform(line,cv2.DIST_L2,5)
        thick=cv2.dilate((dist>10).astype(np.uint8),np.ones((29,29),np.uint8))>0
        thin=line.copy();thin[thick]=0
        n,labels,stats,_=cv2.connectedComponentsWithStats(thin)
        string=np.isin(labels,[i for i in range(1,n) if max(stats[i,2],stats[i,3])>=120])
        # Lyric text shares its peach colour with the source's sweater; keep only thin strokes.
        text=((r>220)&(g>150)&(g<235)&(b>110)&(b<215)&(r-b>30)).astype(np.uint8)
        thick=cv2.dilate((cv2.distanceTransform(text,cv2.DIST_L2,5)>7).astype(np.uint8),np.ones((15,15),np.uint8))>0
        text[thick]=0
        return string|(text>0)

    def restore_overlays(self,frame,art,output):
        """Bring back the source string and lyric text wherever the plate shows backdrop."""
        plate_fg=cv2.dilate(((~red_background(art))&(self.roi>0)).astype(np.uint8),np.ones((7,7),np.uint8))>0
        overlay=self.source_overlay(frame)
        # Frame-blended edges (the text drops in with motion blur, the string
        # boils) sit between backdrop and overlay colour; take them along too.
        bg=background_color(frame)
        off_bg=np.abs(frame.astype(np.int16)-bg.astype(np.int16)).max(axis=2)>30
        b,g,r=cv2.split(frame)
        yellow=(r>200)&(g>150)&(b<100)
        halo=(cv2.dilate(overlay.astype(np.uint8),np.ones((7,7),np.uint8))>0)&off_bg&~yellow
        overlay=(overlay|halo)&(self.roi>0)&(~plate_fg)
        overlay=cv2.morphologyEx(overlay.astype(np.uint8),cv2.MORPH_CLOSE,np.ones((3,3),np.uint8))>0
        output=output.copy();output[overlay]=frame[overlay]
        return output

    def follow_band(self,frame,art,output):
        """Keep the source's boiling letterbox edges; the plate only fills the inside."""
        x1=min(r[0] for r in self.spec['regions']);x2=max(r[2] for r in self.spec['regions'])
        src_top,src_bot,solid=band_edges(frame,x1,x2)
        art_top,art_bot,_=band_edges(art,x1,x2)
        ys=np.arange(H)[:,None];xs=(np.arange(W)>=x1)&(np.arange(W)<x2)
        top=src_top[None,:];bottom=src_bot[None,:]
        source_in=(ys>=top+3)&(ys<=bottom-3)&xs
        plate_in=(ys>=art_top[None,:]+8)&(ys<=art_bot[None,:]-8)&(self.roi>0)
        b,g,r=cv2.split(frame)
        cream_px=frame[source_in&(b>200)&(g>215)&(r>215)]
        cream=np.median(cream_px,axis=0).astype(np.uint8) if len(cream_px) else np.array([238,235,217],np.uint8)
        bg=background_color(frame)
        result=frame.copy()
        # The source hair spilled over the outline; Lynn's does not, so clear it.
        spill=solid&(((ys<top)&(ys>=top-30))|((ys>bottom)&(ys<=bottom+30)))
        result[spill]=bg
        # Redraw the 3px ink outline where the hair covered it.
        on_edge=(((ys>=top)&(ys<top+3))|((ys>bottom-3)&(ys<=bottom)))&xs
        ink=frame.max(axis=2)<70
        result[on_edge&~ink]=(8,8,8)
        keep=source_in&plate_in
        result[keep]=output[keep]
        # Inside the source band but past the plate's own edge: fill with the band colour.
        result[source_in&~keep]=cream
        return result

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
        if basecap:
            baseok,baseframe=basecap.read();assert baseok
        active=[p for p in plates if round(p.spec['start']*FPS)<=n<round(p.spec['end']*FPS)]
        if active:
            gray=cv2.resize(cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY),(960,540))
            needs_features=any(not p.spec.get('static') and not p.spec.get('anatomical_head') and not p.spec.get('source_passthrough') for p in active)
            features=detector.detectAndCompute(gray,None) if needs_features else ([],None)
            for p in active:
                frame,confidence=p.apply(frame,gray,features,n)
                if confidence<8:report.append({'frame':n,'plate':p.spec['art'],'inliers':confidence})
        elif basecap:frame=baseframe
        frame=finish_visible_cuffs(frame,n/FPS)
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
