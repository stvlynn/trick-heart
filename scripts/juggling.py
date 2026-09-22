"""Anatomical head replacement for the juggling shot; shoulders stay source-driven.

Only the hat, hair, face and neck of the Lynn plate are pasted. The source
shoulders, arms, gloves, balls and shirt collar are kept, and the source
jacket/bow tie are recoloured to the Lynn costume by `finish_visible_cuffs`.
The plate is placed per source drawing by `track_juggling.py`, anchored to the
collar, so the neck always meets the shoulders that are actually on screen.
"""
import cv2
import numpy as np

W,H=1920,1080
BOW_TIE=np.array([213,174,76],np.float32)   # BGR cyan used for the Lynn bow tie/cuffs

def build_head_masks(art,red_background):
    # Static removal region in plate space: the whole head above the collar,
    # with a notch reaching down over the source neck to the collar line.
    erase=np.zeros((H,W),np.uint8)
    cv2.fillPoly(erase,[np.array([[1150,0],[1860,0],[1860,450],[1575,450],[1540,478],[1505,450],[1150,450]],np.int32)],255)
    bounds=np.zeros_like(erase)
    cv2.fillPoly(bounds,[np.array([[1320,48],[1740,48],[1740,443],[1665,481],[1595,485],[1570,454],[1540,480],[1510,454],[1477,484],[1385,484],[1320,440]],np.int32)],255)
    fg=(~red_background(art)).astype(np.uint8)*255
    head=cv2.bitwise_and(bounds,fg)
    b,g,r=cv2.split(art.astype(np.float32)); yy=np.arange(H)[:,None]
    navy=(b<135)&(g<120)&(r<95)&(b>r*1.15)&(yy>445)
    head[navy]=0
    return erase,head

def source_head_mask(frame,anchor,scale):
    """Crimson hair and skin of the source character that sit above the collar.

    The source is redrawn every few frames, so the static polygon alone can
    miss a curl or the neck. Hair and jacket share one colour; a component is
    only treated as hair when it does not run down into the jacket.

    Returns (removal mask, face hull mask). The hull covers the eyes and mouth,
    so it can tell eye whites apart from gloves held in front of the face.
    """
    anchor_x,anchor_y=anchor
    b,g,r=cv2.split(frame.astype(np.float32))
    hair=(r>150)&(g<90)&(b>g*1.3)&(b<r*.6)
    # Pale pink skin; excludes the white gloves, the cream frame border and the credits text.
    skin=(r>230)&(g>195)&(b>195)&(r-b>12)&(r-b<60)&(np.abs(g-b)<14)
    mask=np.zeros((H,W),np.uint8);face=np.zeros((H,W),np.uint8)
    for cand,is_skin in ((hair,False),(skin,True)):
        n,labels,stats,cents=cv2.connectedComponentsWithStats(cand.astype(np.uint8))
        for i in range(1,n):
            x,y,w,h,area=stats[i]
            if area<300 or abs(cents[i][0]-anchor_x)>420*scale:continue
            if y<anchor_y-60*scale and y+h<anchor_y+15*scale:
                mask[labels==i]=255
                if is_skin:face[labels==i]=255
    # The eyes touch the bangs, so they are not holes in the skin; use the hull.
    points=cv2.findNonZero(face)
    if points is not None:
        cv2.fillConvexPoly(face,cv2.convexHull(points),255)
    return cv2.dilate(mask,np.ones((7,7),np.uint8)),face

def composite_head(frame,art,erase,head,mat,bg,anchor=None):
    warped=cv2.warpAffine(art,mat,(W,H),flags=cv2.INTER_LINEAR,borderValue=tuple(map(int,bg)))
    removal=cv2.warpAffine(erase,mat,(W,H),flags=cv2.INTER_LINEAR)
    face=np.zeros((H,W),np.uint8)
    if anchor is not None:
        dynamic,face=source_head_mask(frame,anchor,float(np.hypot(mat[0,0],mat[1,0])))
        removal=np.maximum(removal,dynamic)
    removal=removal.astype(np.float32)[...,None]/255
    alpha=cv2.warpAffine(head,mat,(W,H),flags=cv2.INTER_LINEAR).astype(np.float32)[...,None]/255
    cleaned=frame.astype(np.float32)*(1-removal)+np.asarray(bg,dtype=np.float32)*removal
    out=(cleaned*(1-alpha)+warped.astype(np.float32)*alpha).astype(np.uint8)
    # Restore the actual frame border, never the border from the generated plate.
    b,g,r=cv2.split(frame.astype(np.float32))
    cream=((r>190)&(g>180)&(b>160)&(r>b+4)).astype(np.uint8)
    n,labels,stats,_=cv2.connectedComponentsWithStats(cream)
    keep=[i for i in range(1,n) if stats[i,4]>1000 and (stats[i,0]==0 or stats[i,1]==0 or stats[i,0]+stats[i,2]>=W or stats[i,1]+stats[i,3]>=H)]
    border=cv2.dilate(np.isin(labels,keep).astype(np.uint8),np.ones((7,7),np.uint8))
    out[border>0]=frame[border>0]
    # Balls use their contour, preventing a circular patch of the old head showing through.
    balls=((r>200)&(g>165)&(b<90)).astype(np.uint8)
    n,labels,stats,_=cv2.connectedComponentsWithStats(balls)
    props=np.zeros((H,W),np.uint8)
    for i in range(1,n):
        if stats[i,4]<100:continue
        points=cv2.findNonZero((labels==i).astype(np.uint8))
        cv2.fillConvexPoly(props,cv2.convexHull(points),255)
    # Gloves and cuffs remain in front of the head as the hands rise; their ink
    # outlines come along so a hand over the hair keeps its edge.
    hands_region=np.zeros((H,W),np.uint8)
    poly=np.array([[[1150,355],[1860,355],[1900,1080],[1120,1080]]],np.float32)
    cv2.fillPoly(hands_region,[cv2.transform(poly,mat)[0].astype(np.int32)],255)
    neutral=(np.maximum.reduce([b,g,r])-np.minimum.reduce([b,g,r])<22)&(b>190)&(g>190)&(r>190)
    orange=(r>180)&(g>80)&(g<205)&(b<100)&(g<r*.8)
    hands=((neutral|orange)&(hands_region>0)).astype(np.uint8)
    hands=cv2.morphologyEx(hands,cv2.MORPH_OPEN,np.ones((3,3),np.uint8))
    # Drop specks and the eye whites: anything small that lies inside the source
    # face is part of the face being replaced, not a glove in front of it.
    n,labels,stats,_=cv2.connectedComponentsWithStats(hands)
    face_area=max(int(np.count_nonzero(face)),1)
    keep=[]
    for i in range(1,n):
        area=stats[i,4]
        if area<600:continue
        inside=np.count_nonzero(face[labels==i])/area
        if inside>.7 and area<.08*face_area:continue
        keep.append(i)
    hands=np.isin(labels,keep).astype(np.uint8)
    outline=(np.maximum.reduce([b,g,r])<90)&(cv2.dilate(hands,np.ones((9,9),np.uint8))>0)
    props[hands>0]=255;props[outline]=255
    props=cv2.dilate(props,np.ones((3,3),np.uint8))
    out[props>0]=frame[props>0]
    if anchor is not None:
        recolor_bow_tie(out,frame,mat,anchor)
    return out

def recolor_bow_tie(out,frame,mat,anchor):
    """Turn the source's black bow tie cyan, keeping its ink shading."""
    scale=float(np.hypot(mat[0,0],mat[1,0]))
    ax,ay=anchor
    x1,x2=int(ax-95*scale),int(ax+95*scale);y1,y2=int(ay-6*scale),int(ay+80*scale)
    x1,y1=max(0,x1),max(0,y1);x2,y2=min(W,x2),min(H,y2)
    roi=frame[y1:y2,x1:x2]
    dark=(roi.max(axis=2)<80).astype(np.uint8)
    n,labels,stats,_=cv2.connectedComponentsWithStats(dark)
    tie=np.isin(labels,[i for i in range(1,n) if stats[i,4]>150]).astype(np.uint8)
    # Fill the lobes cyan but leave a thin ink rim so the tie keeps its outline.
    fill=cv2.erode(tie,cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(7,7)))>0
    out[y1:y2,x1:x2][fill]=BOW_TIE.astype(np.uint8)
