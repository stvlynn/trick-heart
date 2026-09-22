"""Anatomical head replacement for the juggling shot; shoulders stay source-driven."""
import cv2
import numpy as np

W,H=1920,1080

def build_head_masks(art,red_background):
    # Stop the removal mask at the jaw/neck, above the original moving shoulders.
    erase=np.zeros((H,W),np.uint8)
    cv2.fillPoly(erase,[np.array([[1190,0],[1790,0],[1790,450],[1570,450],[1540,478],[1508,450],[1190,450]],np.int32)],255)
    bounds=np.zeros_like(erase)
    cv2.fillPoly(bounds,[np.array([[1320,48],[1740,48],[1740,443],[1665,481],[1595,485],[1570,454],[1540,480],[1510,454],[1477,484],[1385,484],[1320,440]],np.int32)],255)
    fg=(~red_background(art)).astype(np.uint8)*255
    head=cv2.bitwise_and(bounds,fg)
    b,g,r=cv2.split(art.astype(np.float32)); yy=np.arange(H)[:,None]
    navy=(b<135)&(g<120)&(r<95)&(b>r*1.15)&(yy>445)
    head[navy]=0
    return erase,head

def composite_head(frame,art,erase,head,mat,bg):
    warped=cv2.warpAffine(art,mat,(W,H),flags=cv2.INTER_LINEAR,borderValue=tuple(map(int,bg)))
    removal=cv2.warpAffine(erase,mat,(W,H),flags=cv2.INTER_LINEAR).astype(np.float32)[...,None]/255
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
    # Gloves and cuffs remain in front of the head as the hands rise.
    hands_region=np.zeros((H,W),np.uint8)
    poly=np.array([[[1190,355],[1780,355],[1820,1080],[1170,1080]]],np.float32)
    cv2.fillPoly(hands_region,[cv2.transform(poly,mat)[0].astype(np.int32)],255)
    neutral=(np.maximum.reduce([b,g,r])-np.minimum.reduce([b,g,r])<22)&(b>190)&(g>190)&(r>190)
    orange=(r>180)&(g>80)&(g<205)&(b<100)&(g<r*.8)
    props[(neutral|orange)&(hands_region>0)]=255
    props=cv2.dilate(props,np.ones((3,3),np.uint8))
    out[props>0]=frame[props>0]
    return out
