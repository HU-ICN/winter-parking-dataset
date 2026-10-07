import json,sys
from PIL import Image, ImageDraw
sel=json.load(open(sys.argv[1])); imgdir=sys.argv[2]; out=sys.argv[3]
sel.sort(key=lambda d:-d['conf']); tiles=[]
for d in sel:
    im=Image.open(f"{imgdir}/{d['frame']}").convert('RGB'); x0,y0,x1,y1=d['xyxy']; h=y1-y0; m=20
    c=im.crop((max(0,int(x0-m)),max(0,int(y0-m)),min(im.width,int(x1+m)),min(im.height,int(y1+m))))
    sc=3 if h<80 else (2 if h<200 else 1)
    c=c.resize((min(c.width*sc,520),min(c.height*sc,520)),Image.NEAREST) if c.width*sc<=520 and c.height*sc<=520 else c.resize((520,int(520*c.height/c.width)) if c.width>=c.height else (int(520*c.width/c.height),520))
    dr=ImageDraw.Draw(c); dr.rectangle([0,0,min(c.width,300),16],fill='black'); dr.text((3,2),f"{d['frame'][6:20]} {h:.0f}px c{d['conf']:.2f} x{sc}",fill='white'); tiles.append(c)
w=max(t.width for t in tiles); h=max(t.height for t in tiles); cols=8
for part in range(0,len(tiles),40):
    tt=tiles[part:part+40]; rows=(len(tt)+cols-1)//cols
    s=Image.new('RGB',(w*cols,h*rows),'white')
    for i,t in enumerate(tt): s.paste(t,((i%cols)*w,(i//cols)*h))
    s.save(f"{out}_{part//40+1}.jpg",quality=85); print(f"{out}_{part//40+1}.jpg",s.size,len(tt))
