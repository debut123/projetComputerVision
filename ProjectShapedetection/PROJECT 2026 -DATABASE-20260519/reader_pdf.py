import numpy as np, warnings; warnings.filterwarnings("ignore")
from skimage import color, filters, morphology, measure
from scipy.ndimage import binary_fill_holes
from scipy import ndimage
from skimage.transform import ProjectiveTransform, warp, resize
import fitz

# ---------- PDF ----------
def pdf_to_pages(path, dpi=200):
    try:
        from pdf2image import convert_from_path
        return [np.array(p)[:,:,:3] for p in convert_from_path(path, dpi=dpi)]
    except Exception:
        z=dpi/72.0; out=[]
        with fitz.open(path) as doc:
            for pg in doc:
                pix=pg.get_pixmap(matrix=fitz.Matrix(z,z),alpha=False)
                out.append(np.frombuffer(pix.samples,np.uint8).reshape(pix.height,pix.width,pix.n)[:,:,:3].copy())
        return out

# ---------- low-level preprocess (recette élève) ----------
def preprocess(im, size=5, mn=100):
    im2=color.rgb2gray(im[:,:,:3]); im3=filters.sobel(im2)
    im4=im3>filters.threshold_otsu(im3)*0.7
    im4=morphology.remove_small_objects(im4,min_size=mn)
    im4=ndimage.binary_dilation(im4,structure=np.ones((size,size)))
    im4=morphology.binary_closing(im4,morphology.square(size)); im4=binary_fill_holes(im4)
    lab=measure.label(im4,connectivity=2); return im2,im4,lab,measure.regionprops(lab)

# ---------- coins + recalage (bas niveau) ----------
def find_corners(imlabel, rp, thr=0.22):
    if len(rp)<4: return None
    TL,TR,BL,BR=[],[],[],[]
    with np.errstate(invalid='ignore'):
        for k in range(len(rp)):
            r0,c0,r1,c1=rp[k].bbox; pat=imlabel[r0:r1,c0:c1]>0; dL,dl=pat.shape
            M=np.ones((dL,dl)); e=int(min(dL,dl)*0.20); M[e:,e:]=0
            TL.append(np.corrcoef(M.flatten(),pat.flatten())[0,1]); TR.append(np.corrcoef(np.rot90(M,3).flatten(),pat.flatten())[0,1])
            BL.append(np.corrcoef(np.rot90(M,1).flatten(),pat.flatten())[0,1]); BR.append(np.corrcoef(np.rot90(M,2).flatten(),pat.flatten())[0,1])
    tabs=[np.nan_to_num(np.array(x),nan=-1.0) for x in (TL,TR,BL,BR)]; srt=[np.argsort(t) for t in tabs]
    top=np.array([s[-1] for s in srt]); pos=[-1,-1,-1,-1]; safety=0
    while len(np.unique(top))!=4 and safety<4*len(rp)+10:
        safety+=1; vals,cnt=np.unique(top,return_counts=True)
        for val in vals[cnt>1]:
            idxs=np.where(top==val)[0]; best=idxs[np.argmax([tabs[i][top[i]] for i in idxs])]
            for i in idxs:
                if i!=best:
                    pos[i]-=1
                    if pos[i] < -len(srt[i]): return None      # garde-fou (page blanche)
                    top[i]=srt[i][pos[i]]
    if len(np.unique(top))!=4: return None
    pts=[]
    for i in range(4):
        if tabs[i][top[i]]<thr: return None
        r0,c0,r1,c1=rp[top[i]].bbox; pts.append([(c0,r0),(c1,r0),(c0,r1),(c1,r1)][i])
    return pts

def normalize_page(im, corners, W=1654, H=2339):
    im2=color.rgb2gray(im[:,:,:3])
    if corners is None: return resize(im2,(H,W),anti_aliasing=True)
    src=np.array([[0,0],[W,0],[0,H],[W,H]],float); t=ProjectiveTransform()
    try:
        ok=t.estimate(src,np.array(corners,float))
    except Exception:
        ok=False
    if not ok: return resize(im2,(H,W),anti_aliasing=True)
    return warp(im2,t,output_shape=(H,W),cval=1.0)

def norm_gray(im):
    im2,im4,lab,rp=preprocess(im); return normalize_page(im, find_corners(lab,rp))

# ---------- détection de cases carrées (regionprops, méthode élève) ----------
def detect_boxes(gray_region, size=3, mn=60, mind=12, dens=0.82):
    im3=filters.sobel(gray_region); 
    try: thr=filters.threshold_otsu(im3)*0.7
    except Exception: thr=0.1
    b=im3>thr; b=morphology.remove_small_objects(b,min_size=mn)
    b=ndimage.binary_dilation(b,structure=np.ones((size,size))); b=morphology.binary_closing(b,morphology.square(size)); b=binary_fill_holes(b)
    lab=measure.label(b,connectivity=2); rp=measure.regionprops(lab); out=[];sz=[]
    for k in range(len(rp)):
        r0,c0,r1,c1=rp[k].bbox; pat=lab[r0:r1,c0:c1]>0; dL,dl=r1-r0,c1-c0
        if dl<mind or dL<mind: continue
        if pat.sum()/(dL*dl)>dens and 0.7<dL/dl<1.4:
            out.append([(c0+c1)/2,(r0+r1)/2,(r0,c0,r1,c1)]); sz.append(dl*dL)
    if not sz: return []
    med=np.median(sz)
    return [o for o in out if abs((o[2][2]-o[2][0])*(o[2][3]-o[2][1])-med)<0.5*med]

def interior_ink(gray, bbox, sh=0.30, t=0.45):
    r0,c0,r1,c1=bbox; h=r1-r0; w=c1-c0
    cell=gray[int(r0+h*sh):int(r1-h*sh), int(c0+w*sh):int(c1-w*sh)]
    return float((cell<t).mean()) if cell.size else 0.0

def _gap_cluster(vals, k):
    idx=np.argsort(vals); sv=np.array(vals)[idx]; gaps=np.diff(sv)
    cut=set(np.argsort(gaps)[-(k-1):]) if k>1 and len(gaps)>=k-1 else set()
    ol=np.zeros(len(sv),int); c=0
    for i in range(len(sv)):
        ol[i]=c
        if i in cut: c+=1
    lab=np.zeros(len(vals),int); lab[idx]=ol
    cen=[np.mean([vals[j] for j in range(len(vals)) if lab[j]==ci]) if any(lab==ci) else 1e9 for ci in range(k)]
    return lab, cen

def read_grid(gray, x0f,x1f,y0f,y1f, n_cols, n_rows, lo=0.12, hi=0.85, size=3):
    """Grille de bulles -> liste d'index (un par colonne). Bas niveau de bout en bout."""
    H,W=gray.shape; x0,x1,y0,y1=int(x0f*W),int(x1f*W),int(y0f*H),int(y1f*H)
    reg=gray[y0:y1, x0:x1]; boxes=detect_boxes(reg,size=size)
    if len(boxes) < n_cols*2: return [None]*n_cols
    xs=[b[0] for b in boxes]; ys=[b[1] for b in boxes]
    clab,cc=_gap_cluster(xs,n_cols); rlab,rc=_gap_cluster(ys,n_rows)
    cmap={o:n for n,o in enumerate(np.argsort(cc))}; rmap={o:n for n,o in enumerate(np.argsort(rc))}
    grid={}
    for i,b in enumerate(boxes):
        key=(cmap[clab[i]],rmap[rlab[i]]); grid[key]=max(grid.get(key,0),interior_ink(reg,b[2]))
    out=[]
    for ci in range(n_cols):
        col=[(ri,grid.get((ci,ri),0)) for ri in range(n_rows)]; ri,v=max(col,key=lambda t:t[1])
        out.append(ri if lo<v<hi else None)
    return out

def read_digit_grid(gray, x0f,x1f,y0f,y1f, n_digits):
    r=read_grid(gray,x0f,x1f,y0f,y1f,n_digits,10)
    return "".join(str(d) if d is not None else "?" for d in r)

def read_group_grid(gray, x0f,x1f,y0f,y1f):
    r=read_grid(gray,x0f,x1f,y0f,y1f,3,10)
    d=[str(x) if x is not None else "?" for x in r[:2]]
    let=chr(ord('A')+r[2]) if r[2] is not None else "?"
    return "G"+"".join(d)+let

# ---------- QCM sur pages de questions : strip de gauche + groupage par trous ----------
def read_qcm_page(im, x0f=0.04, x1f=0.12, choices="ABCDEFGH", lo=0.12, hi=0.80):
    H,W=im.shape[:2]; x0,x1=int(x0f*W),int(x1f*W)
    strip=im[:, x0:x1]; gray=color.rgb2gray(strip[:,:,:3])
    boxes=detect_boxes(gray, size=3, mn=60, mind=12)
    boxes=sorted(boxes, key=lambda b:b[1])
    if not boxes: return []
    # groupage vertical par trous (1 groupe = 1 question)
    ys=[b[1] for b in boxes]; dys=np.diff(ys)
    typ=np.median([d for d in dys if d>3]) if len(dys) else 1
    groups=[]; cur=[boxes[0]]
    for i in range(1,len(boxes)):
        if ys[i]-ys[i-1] > 1.8*typ: groups.append(cur); cur=[]
        cur.append(boxes[i])
    groups.append(cur)
    res=[]
    for g in groups:
        g=sorted(g,key=lambda b:b[1]); ans=[]
        for j,b in enumerate(g):
            v=interior_ink(gray,b[2])
            if lo<v<hi and j<len(choices): ans.append(choices[j])
        res.append(ans)
    return res   # liste de questions -> liste de lettres cochées

# ---------- conditions YES/NO (remplissage gris = sélection) ----------
def read_conditions(gray, cols_x, yes_y, no_y, box_h=0.018, box_w=0.018):
    """cols_x: 5 fractions x ; yes_y,no_y: fractions y des cases YES et NO. Compare l'encre."""
    H,W=gray.shape; out=[]
    def ink_at(xf,yf):
        x,y=int(xf*W),int(yf*H); hw=int(box_w*W/2); hh=int(box_h*H/2)
        cell=gray[y-hh:y+hh, x-hw:x+hw]; return float((cell<0.6).mean()) if cell.size else 0
    for xf in cols_x:
        out.append(1 if ink_at(xf,yes_y) >= ink_at(xf,no_y) else 0)
    return out

# ---------- cryptogramme (petit graphique bas de page, identique sur ttes les pages) ----------
def crypto_sig(im_or_gray, roi=(0.235,0.915,0.07,0.06)):
    """Signature binaire du cryptogramme. Travaille sur l'image BRUTE (le repere
    est a position physique stable) ; renvoie None si la zone est vide/blanche."""
    g = color.rgb2gray(im_or_gray[:,:,:3]) if im_or_gray.ndim==3 else im_or_gray
    H,W=g.shape; x,y,w,h=roi
    sub=g[int(y*H):int((y+h)*H), int(x*W):int((x+w)*W)]
    if sub.size==0: return None
    b=(sub<0.5).astype(float)
    return b if b.sum()>0 else None

def _ncc(a,b):
    a=a.ravel()-a.mean(); b=b.ravel()-b.mean(); d=np.linalg.norm(a)*np.linalg.norm(b)
    return 1.0 if d<1e-9 else float(a@b/d)

def _ncc_shift(a,b,mx=6):
    """NCC tolerante a un petit decalage (alignement de scan) : max sur translations."""
    best=-1.0
    for dy in range(-mx,mx+1):
        for dx in range(-mx,mx+1):
            best=max(best,_ncc(a,np.roll(np.roll(b,dy,0),dx,1)))
    return best

def cryptograms_identical(sigs, thr=0.60):
    sigs=[s for s in sigs if s is not None]
    if len(sigs)<2: return 1
    return int(all(_ncc_shift(sigs[0],s)>=thr for s in sigs[1:]))

# ---------- pages d'examen : blocs question (numerotation) + QCM par bloc ----------
def question_blocks(im):
    H,W=im.shape[:2]; im2=color.rgb2gray(im[:,:,:3]); im3=filters.sobel(im2)
    b=im3>filters.threshold_otsu(im3)*0.7; b=morphology.remove_small_objects(b,min_size=100)
    b=ndimage.binary_dilation(b,structure=np.ones((5,5))); b=morphology.binary_closing(b,morphology.square(5)); b=binary_fill_holes(b)
    lab=measure.label(b,connectivity=2); rp=measure.regionprops(lab); blocks=[]
    for r in rp:
        r0,c0,r1,c1=r.bbox; h=r1-r0; w=c1-c0
        if w>0.55*W and 0.05*H<h<0.62*H: blocks.append((r0,r1))
    blocks=sorted(blocks)
    # fusionner chevauchements
    merged=[]
    for b0,b1 in blocks:
        if merged and b0<=merged[-1][1]: merged[-1]=(merged[-1][0],max(merged[-1][1],b1))
        else: merged.append((b0,b1))
    return merged

def read_exam_page(im, x0f=0.04, x1f=0.12, choices="ABCDEFGH", lo=0.12, hi=0.80):
    """Retourne une liste (1 par question du bas vers le haut deja triee) :
       {'type':'QCM','choices':[...]} ou {'type':'NUM'} (mantisse/exposant = hooks)."""
    H,W=im.shape[:2]; blocks=question_blocks(im)
    gray_strip=color.rgb2gray(im[:, int(x0f*W):int(x1f*W)][:,:,:3]) if im.ndim==3 else im[:, int(x0f*W):int(x1f*W)]
    allboxes=detect_boxes(gray_strip, size=3, mn=60, mind=12)
    out=[]
    for (b0,b1) in blocks:
        bx=[bb for bb in allboxes if b0<=bb[1]<=b1]
        if bx:
            bx=sorted(bx,key=lambda b:b[1]); ans=[]
            for j,b in enumerate(bx):
                v=interior_ink(gray_strip,b[2])
                if lo<v<hi and j<len(choices): ans.append(choices[j])
            out.append({'type':'QCM','choices':ans})
        else:
            out.append({'type':'NUM'})
    return out
