"""CPU float32 emulation of kernelPixelDetector_parallel_interpolated (software trilinear) for projection 0 of
test_forward_is_invariant_to_a_full_turn[parallel-interpolated] at angle 0.3 and 0.3+2pi (64^3 head phantom, accuracy 0.5).
The per-projection parameters are the ones the kernel received on gfx942 (printed from the device, identical host code
on every card). Usage: python fullturn_cpu_ref.py <TIGRE>/Python <gpu.npy from a run: stack([Ax(a), Ax(a+2pi)])>
Result 2026-11-20: 768/4096 rays get 257 samples at 0.3, none at 0.3+2pi; CPU full-turn 3.579e-4 = gfx942 3.579e-4;
CPU vs gfx942 1.3e-7 at both angles; forcing 256 samples gives 1.64e-5."""
import sys, numpy as np
sys.path.insert(0, sys.argv[1])
from tigre.utilities import sample_loader
import tigre
f=np.float32
geo=tigre.geometry(mode="parallel", nVoxel=np.array([64,64,64]))
img=sample_loader.load_head_phantom(geo.nVoxel).astype(np.float32)  # [z,y,x]
params={ "0.3":dict(src=(101.950424,20.3201942,63),uvO=(-20.3326492,-17.5063934,63),dU=(-0.295520782,0.955337524,0),dV=(0,0,-1)),
         "0.3+2pi":dict(src=(101.950424,20.320219,63),uvO=(-20.332634,-17.5064087,63),dU=(-0.295516968,0.95533371,0),dV=(0,0,-1))}
def fetch(ix,iy,iz):
    ok=(ix>=0)&(iy>=0)&(iz>=0)&(ix<64)&(iy<64)&(iz<64)
    out=np.zeros(ix.shape,f); out[ok]=img[iz[ok],iy[ok],ix[ok]]; return out
def tri(x,y,z):
    xb=f(x-f(.5)).astype(f); yb=(y-f(.5)).astype(f); zb=(z-f(.5)).astype(f)
    fx=np.floor(xb); fy=np.floor(yb); fz=np.floor(zb); dx=xb-fx; dy=yb-fy; dz=zb-fz
    ix=fx.astype(int); iy=fy.astype(int); iz=fz.astype(int)
    c=lambda a,b,cc: fetch(ix+a,iy+b,iz+cc)
    c00=c(0,0,0)*(f(1)-dx)+c(1,0,0)*dx; c10=c(0,1,0)*(f(1)-dx)+c(1,1,0)*dx
    c01=c(0,0,1)*(f(1)-dx)+c(1,0,1)*dx; c11=c(0,1,1)*(f(1)-dx)+c(1,1,1)*dx
    c0=c00*(f(1)-dy)+c10*dy; c1=c01*(f(1)-dy)+c11*dy
    return (c0*(f(1)-dz)+c1*dz).astype(f)
def proj(p, forcelen=None):
    src=np.array(p["src"],f); uvO=np.array(p["uvO"],f); dU=np.array(p["dU"],f); dV=np.array(p["dV"],f)
    v,u=np.meshgrid(np.arange(64),np.arange(64),indexing="ij"); pu=u.astype(f); pv=(63-v).astype(f)
    P=[((uvO[k]+pu*dU[k]).astype(f)+pv*dV[k]).astype(f) for k in range(3)]
    S=[((src[k]+pu*dU[k]).astype(f)+pv*dV[k]).astype(f) for k in range(3)]
    d=[(S[k]-P[k]).astype(f) for k in range(3)]
    L=np.sqrt((d[0]*d[0]+d[1]*d[1]+d[2]*d[2]).astype(f)).astype(np.float64)
    length=np.ceil(L/0.5).astype(f).astype(np.float64)
    if forcelen is not None: length[:]=forcelen
    vect=[((P[k]-S[k]).astype(np.float64)/length).astype(f) for k in range(3)]
    cap=np.float64((2*64/1.0+18)/0.5)
    length=np.where(cap<length, np.ceil(cap), length)
    s=np.zeros((64,64),f); i0=np.floor(f(18)/f(.5))
    nmax=int(length.max())
    for i in range(int(i0), nmax+1):
        act=i<=length; fi=f(i)
        tx=(vect[0]*fi+S[0]).astype(f); ty=(vect[1]*fi+S[1]).astype(f); tz=(vect[2]*fi+S[2]).astype(f)
        val=tri(tx+f(.5),ty+f(.5),tz+f(.5)); s=np.where(act,(s+val).astype(f),s)
    dl=np.sqrt((vect[0]*vect[0]+vect[1]*vect[1]+vect[2]*vect[2]).astype(f))
    return (s*dl).astype(f), length
g=np.load(sys.argv[2])
c0,l0=proj(params["0.3"]); c1,l1=proj(params["0.3+2pi"])
rel=lambda a,b: float(np.linalg.norm(a-b)/np.linalg.norm(b))
print("samples>256 at 0.3:",int((l0>256).sum())," at 0.3+2pi:",int((l1>256).sum()))
print("CPU full-turn rel diff proj0: %.3e"%rel(c1,c0))
print("GPU gfx942 full-turn rel diff proj0: %.3e"%rel(g[1][0],g[0][0]))
print("CPU vs GPU angle0.3: %.3e, angle0.3+2pi: %.3e"%(rel(c0,g[0][0]),rel(c1,g[1][0])))
c0f,_=proj(params["0.3"],forcelen=256.0)
print("CPU with sample count forced 256 at 0.3 vs 0.3+2pi: %.3e"%rel(c1,c0f))
