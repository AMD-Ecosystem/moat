"""Ray sample counts ceil(|S-P|/0.5) for every pixel of projection 0 at angle 0.3 and 0.3+2pi under the three ways a
compiler may contract the squared norm dx*dx+dy*dy+dz*dz (dz is 0 here). gfx90a/gfx942 compile it as rd(dx*dx)+rd(dy*dy)
(v_pk_mul_f32 + add): 768 rays get 257 samples at 0.3. gfx1201 compiles fma(dz,dz,fma(dx,dx,rd(dy*dy))): 256 everywhere."""
import numpy as np
f=np.float32
def fma(a,b,c): return f(np.float64(a)*np.float64(b)+np.float64(c))
params={ "0.3":dict(src=(101.950424,20.3201942,63),uvO=(-20.3326492,-17.5063934,63),dU=(-0.295520782,0.955337524,0),dV=(0,0,-1)),
         "0.3+2pi":dict(src=(101.950424,20.320219,63),uvO=(-20.332634,-17.5064087,63),dU=(-0.295516968,0.95533371,0),dV=(0,0,-1))}
def run(p, sp_fma, norm):
    src=[f(v) for v in p["src"]]; uvO=[f(v) for v in p["uvO"]]; dU=[f(v) for v in p["dU"]]; dV=[f(v) for v in p["dV"]]
    hist={}; pix=None
    for v in range(64):
        for u in range(64):
            pu=f(u); pv=f(63-v)
            if sp_fma: P=[fma(pv,dV[k],fma(pu,dU[k],uvO[k])) for k in range(3)]; S=[fma(pv,dV[k],fma(pu,dU[k],src[k])) for k in range(3)]
            else: P=[f(f(uvO[k]+f(pu*dU[k]))+f(pv*dV[k])) for k in range(3)]; S=[f(f(src[k]+f(pu*dU[k]))+f(pv*dV[k])) for k in range(3)]
            d=[f(S[k]-P[k]) for k in range(3)]
            if norm=="plain": q=f(f(f(d[0]*d[0])+f(d[1]*d[1]))+f(d[2]*d[2]))
            elif norm=="fma x first": q=fma(d[2],d[2],fma(d[1],d[1],f(d[0]*d[0])))
            elif norm=="fma y first": q=fma(d[2],d[2],fma(d[0],d[0],f(d[1]*d[1])))
            L=np.sqrt(q,dtype=f); n=int(np.ceil(np.float64(L)/0.5)); hist[n]=hist.get(n,0)+1
            if (u,v)==(9,31): pix=(float(L),n)
    return dict(sorted(hist.items())), pix
for sp in (False,True):
    for norm in ("plain","fma x first","fma y first"):
        r=[run(params[k],sp,norm) for k in params]
        print("S/P fma=%-5s norm=%-12s  0.3: %s pix %s | 0.3+2pi: %s pix %s"%(sp,norm,r[0][0],r[0][1],r[1][0],r[1][1]))
