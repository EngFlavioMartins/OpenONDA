"""Independent f64 equations, manufactured flows, and test-only integrators."""
from __future__ import annotations
from dataclasses import dataclass, asdict
import numpy as np
from scipy.integrate import solve_ivp
from scipy.linalg import expm
from scipy.special import erf

Q4PI = 1/(4*np.pi); PI15 = np.pi**-1.5; EPS = 1e-14

def q(r): return (erf(r)-2/np.sqrt(np.pi)*r*np.exp(-r*r))*Q4PI
def zeta(r): return PI15*np.exp(-np.asarray(r)**2)
def winckelmans_q(r):
    r=np.asarray(r); base=r*r+1; return r**3*(r*r+2.5)/(base*base*np.sqrt(base))*Q4PI
def winckelmans_zeta(r):
    r=np.asarray(r); base=r*r+1; return 7.5/(base**3*np.sqrt(base))*Q4PI
def kernel_pair(kernel): return (winckelmans_q,winckelmans_zeta) if kernel.upper()=='WINCKELMANS' else (q,zeta)
def gfun(r):
    r=np.asarray(r); out=np.empty_like(r); small=np.abs(r)<1e-8
    out[small]=PI15*(.5-r[small]**2/6); out[~small]=erf(r[~small])/r[~small]*Q4PI
    return out

def pair_geometry(x,sigma):
    d=x[:,None,:]-x[None,:,:]; d2=np.einsum('ijk,ijk->ij',d,d); dist=np.sqrt(d2)
    sig=.5*(sigma[:,None]+sigma[None,:]); mask=dist>EPS; safe=np.where(mask,dist,1.)
    return d,d2,safe,sig,safe/sig,mask

def velocity(x,gamma,sigma,kernel='GAUSSIAN'):
    d,_,safe,_,rho,mask=pair_geometry(x,sigma)
    qf,_=kernel_pair(kernel); w=np.where(mask,qf(rho)/safe**3,0.)
    return -np.einsum('ij,ijk->ik',w,np.cross(d,gamma[None,:,:]))

def gradient(x,gamma,sigma,kernel='GAUSSIAN'):
    d,d2,safe,sig,rho,mask=pair_geometry(x,sigma); qf,zf=kernel_pair(kernel); qq=qf(rho)
    a=np.where(mask,qq/safe**3,0.); b=np.where(mask,3*qq/safe**5-zf(rho)/sig**3/np.where(mask,d2,1.),0.)
    skew=np.zeros((len(gamma),3,3)); skew[:,0,1]=-gamma[:,2]; skew[:,0,2]=gamma[:,1]
    skew[:,1,0]=gamma[:,2]; skew[:,1,2]=-gamma[:,0]; skew[:,2,0]=-gamma[:,1]; skew[:,2,1]=gamma[:,0]
    return np.einsum('ij,jab->iab',a,skew)+np.einsum('ij,ija,ijb->iab',b,np.cross(d,gamma[None,:,:]),d)

def contract(j,gamma,mode):
    op=j if mode=='DIRECT' else j.transpose(0,2,1) if mode=='TRANSPOSED' else .5*(j+j.transpose(0,2,1))
    return np.einsum('nij,nj->ni',op,gamma)

def pair_rate(x,gamma,sigma,mode,kernel='GAUSSIAN'):
    d,_,safe,sig,rho,mask=pair_geometry(x,sigma); qf,zf=kernel_pair(kernel); qq=qf(rho); zz=zf(rho)
    a=np.where(mask,qq/safe**3,0.); b=np.where(mask,(3*qq-zz*rho**3)/(sig**5*rho**5),0.)
    gi=gamma[:,None,:]; gj=gamma[None,:,:]; gx=np.cross(gi,gj); rx=np.cross(d,gj)
    gir=np.einsum('ijk,ijk->ij',gi,d); girx=np.einsum('ijk,ijk->ij',gi,rx)
    if mode=='DIRECT': c=-a[...,None]*gx+b[...,None]*gir[...,None]*rx
    elif mode=='TRANSPOSED': c=a[...,None]*gx+b[...,None]*girx[...,None]*d
    else: c=.5*b[...,None]*(gir[...,None]*rx+girx[...,None]*d)
    return c.sum(axis=1)

def target_fields(target,x,gamma,sigma,kernel='GAUSSIAN'):
    d=target[:,None,:]-x[None,:,:]; d2=np.einsum('mnj,mnj->mn',d,d); dist=np.sqrt(d2)
    mask=dist>EPS; safe=np.where(mask,dist,1.); rho=safe/sigma[None,:]; qf,zf=kernel_pair(kernel)
    a=np.where(mask,qf(rho)/safe**3,0.); b=np.where(mask,3*qf(rho)/safe**5-zf(rho)/sigma[None,:]**3/np.where(mask,d2,1.),0.)
    u=-np.einsum('mn,mnj->mj',a,np.cross(d,gamma[None,:,:])); w=np.einsum('mn,nj->mj',np.where(mask,zf(rho)/sigma[None,:]**3,0.),gamma)
    skew=np.zeros((len(gamma),3,3)); skew[:,0,1]=-gamma[:,2]; skew[:,0,2]=gamma[:,1]
    skew[:,1,0]=gamma[:,2]; skew[:,1,2]=-gamma[:,0]; skew[:,2,0]=-gamma[:,1]; skew[:,2,1]=gamma[:,0]
    j=np.einsum('mn,nab->mab',a,skew)+np.einsum('mn,mna,mnb->mab',b,np.cross(d,gamma[None,:,:]),d)
    return u,w,j

@dataclass
class State:
    x: np.ndarray; gamma: np.ndarray
    def copy(self): return State(self.x.copy(),self.gamma.copy())

@dataclass
class Counts:
    velocity_evaluations:int=0; gradient_evaluations:int=0; fused_evaluations:int=0
    pairwise_stretching_sweeps:int=0; tree_builds:int=0; tree_refits:int=0
    tree_traversals:int=0; kernel_launches:int=0; synchronisations:int=0; host_device_transfers:int=0

class AnalyticEvaluator:
    def __init__(self,flow): self.flow=flow; self.counts=Counts()
    def vel(self,x,g,t): self.counts.velocity_evaluations+=1; return self.flow.velocity(x,t)
    def grad(self,x,g,t): self.counts.gradient_evaluations+=1; return self.flow.gradient(x,t)
    def rate(self,x,g,t,mode): return contract(self.grad(x,g,t),g,mode)
    def rhs(self,x,g,t,mode): self.counts.fused_evaluations+=1; return self.flow.velocity(x,t),contract(self.flow.gradient(x,t),g,mode)

class ParticleEvaluator:
    def __init__(self,sigma,implementation='pairwise',kernel='GAUSSIAN'): self.sigma=np.asarray(sigma); self.implementation=implementation; self.kernel=kernel; self.counts=Counts()
    def vel(self,x,g,t): self.counts.velocity_evaluations+=1; return velocity(x,g,self.sigma,self.kernel)
    def grad(self,x,g,t): self.counts.gradient_evaluations+=1; return gradient(x,g,self.sigma,self.kernel)
    def rate(self,x,g,t,mode):
        if self.implementation=='pairwise': self.counts.pairwise_stretching_sweeps+=1; return pair_rate(x,g,self.sigma,mode,self.kernel)
        return contract(self.grad(x,g,t),g,mode)
    def rhs(self,x,g,t,mode): return self.vel(x,g,t),self.rate(x,g,t,mode)

def ledger_add(ledger,method,step,stage,t,x,source,contraction,operation):
    if ledger is not None: ledger.append(dict(method=method,step=step,stage=stage,stage_time=t,position_state=x,source_strength_state=source,contraction_strength_state=contraction,field_operation=operation))

def xstep(method,e,s,t,dt,ledger,step,tag):
    x0,g=s.x,s.gamma; ledger_add(ledger,method,step,tag+'.k1',t,'x_n',tag+':fixed','none','velocity'); k1=e.vel(x0,g,t)
    x1=x0+dt*k1; ledger_add(ledger,method,step,tag+'.k2',t+dt,'x_1',tag+':fixed','none','velocity'); k2=e.vel(x1,g,t+dt)
    x2=.75*x0+.25*(x1+dt*k2); ledger_add(ledger,method,step,tag+'.k3',t+.5*dt,'x_2',tag+':fixed','none','velocity'); k3=e.vel(x2,g,t+.5*dt)
    return State(x0/3+2/3*(x2+dt*k3),g.copy())

def gstep(method,e,s,t,dt,mode,ledger,step,tag):
    x,g0=s.x,s.gamma; ledger_add(ledger,method,step,tag+'.k1',t,tag+':fixed','Gamma_n','Gamma_n','stretching'); k1=e.rate(x,g0,t,mode)
    g1=g0+dt*k1; ledger_add(ledger,method,step,tag+'.k2',t+dt,tag+':fixed','Gamma_1','Gamma_1','stretching'); k2=e.rate(x,g1,t+dt,mode)
    g2=.75*g0+.25*(g1+dt*k2); ledger_add(ledger,method,step,tag+'.k3',t+.5*dt,tag+':fixed','Gamma_2','Gamma_2','stretching'); k3=e.rate(x,g2,t+.5*dt,mode)
    return State(x.copy(),g0/3+2/3*(g2+dt*k3))

def coupled(e,s,t,dt,mode,order,ledger,method,step):
    x,g=s.x,s.gamma; ledger_add(ledger,method,step,'k1',t,'x_n','Gamma_n','Gamma_n','velocity+stretching'); ax,ag=e.rhs(x,g,t,mode)
    if order==2:
        x1,g1=x+dt*ax,g+dt*ag; ledger_add(ledger,method,step,'k2',t+dt,'x_1','Gamma_1','Gamma_1','velocity+stretching'); bx,bg=e.rhs(x1,g1,t+dt,mode)
        return State(x+.5*dt*(ax+bx),g+.5*dt*(ag+bg))
    if order==3:
        x1,g1=x+dt*ax,g+dt*ag; ledger_add(ledger,method,step,'k2',t+dt,'x_1','Gamma_1','Gamma_1','velocity+stretching'); bx,bg=e.rhs(x1,g1,t+dt,mode)
        x2,g2=.75*x+.25*(x1+dt*bx),.75*g+.25*(g1+dt*bg); ledger_add(ledger,method,step,'k3',t+.5*dt,'x_2','Gamma_2','Gamma_2','velocity+stretching'); cx,cg=e.rhs(x2,g2,t+.5*dt,mode)
        return State(x/3+2/3*(x2+dt*cx),g/3+2/3*(g2+dt*cg))
    x1,g1=x+.5*dt*ax,g+.5*dt*ag; ledger_add(ledger,method,step,'k2',t+.5*dt,'x_1','Gamma_1','Gamma_1','velocity+stretching'); bx,bg=e.rhs(x1,g1,t+.5*dt,mode)
    x2,g2=x+.5*dt*bx,g+.5*dt*bg; ledger_add(ledger,method,step,'k3',t+.5*dt,'x_2','Gamma_2','Gamma_2','velocity+stretching'); cx,cg=e.rhs(x2,g2,t+.5*dt,mode)
    x3,g3=x+dt*cx,g+dt*cg; ledger_add(ledger,method,step,'k4',t+dt,'x_3','Gamma_3','Gamma_3','velocity+stretching'); dx,dg=e.rhs(x3,g3,t+dt,mode)
    return State(x+dt*(ax+2*bx+2*cx+dx)/6,g+dt*(ag+2*bg+2*cg+dg)/6)

def reuse(e,s,t,dt,mode,ledger,method,step,exponential=False):
    x,g=s.x,s.gamma; us=[]; js=[]
    ledger_add(ledger,method,step,'k1',t,'x_n','Gamma_n','Gamma_n','velocity+gradient'); us.append(e.vel(x,g,t)); js.append(e.grad(x,g,t))
    x1=x+dt*us[0]; ledger_add(ledger,method,step,'k2',t+dt,'x_1','Gamma_n','Gamma_n','velocity+gradient'); us.append(e.vel(x1,g,t+dt)); js.append(e.grad(x1,g,t+dt))
    x2=.75*x+.25*(x1+dt*us[1]); ledger_add(ledger,method,step,'k3',t+.5*dt,'x_2','Gamma_n','Gamma_n','velocity+gradient'); us.append(e.vel(x2,g,t+.5*dt)); js.append(e.grad(x2,g,t+.5*dt))
    xn=x/3+2/3*(x2+dt*us[2]); ops=[j if mode=='DIRECT' else j.transpose(0,2,1) if mode=='TRANSPOSED' else .5*(j+j.transpose(0,2,1)) for j in js]
    if exponential:
        avg=(ops[0]+ops[1]+4*ops[2])/6; gn=np.vstack([expm(dt*a)@b for a,b in zip(avg,g)])
    else:
        k1=np.einsum('nij,nj->ni',ops[0],g); g1=g+dt*k1; k2=np.einsum('nij,nj->ni',ops[1],g1); g2=.75*g+.25*(g1+dt*k2); k3=np.einsum('nij,nj->ni',ops[2],g2); gn=g/3+2/3*(g2+dt*k3)
    return State(xn,gn)

def advance(method,e,s,t,dt,mode,ledger=None,step=0):
    if method=='coupled_rk2': return coupled(e,s,t,dt,mode,2,ledger,method,step)
    if method=='coupled_rk3': return coupled(e,s,t,dt,mode,3,ledger,method,step)
    if method=='coupled_rk4_reference': return coupled(e,s,t,dt,mode,4,ledger,method,step)
    if method=='fractional_x_gamma': return gstep(method,e,xstep(method,e,s,t,dt,ledger,step,'x'),t,dt,mode,ledger,step,'Gamma')
    if method=='fractional_gamma_x': return xstep(method,e,gstep(method,e,s,t,dt,mode,ledger,step,'Gamma'),t,dt,ledger,step,'x')
    if method=='parallel_lagged': return State(xstep(method,e,s,t,dt,ledger,step,'x').x,gstep(method,e,s,t,dt,mode,ledger,step,'Gamma').gamma)
    if method=='strang_x_gamma_x': return xstep(method,e,gstep(method,e,xstep(method,e,s,t,.5*dt,ledger,step,'x_a'),t,dt,mode,ledger,step,'Gamma'),t+.5*dt,.5*dt,ledger,step,'x_b')
    if method=='strang_gamma_x_gamma': return gstep(method,e,xstep(method,e,gstep(method,e,s,t,.5*dt,mode,ledger,step,'Gamma_a'),t,dt,ledger,step,'x'),t+.5*dt,.5*dt,mode,ledger,step,'Gamma_b')
    if method=='reuse_stage_gradients': return reuse(e,s,t,dt,mode,ledger,method,step)
    if method=='averaged_gradient_exponential': return reuse(e,s,t,dt,mode,ledger,method,step,True)
    raise ValueError(method)

def integrate(method,e,s,horizon,steps,mode,ledger=None):
    out=s.copy(); dt=horizon/steps
    for n in range(steps): out=advance(method,e,out,n*dt,dt,mode,ledger,n)
    return out

def reference_particle(s,sigma,horizon,mode,rtol=2e-12,atol=2e-14,kernel='GAUSSIAN'):
    n=len(s.x); y0=np.r_[s.x.ravel(),s.gamma.ravel()]
    def fun(t,y):
        x=y[:3*n].reshape(n,3); ga=y[3*n:].reshape(n,3)
        return np.r_[velocity(x,ga,sigma,kernel).ravel(),pair_rate(x,ga,sigma,mode,kernel).ravel()]
    sol=solve_ivp(fun,(0,horizon),y0,method='DOP853',rtol=rtol,atol=atol)
    if not sol.success: raise RuntimeError(sol.message)
    return State(sol.y[:3*n,-1].reshape(n,3),sol.y[3*n:,-1].reshape(n,3))

class AffineFlow:
    def __init__(self,name,matrix,deformation=None): self.name=name; self.matrix=matrix; self.deformation=deformation
    def mat(self,t): return np.asarray(self.matrix(t) if callable(self.matrix) else self.matrix)
    def velocity(self,x,t): return x@self.mat(t).T
    def gradient(self,x,t): return np.broadcast_to(self.mat(t),(len(x),3,3)).copy()
    def exact(self,s,horizon,mode):
        if not callable(self.matrix):
            a=self.mat(0); op=a if mode=='DIRECT' else a.T if mode=='TRANSPOSED' else .5*(a+a.T)
            return State(s.x@expm(horizon*a).T,s.gamma@expm(horizon*op).T)
        return reference_flow(self,s,horizon,mode)

class ClosedShear:
    name='nonlinear_closed_shear'
    def __init__(self,horizon=1.): self.T=horizon; self.k=1.3; self.amps=(.32,-.27,.23); self.freq=(1,2,3)
    def coef(self,t):
        c=[]; d=[]
        for a,f in zip(self.amps,self.freq): z=f*np.pi*t/self.T; c.append(a*np.sin(z)**2); d.append(a*f*np.pi/self.T*np.sin(2*z))
        return c,d
    def inverse(self,x,t):
        xx,y,z=x.T; (a,b,c),_=self.coef(t); cc=z-c*np.sin(self.k*xx); bb=y-b*np.sin(self.k*cc); aa=xx-a*np.sin(self.k*bb); return np.column_stack((aa,bb,cc))
    def material(self,l,t):
        a,b,c=l.T; (al,be,ga),_=self.coef(t); x=a+al*np.sin(self.k*b); y=b+be*np.sin(self.k*c); z=c+ga*np.sin(self.k*x); return np.column_stack((x,y,z))
    def velocity(self,x,t):
        a,b,c=self.inverse(x,t).T; (al,be,ga),(ad,bd,gd)=self.coef(t); xx=a+al*np.sin(self.k*b); xd=ad*np.sin(self.k*b); yd=bd*np.sin(self.k*c); zd=gd*np.sin(self.k*xx)+ga*self.k*np.cos(self.k*xx)*xd; return np.column_stack((xd,yd,zd))
    def gradient(self,x,t):
        out=np.empty((len(x),3,3)); eps=1e-30
        for k in range(3): y=np.asarray(x,dtype=complex); y[:,k]+=1j*eps; out[:,:,k]=np.imag(self.velocity(y,t))/eps
        return out
    def deformation(self,l,t):
        a,b,c=l.T; (al,be,ga),_=self.coef(t); k=self.k; d1=np.tile(np.eye(3),(len(l),1,1)); d2=d1.copy(); d3=d1.copy(); d1[:,0,1]=al*k*np.cos(k*b); d2[:,1,2]=be*k*np.cos(k*c); xx=a+al*np.sin(k*b); d3[:,2,0]=ga*k*np.cos(k*xx); return np.einsum('nij,njk,nkl->nil',d3,d2,d1)
    def exact(self,s,horizon,mode):
        labels=self.inverse(s.x,0); x=self.material(labels,horizon)
        if mode=='DIRECT': return State(x,np.einsum('nij,nj->ni',self.deformation(labels,horizon),s.gamma))
        return reference_flow(self,s,horizon,mode)

def reference_flow(flow,s,horizon,mode):
    n=len(s.x); y0=np.r_[s.x.ravel(),s.gamma.ravel()]
    def fun(t,y):
        x=y[:3*n].reshape(n,3); ga=y[3*n:].reshape(n,3); return np.r_[flow.velocity(x,t).ravel(),contract(flow.gradient(x,t),ga,mode).ravel()]
    sol=solve_ivp(fun,(0,horizon),y0,method='DOP853',rtol=2e-13,atol=2e-15)
    return State(sol.y[:3*n,-1].reshape(n,3),sol.y[3*n:,-1].reshape(n,3))

def flows():
    strain=np.diag((.7,-.45,-.25)); planar=np.diag((1.15,-1.15,0)); shear=np.array(((0,.8,0),(0,0,-.35),(0,0,0.))); rot=np.array(((0,-.75,0),(.75,0,0),(0,0,0.))); combo=strain+rot
    rate=.65; diag=np.diag((.8,-.55,-.25)); omega=np.array(((0,-rate,0),(rate,0,0),(0,0,0.)))
    def rm(t): r=np.array(((np.cos(rate*t),-np.sin(rate*t),0),(np.sin(rate*t),np.cos(rate*t),0),(0,0,1))); return omega+r@diag@r.T
    return (AffineFlow('rigid_rotation',rot),AffineFlow('planar_strain',planar),AffineFlow('three_dimensional_strain',strain),AffineFlow('simple_shear',shear),AffineFlow('rotation_plus_strain',combo),AffineFlow('time_rotating_strain',rm),ClosedShear())

def rel(a,b,center=False):
    if center: a=a-a.mean(0); b=b-b.mean(0)
    return float(np.linalg.norm(a-b)/max(np.linalg.norm(b),1e-30))
def errors(out,ref):
    ma=np.linalg.norm(out.gamma,axis=1); mr=np.linalg.norm(ref.gamma,axis=1); cos=np.einsum('ij,ij->i',out.gamma,ref.gamma)/np.maximum(ma*mr,1e-30)
    pe=np.linalg.norm(out.gamma-ref.gamma,axis=1)/np.maximum(mr,1e-30)
    return dict(position_error=rel(out.x,ref.x,True),strength_error=rel(out.gamma,ref.gamma),strength_magnitude_error=rel(ma,mr),strength_angle_mean_deg=float(np.degrees(np.arccos(np.clip(cos,-1,1))).mean()),strength_error_p95=float(np.percentile(pe,95)),strength_error_max=float(pe.max()),excess_growth=float(np.log(max(ma.max(),1e-30)/max(mr.max(),1e-30))))
def invariants(s,sigma):
    total=s.gamma.sum(0); linear=.5*np.cross(s.x,s.gamma).sum(0); angular=np.cross(s.x,np.cross(s.x,s.gamma)).sum(0)/3-(sigma[:,None]**2*s.gamma).sum(0)/3
    d=np.linalg.norm(s.x[:,None,:]-s.x[None,:,:],axis=2); sig=np.sqrt(.5*(sigma[:,None]**2+sigma[None,:]**2)); energy=.5*np.einsum('ij,ij->',gfun(d/sig)/sig,s.gamma@s.gamma.T)
    return total,linear,angular,float(energy)
