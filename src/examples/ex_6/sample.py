import jax
import jax.numpy as jnp
import jax.random as jr
from utils.utils import *

zeta = 0.3
def Q(x,y):
    ini = jnp.exp(-(x**2+y**2)/2/zeta**2)/2/jnp.pi/zeta**2/4/jnp.pi
    return jnp.maximum(ini,jnp.zeros_like(ini)+1e-8)

def get_date(seed,Time,lb,ub,Nr,Nb,Ni,Ne,Model,old_T=0,epoch = 0):
    lb, ub = jnp.array(lb), jnp.array(ub)
    lb,ub = lb[:3],ub[:3]

    def lhs(dim, N, keys):
        def lhs1d(N, key):
            lb = jnp.linspace(0,1,N+1)[:-1]
            x = lb+jr.uniform(key,shape=(N,))/N
            return jr.permutation(key,x)
        return jnp.stack([lhs1d(N,key) for key in jr.split(keys,dim)],1)
    def sphere(N,key):
        r,t = jnp.split(lhs(2, N, key),2,1)
        r = 2*r-1
        phi = jnp.pi*2*t
        theta = jnp.arccos(r)
        return jnp.concatenate((theta,phi),1)

    
    Kr = jr.split(jr.PRNGKey(seed+1+epoch),ipdim)
    Kb = jr.split(jr.PRNGKey(seed+2+epoch),ipdim)
    Ki = jr.split(jr.PRNGKey(seed+3+epoch),ipdim)
    Ke = jr.split(jr.PRNGKey(seed+4+epoch),ipdim)
    
    Xr = lhs(3,Nr,Kr[0])
    Xr = Xr*(ub-lb)+lb
    Xr = Xr.at[:,0].set(Xr[:,0]/((ub-lb)[0])*Time) 
    Xr = jnp.concatenate((Xr,sphere(Nr,Kr[1])),1)

    
    Xb = lhs(3,Nb*4,Kb[0])
    Xb = Xb*(ub-lb)+lb
    Sb = sphere(Nb*4,Kb[1])
    idx1 = (jnp.pi/2<Sb[:2*Nb,-1])*(Sb[:2*Nb,-1]<jnp.pi*3/2)
    idx2 = (0<Sb[2*Nb:,-1])*(Sb[2*Nb:,-1]<jnp.pi)
    Xb = Xb.at[:2*Nb,1].set(jnp.where(idx1,Ub[1],Lb[1]))
    Xb = Xb.at[2*Nb:4*Nb,2].set(jnp.where(idx2,Lb[2],Ub[2]))
    Xb = jnp.concatenate((Xb,Sb),1)
    Xb = jr.permutation(jr.PRNGKey(seed+6+epoch),Xb)
    

    Xi = lhs(3,Ni,Ki[0])
    Xi = Xi*(ub-lb)+lb
    Xi = Xi.at[:,0].set(lb[0])
    Xi = Xi.at[:Ni//2,1:3].set(Xi[:Ni//2,1:3]/3)
    Xi = jnp.concatenate((Xi,sphere(Ni,Ki[1])),1)
    Yi = Q(Xi[:,1:2],Xi[:,2:3])
    Xi = jnp.concatenate([Xi,Yi],-1)


    if old_T==0:
        Ne = (sharding.shape[0])**2
        Xe = jnp.zeros((Ne,5))
    else :
        Xe = lhs(3,Ne,Ke[0])*(ub-lb)+lb
        Xe = Xe.at[:,0].set(Xe[:,0]/((ub-lb)[0])*old_T)
        Xe = jnp.concatenate((Xe,sphere(Ne,Ke[1])),1)
        te,xe,ye,ue,ve = jnp.split(Xe,ipdim,1)
        Xe = jnp.concatenate((Xe,Model(te,xe,ye,ue,ve)[:,0:2]),-1)
    return Xr,Xb,Xi,Xe