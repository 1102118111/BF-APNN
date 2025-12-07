import jax
import jax.numpy as jnp
import jax.random as jr
from utils.utils import *
from utils.common import gradtt

a1,a2,b1,b2 = 0.8,0.8,0.1,0.1
sigma = 1.
def Rho(x,y):
    return (a1+b1*jnp.sin(x))*(a2+b2*jnp.sin(y))**4
def Te(x,y):
    return (a1+b1*jnp.sin(x))*(a2+b2*jnp.sin(y))
if model_type in [1,2]:
    def G(x,y,theta,phi):
        v1 = jnp.sin(theta)*jnp.cos(phi)
        v2 = jnp.sin(theta)*jnp.sin(phi)
        rho_x = gradtt(Rho,0)(x,y)
        rho_y = gradtt(Rho,1)(x,y)
        return (-v1*rho_x-v2*rho_y)/sigma
elif model_type in [0]:
    def G(x,y,v1,v2):
        rho_x = gradtt(Rho,0)(x,y)
        rho_y = gradtt(Rho,1)(x,y)
        return (-v1*rho_x-v2*rho_y)/sigma


if model_type in [1,2]:
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
        
        Xb = lhs(3,Nb*2,Kb[0])
        Xb = Xb*(ub-lb)+lb
        Xb = Xb.at[:Nb,1].set(Lb[1])
        Xb = Xb.at[Nb:,2].set(Lb[2])
        Xb = jnp.concatenate((Xb,sphere(Nb*2,Kb[1])),1)
        
        Xi = lhs(3,Ni,Ki[0])
        Xi = Xi*(ub-lb)+lb    
        Xi = Xi.at[:,0].set(lb[0])
        Xi = jnp.concatenate((Xi,sphere(Ni,Ki[1])),1)
        Xi = jnp.concatenate((Xi,Rho(Xi[:,1:2],Xi[:,2:3]),Te(Xi[:,1:2],Xi[:,2:3]),G(Xi[:,1:2],Xi[:,2:3],Xi[:,3:4],Xi[:,4:5])),1)
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

elif model_type in [0]:
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
            v1 = jnp.sin(theta)*jnp.cos(phi)
            v2 = jnp.sin(theta)*jnp.sin(phi)
            return jnp.concatenate((v1,v2),1)
    
        
        Kr = jr.split(jr.PRNGKey(seed+1+epoch),ipdim)
        Kb = jr.split(jr.PRNGKey(seed+2+epoch),ipdim)
        Ki = jr.split(jr.PRNGKey(seed+3+epoch),ipdim)
        Ke = jr.split(jr.PRNGKey(seed+4+epoch),ipdim)
    
        
        Xr = lhs(3,Nr,Kr[0])
        Xr = Xr*(ub-lb)+lb
        Xr = Xr.at[:,0].set(Xr[:,0]/((ub-lb)[0])*Time) 
        Xr = jnp.concatenate((Xr,sphere(Nr,Kr[1])),1)
        
        Xb = lhs(3,Nb*2,Kb[0])
        Xb = Xb*(ub-lb)+lb
        Xb = Xb.at[:Nb,1].set(Lb[1])
        Xb = Xb.at[Nb:,2].set(Lb[2])
        Xb = jnp.concatenate((Xb,sphere(Nb*2,Kb[1])),1)
        
        Xi = lhs(3,Ni,Ki[0])
        Xi = Xi*(ub-lb)+lb    
        Xi = Xi.at[:,0].set(lb[0])
        Xi = jnp.concatenate((Xi,sphere(Ni,Ki[1])),1)
        Xi = jnp.concatenate((Xi,Rho(Xi[:,1:2],Xi[:,2:3]),Te(Xi[:,1:2],Xi[:,2:3]),G(Xi[:,1:2],Xi[:,2:3],Xi[:,3:4],Xi[:,4:5])),1)
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