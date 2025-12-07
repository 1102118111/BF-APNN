import jax
import jax.numpy as jnp
import jax.random as jr
from utils.utils import *


a = 1.5
left_boundry = -a;right_boundry = a
k = 2


def heaviside(x):
    return jnp.where(x >= 0, 1, 0.)
def mask(x,y):
    mask0 = (x+y>0)*(y-x>0)*(y<right_boundry)
    mask1 = (x>left_boundry)*(y>left_boundry)*(x+y<=0)
    return mask0,mask1
def I0(x,y):
    return (1-jnp.tanh(k*(x+y)))/2
def sigma(x,y):
    return jnp.where(mask(x,y)[0],10,1)

if model_type in [1,2]:
    def get_date(seed,Time,lb,ub,Nr,Nb,Ni,Ne,Model,old_T=0,epoch = 0):
        lb, ub = jnp.array(lb), jnp.array(ub)
        lb,ub = lb[:3],ub[:3]
        # 实现 LHS 算法
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
        Yb = I0(Xb[:,1:2],Xb[:,2:3])
        Xb = jnp.concatenate([Xb,Yb],-1)
        
    
        Xi = lhs(3,Ni,Ki[0])
        Xi = Xi*(ub-lb)+lb
        Xi = Xi.at[:,0].set(lb[0])
        Xi = jnp.concatenate((Xi,sphere(Ni,Ki[1])),1)
        Yi = I0(Xi[:,1:2],Xi[:,2:3])
        Xi = jnp.concatenate([Xi,Yi],-1)
      
        Xr = jnp.concatenate((Xr,sigma(Xr[:,1:2],Xr[:,2:3]),),1)
    
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
        lb, ub = lb[:3], ub[:3]
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
        
        
        Xd1 = lhs(3,Nb,Kb[0])
        Xd1 = Xd1*(ub-lb)+lb
        Xd1 = Xd1.at[:,1].set(left_boundry)
    
        Xd2 = lhs(3,Nb,Kb[1])
        Xd2 = Xd2*(ub-lb)+lb
        Xd2 = Xd2.at[:,2].set(left_boundry)
    
        Xd3 = lhs(3,Nb,Kb[2])
        Xd3 = Xd3*(ub-lb)+lb
        Xd3 = Xd3.at[:,1].set(right_boundry)
    
        Xd4 = lhs(3,Nb,Kb[3])
        Xd4 = Xd4*(ub-lb)+lb
        Xd4 = Xd4.at[:,2].set(right_boundry)
    
        Xd = jnp.concatenate((Xd1,Xd2,Xd3,Xd4),0)

        Xd = jnp.concatenate((Xd,sphere(Nb*4,Kb[4])),1)
    
        Xd = Xd.at[0*Nb:1*Nb, 3].set(jnp.abs(Xd[0*Nb:1*Nb, 3]))
        Xd = Xd.at[1*Nb:2*Nb, 3].set(jnp.abs(Xd[1*Nb:2*Nb, 4]))
        Xd = Xd.at[2*Nb:3*Nb, 4].set(jnp.abs(Xd[2*Nb:3*Nb, 3]) * (-1))
        Xd = Xd.at[3*Nb:4*Nb, 4].set(jnp.abs(Xd[3*Nb:4*Nb, 4]) * (-1))
    
    

        Yd = I0(Xd[:,1:2],Xd[:,2:3])

        Xb = jnp.concatenate([Xd,Yd],-1)
        
    
        Xi = lhs(3,Ni,Ki[0])
        Xi = Xi*(ub-lb)+lb
        Xi = Xi.at[:,0].set(lb[0])
        Xi = jnp.concatenate((Xi,sphere(Ni,Ki[1])),1)
        Yi = I0(Xi[:,1:2],Xi[:,2:3])
        Xi = jnp.concatenate([Xi,Yi],-1)
      
        Xr = jnp.concatenate((Xr,sigma(Xr[:,1:2],Xr[:,2:3]),),1)
    
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