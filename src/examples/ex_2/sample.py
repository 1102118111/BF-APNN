import jax
import jax.numpy as jnp
import jax.random as jr
from utils.utils import *


def get_data(seed,Time,lb,ub,Nr,Nb,Ni,Ne,Model,old_T = 0.,epoch = 0):
    Nr ,Nb , Ni , Ne  = int(Nr), int(Nb), int(Ni), int(Ne)
    lb, ub = jnp.array(lb), jnp.array(ub)
    def lhs(dim, N, keys):
        def lhs1d(N, key):
            lb = jnp.linspace(0,1,N+1)[:-1]
            x = lb+jr.uniform(key,shape=(N,))/N
            return jr.permutation(key,x)
        return jnp.stack([lhs1d(N,key) for key in jr.split(keys,dim)],1)

    
    Kr = jr.split(jr.PRNGKey(seed+epoch+1),ipdim)
    Kb = jr.split(jr.PRNGKey(seed+epoch+2),ipdim)
    Ki = jr.split(jr.PRNGKey(seed+epoch+3),ipdim)
    Ke = jr.split(jr.PRNGKey(seed+epoch+4),ipdim)

    Xr = lhs(ipdim,Nr,Kr[0])*(ub-lb)+lb

    Xb = lhs(ipdim,Nb,Kb[0])*(ub-lb)+lb
    Xb = Xb.at[:,1].set(jnp.zeros(Nb))
    
    Xi = lhs(ipdim,Ni,Ki[0])*(ub-lb)+lb
    Xi = Xi.at[:,0].set(jnp.zeros(Ni))
    Yi = 0.75+0.25*jnp.sin(jnp.pi*Xi[:,1:2])
    Xi = jnp.concatenate([Xi,Yi],-1)
    
    Xr = jr.permutation(Kr[1],Xr)
    Xb = jr.permutation(Kb[1],Xb)
    Xi = jr.permutation(Ki[1],Xi)

    if old_T==0.:
        Ne = 64
        Xe = jnp.zeros((Ne,1))
    else :
        Xe = lhs(ipdim,Ne,Ke[0])*(ub.at[0].set(old_T)-lb)+lb
        Xe = jr.permutation(jr.PRNGKey(seed+8+epoch),Xe)
        oe = jnp.split(Xe,Xe.shape[-1],-1)
        print(Xe.shape)
        Ye= jax.vmap(Model)(*list(map(jnp.squeeze, oe)))[:,0:1]
        Xe = jnp.concatenate([Xe,Ye],-1)

    Xr = jr.permutation(Kr[1],Xr)
    Xb = jr.permutation(Kb[1],Xb)
    Xi = jr.permutation(Ki[1],Xi)
    Xe = jr.permutation(Ke[1],Xe)
    return Xr,Xb,Xi,Xe