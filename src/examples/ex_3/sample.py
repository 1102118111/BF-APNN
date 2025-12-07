import jax
import jax.numpy as jnp
import numpy as np
import jax.random as jr
from utils.utils import *

a = 0.01372
c= 29.97924580
NX = 3240
qp = 1297
constant = 30
T0 = 1e-2
def get_data(seed,Time,lb,ub,Nr,Nb,Ni,Ne,Model,old_T = 0.,epoch = 0):
    Nr ,Nb , Ni , Ne  = int(Nr), int(Nb), int(Ni), int(Ne)
    lb, ub = jnp.array(lb), jnp.array(ub)
    # 实现 LHS 算法
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
    Xr = Xr.at[19*Nr//20:1].set(Xr[19*Nr//20:1]*0.135/0.2)


#     Xb = lhs(ipdim,4*Nb,Kb[0])*(ub.at[0].set(Time+1e-3)-lb)+lb
    Xb = lhs(ipdim,Nb,Kb[0])*(ub-lb)+lb
    if model_type in [0 ,2]:
        idx = jnp.cos(Xb[:,2:3])>0
    elif model_type in [1 ,3]:
        idx = Xb[:,2:3]>0
    Xb = Xb.at[:,1:2].set(jnp.where(idx,lb[1],ub[1]))
    bnd_I = jnp.where(idx,0.5*a*c,0.5*a*c*T0**4)
    bnd_T = jnp.where(idx,1.,T0)
    Xb = jnp.concatenate((Xb,bnd_I,bnd_T),1)

    data_ini= np.loadtxt(r'../data/marshak/examples/sigma{}/{:.0e}_{}_0.10'.format(constant,T0,NX))[:qp]
    t_ini = 0.*np.ones((data_ini.shape[0],1))
    x_ini = data_ini[:,0:1]
    Tr_ini = (data_ini[:,1:2]/a/c)**0.25
    T_ini = data_ini[:,2:]
    if model_type in [0 ,2]:
        tehta_ini = jr.uniform(Ki[2],t_ini.shape)*jnp.pi
    elif model_type in [1 ,3]:
        tehta_ini = jr.uniform(Ki[2],t_ini.shape)*2-1.
    ini_txtheta = jnp.concatenate([t_ini,x_ini,tehta_ini],axis=-1)
    ini_Y = jnp.concatenate([Tr_ini,T_ini],axis=-1)
    Xi = jnp.concatenate([ini_txtheta,ini_Y],-1)

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