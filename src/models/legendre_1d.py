import jax.numpy as jnp
from .resblock import Res_2L_nnx
from flax import nnx
from utils.utils import *
from scipy.special import legendre


k = jnp.arange(embed_size+1)

coff = jnp.zeros((embed_size+1, embed_size+1))

for i in range(embed_size+1):

    leg_coeffs = legendre(i).coef[::-1]

    coff = coff.at[i, :len(leg_coeffs)].set(leg_coeffs)

coff = coff.at[0, 0].set(0)
coff = coff.T

class PINN_LRTE(nnx.Module):
    def __init__(self,seed,layer_sizes1,layer_sizes2):
        rngs = nnx.Rngs(seed)
        self.layers1  = [Res_2L_nnx(rngs,layer_sizes1[:-1],defult_activation),defult_activation]
        self.layers2  = nnx.Linear(layer_sizes1[-2], layer_sizes1[-1],\
                                    rngs=rngs,kernel_init=nnx.initializers.xavier_normal(),bias_init=nnx.initializers.zeros,param_dtype=dtype,)
        self.layers3  = Res_2L_nnx(rngs,layer_sizes2,defult_activation)
        
    def __call__(self,t,x,u,):
        o = jnp.stack([t,x], axis=-1)
        for layer in self.layers1:
            o = layer(o)
        rho = self.layers2(o)


        weight = self.layers3(o)

        base = jnp.einsum("i,ij->j",jnp.power(u,k),coff)


        g = jnp.sum(base*weight,keepdims=True)
        int_ug = weight[1:2]*1/3



        return jnp.concatenate((rho,g,int_ug),0)
    

class PINN_GRTE(nnx.Module):
    def __init__(self,seed,layer_sizes1,layer_sizes2):
        rngs = nnx.Rngs(seed)
        self.layers1  = [Res_2L_nnx(rngs,layer_sizes1[:-1],defult_activation),defult_activation]
        self.layers2  = nnx.Linear(layer_sizes1[-2], layer_sizes1[-1],\
                                    rngs=rngs,kernel_init=nnx.initializers.xavier_normal(),bias_init=nnx.initializers.zeros,param_dtype=dtype,)
        self.layers3  = Res_2L_nnx(rngs,layer_sizes2,defult_activation)
        
    def __call__(self,t,x,u,):
        o = jnp.stack([t,x], axis=-1)
        for layer in self.layers1:
            o = layer(o)
        TrT = jnp.log(1+jnp.exp(self.layers2(o)/2))/2+1e-2


        weight = self.layers3(o)

        base = jnp.einsum("i,ij->j",jnp.power(u,k),coff)


        g = jnp.sum(base*weight,keepdims=True)
        int_ug = weight[1:2]*1/3

        return jnp.concatenate((TrT,g,int_ug),0)