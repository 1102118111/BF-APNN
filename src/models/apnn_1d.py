import jax.numpy as jnp
from .resblock import Res_2L_nnx
from flax import nnx
from utils.utils import *


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

        g = self.layers3(jnp.stack((t,x,u),0))
        return jnp.concatenate((rho,g),0)