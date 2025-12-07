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
        
    def __call__(self,t,x,theta,):
        o = jnp.stack([t,x], axis=-1)
        for layer in self.layers1:
            o = layer(o)
        rho = self.layers2(o)

        k_cos = jnp.arange(1,2*embed_size//2,2,dtype = dtype)
        k_sin = jnp.arange(2,embed_size//2+2,1,dtype = dtype)
        bkinte_cos = ((-1)**k_cos-1)/(k_cos**2-4)


        weight = self.layers3(o)
        base = jnp.concatenate([jnp.cos(k_cos*theta)/k_cos,jnp.sin(k_sin*theta)/k_sin],0)

        g = jnp.sum(base*weight,keepdims=True)
        int_ug = (jnp.sum(weight[:embed_size//2]*bkinte_cos/k_cos,0,keepdims=True)+weight[embed_size//2:embed_size//2+1]*jnp.pi/8)/2 

        return jnp.concatenate((rho,g,int_ug),0)
    

class PINN_GRTE(nnx.Module):
    def __init__(self,seed,layer_sizes1,layer_sizes2):
        rngs = nnx.Rngs(seed)
        self.layers1  = [Res_2L_nnx(rngs,layer_sizes1[:-1],dtype,defult_activation),defult_activation]
        self.layers2  = nnx.Linear(layer_sizes1[-2], layer_sizes1[-1],\
                                    rngs=rngs,kernel_init=nnx.initializers.xavier_normal(),bias_init=nnx.initializers.zeros,param_dtype=dtype,)
        self.layers3  = Res_2L_nnx(rngs,layer_sizes2,dtype,defult_activation)
        
        def __call__(self,t,x,theta,):
            o = jnp.stack([t,x], axis=-1)

            for layer in self.layers1:
                o = layer(o)
            TrT = jnp.log(1+jnp.exp(self.layers2(o)/2))/2+1e-2

            k_cos = jnp.arange(1,2*embed_size//2,2,dtype = dtype)
            k_sin = jnp.arange(2,embed_size//2+2,1,dtype = dtype)
            bkinte_cos = ((-1)**k_cos-1)/(k_cos**2-4)


            weight = self.layers3(o)
            base = jnp.concatenate([jnp.cos(k_cos*theta)/k_cos,jnp.sin(k_sin*theta)/k_sin],0)


            g = jnp.sum(base*weight,keepdims=True)
            int_ug = (jnp.sum(weight[:embed_size//2]*bkinte_cos/k_cos,0,keepdims=True)+weight[embed_size//2:embed_size//2+1]*jnp.pi/8)/2 


            return jnp.concatenate((TrT,g,int_ug),0)