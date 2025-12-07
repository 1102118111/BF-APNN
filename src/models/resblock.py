from flax import nnx
import jax
import jax.numpy as jnp
import jax.random as jr
import pytreeclass as pytc
from utils.utils import defult_activation,dtype

class Res_2L_nnx(nnx.Module):
    def __init__(self,  rngs,layer_sizes,activation=defult_activation,kernel_init = nnx.initializers.xavier_normal(),bias_init =nnx.initializers.zeros,  ):
        self.activation = activation
        self.layers1 = [nnx.Linear(layer_sizes[0], layer_sizes[1],\
                                    rngs=rngs,kernel_init=kernel_init,bias_init=bias_init,param_dtype=dtype),
                        activation]
                        
        self.layers2 = []

        for i in range(1,len(layer_sizes) - 2):
            self.layers2.append(nnx.Linear(layer_sizes[i], layer_sizes[i + 1],\
                                            rngs=rngs,kernel_init=kernel_init,bias_init=bias_init,param_dtype=dtype))
            self.layers2.append(activation)
        self.layers3 = nnx.Linear(layer_sizes[-2], layer_sizes[-1],\
                                    rngs=rngs,kernel_init=kernel_init,bias_init=bias_init,param_dtype=dtype)

    def __call__(self, inputs):  
        inputs = self.layers1[0](inputs)
        inputs = self.layers1[1](inputs)
        residual = inputs  
        for i, layer in enumerate(self.layers2):
            inputs = layer(inputs)
            if (i+1) % 4 == 0: 
                inputs += residual
                residual = inputs  
        inputs = self.layers3(inputs)
        return inputs 
    
class Linear(pytc.TreeClass):
    def __init__(self,key, in_dim, out_dim):

        scale = jnp.sqrt(1 / (in_dim))
        self.w = jr.uniform(key, shape = (in_dim, out_dim), minval = -scale, maxval = scale) 
        self.b = jr.uniform(key, shape = (out_dim, ), minval = -scale, maxval = scale)

    def __call__(self, x):
        return x @ self.w + self.b


class LinearAct(pytc.TreeClass):
    def __init__(self, m, act):
        self.linear_i = m
        self.activate = act
    def __call__(self, x):
        return self.activate(self.linear_i(x))


class Activation(pytc.TreeClass):
    def __call__(self, x):
        return jax.nn.tanh(x)
activation = Activation()


class Seq(pytc.TreeClass):
    def __init__(self, *models):
        self.models = models
    def __call__(self, x):
        for m in self.models:
            x = m(x)
        return x

class Res(pytc.TreeClass):
    def __init__(self, *models):
        self.models = models#Seq(*models)
    def __call__(self, x):
        h = x.copy()
        for m in self.models:
            h = m(h)
        return x + h
class Res2L(pytc.TreeClass):
    def __init__(self, dim, key,act):
        k1,k2 = jr.split(key,2)
        self.linear_1 = Linear(k1 , dim,dim,)
        self.linear_2 = Linear(k2 , dim,dim,)
        self.activate = act
    def __call__(self, x):
        h = self.activate(self.linear_1(x))
        return self.activate(x + self.linear_2(h))