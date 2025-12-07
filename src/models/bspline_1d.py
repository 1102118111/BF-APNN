import jax.numpy as jnp
from .resblock import Res_2L_nnx
from flax import nnx
from utils.utils import *
from scipy.interpolate import BSpline
import numpy as np

class BSplineBasis:
    def __init__(self, n, k, num_samples=5000, num_interp_points=10000):
        self.n = n
        self.k = k
        self.num_interp_points = num_interp_points
        
        num_knots = n + k
        self.t = jnp.concatenate([
            jnp.zeros(k-1,),
            jnp.linspace(0., jnp.pi, num_knots - 2*(k-1)),
            jnp.full(k-1, jnp.pi,)
        ])

        t_np = np.array(self.t)
        self.basis_elements = [
            BSpline.basis_element(t_np[i:i+k+1], extrapolate=False)
            for i in range(n)
        ]


        
        self._precompute_integrals(num_samples)
        self._precompute_basis_values(num_interp_points)
        self._compiled_eval = jax.jit(self._evaluate_interp)

    def _precompute_integrals(self, num_samples):
        x_vals = jnp.linspace(0, jnp.pi, num_samples)
        x_np = np.array(x_vals)
        
        basis_matrix = np.zeros((self.n, num_samples))
        for i in range(self.n):
            basis_matrix[i] = np.nan_to_num(self.basis_elements[i](x_np))
        
        sin_x = np.sin(x_np)
        cos_sin_x = np.cos(x_np) * sin_x
        
        integrals = np.trapz(basis_matrix * sin_x, x_np, axis=1) / 2
        xf_integrals = np.trapz(basis_matrix * cos_sin_x, x_np, axis=1) / 2
        
        self.integrals = jnp.array(integrals).reshape(1, -1)
        self.xf_integrals = jnp.array(xf_integrals).reshape(1, -1)

    def _precompute_basis_values(self, num_points):
        self.x_grid = jnp.linspace(0, jnp.pi, num_points)
        x_np = np.array(self.x_grid)
        basis_values = np.zeros((self.n, num_points))
        for i in range(self.n):
            basis_values[i] = np.nan_to_num(self.basis_elements[i](x_np))
        self.basis_values = jnp.array(basis_values.T)  

    def _evaluate_interp(self, x):
        basis_tensor = jnp.array([jnp.interp(x, self.x_grid, self.basis_values[:, i]) for i in range(self.n)])
        integrals = self.integrals.reshape(self.n)
        xf_integrals = self.xf_integrals.reshape(self.n)
        result = basis_tensor - integrals
        return result, xf_integrals

    def __call__(self, x):
        x = jnp.asarray(x)
        if x.ndim != 0:
            raise ValueError("This implementation only accepts scalar inputs. Use jax.vmap for vectorized evaluation.")
        return self._compiled_eval(x)
base_func = jax.jit(BSplineBasis(embed_size, 3))


    
class PINN_LRTE(nnx.Module):
    def __init__(self,seed,layer_sizes1,layer_sizes2):
        rngs = nnx.Rngs(seed)
        self.layers1  = [Res_2L_nnx(rngs,layer_sizes1[:-1]),defult_activation]
        self.layers2  = nnx.Linear(layer_sizes1[-2], layer_sizes1[-1],\
                                    rngs=rngs,kernel_init=nnx.initializers.xavier_normal(),bias_init=nnx.initializers.zeros,param_dtype=dtype,)
        self.layers3  = Res_2L_nnx(rngs,layer_sizes2,defult_activation)

    def __call__(self,t,x,theta,):
        o = jnp.stack([t,x], axis=-1)
        for layer in self.layers1:
            o = layer(o)
        rho = self.layers2(o)

        weight = self.layers3(o)
        base,bk = base_func(theta)
        g = jnp.sum(base*weight,keepdims=True)
        int_ug = jnp.sum(bk*weight,keepdims=True)
        return jnp.concatenate((rho,g,int_ug),0)
    
class PINN_GRTE(nnx.Module):
    def __init__(self,seed,layer_sizes1,layer_sizes2):
        rngs = nnx.Rngs(seed)
        self.layers1  = [Res_2L_nnx(rngs,layer_sizes1[:-1]),defult_activation]
        self.layers2  = nnx.Linear(layer_sizes1[-2], layer_sizes1[-1],\
                                    rngs=rngs,kernel_init=nnx.initializers.xavier_normal(),bias_init=nnx.initializers.zeros,param_dtype=dtype,)
        self.layers3  = Res_2L_nnx(rngs,layer_sizes2,defult_activation)

    def __call__(self,t,x,theta,):
        o = jnp.stack([t,x], axis=-1)

        for layer in self.layers1:
            o = layer(o)
        TrT = jnp.log(1+jnp.exp(self.layers2(o)/2))/2+1e-2

        weight = self.layers3(o)
        base,bk = base_func(theta)
        g = jnp.sum(base*weight,keepdims=True)
        int_ug = jnp.sum(bk*weight,keepdims=True)
        return jnp.concatenate((TrT,g,int_ug),0)