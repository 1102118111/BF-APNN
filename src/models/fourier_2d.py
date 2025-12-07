import jax.numpy as jnp
import jax.scipy as jsp
import jax.random as jr
import pytreeclass as pytc
from .resblock import *
from utils.utils import *
from functools import partial
fourier_base_num = l
def affine(x,Lb,Ub):
    return (x-Lb)/(Ub-Lb) * 2 - 1
def outactivate(x):
    return x
@partial(jax.jit, static_argnames=['flag', 'i_max', 'j_max', 'num_points'])
def fourier_basis_integral_on_sphere(flag = 0,omega=1, i_max=fourier_base_num, j_max=fourier_base_num, num_points=5000):

    theta_vals = jnp.linspace(0, jnp.pi, num_points)  
    phi_vals = jnp.linspace(0, 2 * jnp.pi, num_points)  


    theta, phi = jnp.meshgrid(theta_vals, phi_vals)
    

    i_vals = jnp.arange(1, i_max + 1) 
    j_vals = jnp.arange(1, j_max + 1) 
    

    cos_theta = jnp.cos(i_vals * omega * theta[...,None])/i_vals  
    sin_theta = jnp.sin(i_vals * omega * theta[...,None])/i_vals  
    cos_phi = jnp.cos(j_vals * omega * phi[...,None])/j_vals      
    sin_phi = jnp.sin(j_vals * omega * phi[...,None])/j_vals      



    cosx_cosy = jnp.einsum('ijk,ijl->ijkl', cos_theta, cos_phi).reshape(num_points,num_points,i_max*j_max)
    sinx_cosy = jnp.einsum('ijk,ijl->ijkl', sin_theta, cos_phi).reshape(num_points,num_points,i_max*j_max)
    cosx_siny = jnp.einsum('ijk,ijl->ijkl', cos_theta, sin_phi).reshape(num_points,num_points,i_max*j_max)
    sinx_siny = jnp.einsum('ijk,ijl->ijkl', sin_theta, sin_phi).reshape(num_points,num_points,i_max*j_max)
    

    basis_functions = jnp.concatenate([cosx_cosy, sinx_cosy, cosx_siny, sinx_siny], axis=-1)



    if flag == 0:
        weights = jnp.sin(theta)**2*jnp.sin(phi)
    elif flag == 1:
        weights = jnp.sin(theta)**2*jnp.cos(phi)
    

    integral = jnp.trapezoid(jnp.trapezoid(basis_functions * weights[..., None], theta_vals, axis=0), phi_vals, axis=0)
    
    return (integral/4/jnp.pi)[None,...]


x_int = fourier_basis_integral_on_sphere(1)
y_int = fourier_basis_integral_on_sphere(0)


i_vals = jnp.arange(1, fourier_base_num + 1)
j_vals = jnp.arange(1, fourier_base_num + 1)

class Fourier_Net(pytc.TreeClass):
    def __init__(self,layers1_list,layers2_list,activation=activation,seed=seed):
        LA = layers1_list
        _,KA,KB = jr.split(jr.PRNGKey(seed),3)
        _,*KA = jr.split(KA,len(LA)+1)
        
        self.layers1 = Seq(*[Res2L(i,key,Activation()) if i==o else LinearAct(Linear(key , i, o),activation)
                          for key , i, o,  in zip(KA[:-1] , LA[:-2], LA[1:-1], )])
        self.layers3 = Linear(KA[-1] , LA[-2], LA[-1], )
        LB = layers2_list
        _,*KB = jr.split(KB,len(LB)+1)
        self.layers2 = Seq(*[Res2L(i,key,Activation()) if i==o else LinearAct(Linear(key , i, o ),activation)
                          for key , i, o,  in zip(KB[:-1] , LB[:-2], LB[1:-1] )] + [Linear(KB[-1] , LB[-2], LB[-1])])
    def summary(self):

        print(pytc.tree_diagram(self))
    
    def __call__(self, *o):
        o = jnp.stack(o, axis=-1).reshape(-1,5)

        t,x,y,theta,phi = jnp.split(o,5,axis=-1)
        

        txy = jnp.concatenate([t,x,y],-1)
        txy = affine(txy,Lb[:3],Ub[:3])

        Y = self.layers1(txy)
        rho = outactivate(self.layers3(Y))
        weight = self.layers2(Y)


        omega = 1
        cosx = jnp.cos(i_vals * omega * theta)/i_vals
        sinx = jnp.sin(i_vals * omega * theta)/i_vals
        cosy = jnp.cos(j_vals * omega * phi)/j_vals
        siny = jnp.sin(j_vals * omega * phi)/j_vals

        cosx_cosy = jnp.einsum('ni,nj->nij', cosx, cosy).reshape(theta.shape[0], -1)
        sinx_cosy = jnp.einsum('ni,nj->nij', sinx, cosy).reshape(theta.shape[0], -1)
        cosx_siny = jnp.einsum('ni,nj->nij', cosx, siny).reshape(theta.shape[0], -1)
        sinx_siny = jnp.einsum('ni,nj->nij', sinx, siny).reshape(theta.shape[0], -1)
        base = jnp.concatenate([cosx_cosy, sinx_cosy,  cosx_siny,sinx_siny], axis=-1)
        g = jnp.sum(weight*base,-1,keepdims=True)

        int_v1g = jnp.dot(weight,x_int.T)
        int_v2g = jnp.dot(weight,y_int.T)
        return jnp.concatenate((rho,g,int_v1g,int_v2g),-1)