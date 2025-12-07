import jax
import jax.numpy as jnp
import numpy as np
from utils.utils import *
from functools import partial
from flax import nnx
from jax import vmap


lb = Lb[:-1]
ub = Ub[:-1]
c = 1.
sigma0 = np.sqrt(10)
eps = 1e-2

if model_type in [0 ,2]:
    @partial(jax.jit, static_argnums=(0,))
    def loss_res(model, *args,**kwargs):
        t,x,theta,sigmar = args
        fn_rho = lambda i,j,k: model(i,j,k)[0]
        fn_g = lambda i,j,k: model(i,j,k)[1]
        fn_intvg = lambda i,j,k: model(i,j,k)[2]
    
        fn_rho_t = jax.grad(fn_rho, argnums=0)
        fn_rho_x = jax.grad(fn_rho, argnums=1)
    
    
        fn_g_t = jax.grad(fn_g, argnums=0)
        fn_g_x = jax.grad(fn_g, argnums=1)
    
        fn_intvg_x = jax.grad(fn_intvg, argnums=1)
    
        rho_t = fn_rho_t(t,x,theta)
        rho_x = fn_rho_x(t,x,theta)
        g_t = fn_g_t(t,x,theta)
        g_x = fn_g_x(t,x,theta)
        Avg_g = fn_intvg_x(t,x,theta)
        rho,g,_ = model(t,x,theta)
        mu = jnp.cos(theta)
    
        res1 = rho_t/c+Avg_g/sigma0
        res2 = eps**2*g_t + sigma0*mu*rho_x + eps * mu * g_x + sigmar * g - eps * Avg_g
    
        return res1,res2

elif model_type in [1]:
    @partial(jax.jit, static_argnums=(0,))
    def loss_res(model, *args,**kwargs):
        t,x,mu,sigmar = args
        fn_rho = lambda i,j,k: model(i,j,k)[0]
        fn_g = lambda i,j,k: model(i,j,k)[1]
        fn_intvg = lambda i,j,k: model(i,j,k)[2]
    
        fn_rho_t = jax.grad(fn_rho, argnums=0)
        fn_rho_x = jax.grad(fn_rho, argnums=1)
    
        fn_g_t = jax.grad(fn_g, argnums=0)
        fn_g_x = jax.grad(fn_g, argnums=1)
    
        fn_intvg_x = jax.grad(fn_intvg, argnums=1)
    
        rho_t = fn_rho_t(t,x,mu)
        rho_x = fn_rho_x(t,x,mu)
        g_t = fn_g_t(t,x,mu)
        g_x = fn_g_x(t,x,mu)
        Avg_g = fn_intvg_x(t,x,mu)
        rho,g,_ = model(t,x,mu)
    
        res1 = rho_t/c+Avg_g/sigma0
        res2 = eps**2*g_t + sigma0*mu*rho_x + eps * mu * g_x + sigmar * g - eps * Avg_g
    
        return res1,res2


elif model_type in [3,4]:
    NS = 10
    mu_prime, w = np.polynomial.legendre.leggauss(NS)
    w = jnp.array(w)
    mu_prime = jnp.array(mu_prime)
    @partial(jax.jit, static_argnums=(0,))
    def loss_res(model, *args,**kwargs):
        t,x,mu,sigmar = args
        fn_rho = lambda i,j,k: model(i,j,k)[0]
        fn_g1 = lambda i,j,k: model(i,j,k)[1]

        def vgx(i,j,k):
            return k*jax.grad(fn_g1, argnums=1)(i,j,k)
            
        def fn_g(i,j,k):
            return fn_g1(i,j,k) - int_g(i,j)
        
        def complus_integral(f,i,j):
            Iw = w/2
            return jnp.dot((jax.vmap(lambda k: f(i,j,k),in_axes=0)(mu_prime)).flatten(),Iw)
               
        def int_g(i,j):
            return complus_integral(fn_g1,i,j)

        def fn_intvg_x(i,j):
            return complus_integral(vgx,i,j)

        fn_rho_t = jax.grad(fn_rho, argnums=0)
        fn_rho_x = jax.grad(fn_rho, argnums=1)

    
        fn_g_t = jax.grad(fn_g, argnums=0)
        fn_g_x = jax.grad(fn_g, argnums=1)
    
        rho_t = fn_rho_t(t,x,mu)
        rho_x = fn_rho_x(t,x,mu)
        g_t = fn_g_t(t,x,mu)
        g_x = fn_g_x(t,x,mu)
        Avg_g = fn_intvg_x(t,x)
        rho,_ = model(t,x,mu)
        g = fn_g(t,x,mu)
    
        res1 = rho_t/c+Avg_g/sigma0
        res2 = eps**2*g_t + sigma0*mu*rho_x + eps * mu * g_x + sigmar * g - eps * Avg_g
    
        return res1,res2
    
    @partial(jax.jit, static_argnums=(0,))
    def loss_bc(model,*args,**kwargs):
        *o,val = args
        fn_g1 = lambda i,j,k: model(i,j,k)[1]

        def vgx(i,j,k):
            return k*jax.grad(fn_g1, argnums=1)(i,j,k)
            
        def fn_g(i,j,k):
            return fn_g1(i,j,k) - int_g(i,j)
        
        def complus_integral(f,i,j):
            Iw = w/2
            return jnp.dot((jax.vmap(lambda k: f(i,j,k),in_axes=0)(mu_prime)).flatten(),Iw)
               
        def int_g(i,j):
            return complus_integral(fn_g1,i,j)

        def fn_intvg_x(i,j):
            return complus_integral(vgx,i,j)
        tb,xb,ub = o
        rho,_ = model(*o)
        g = fn_g(tb,xb,ub)
        I = rho + eps*g/sigma0
        return I-val,rho-val
    @partial(jax.jit, static_argnums=(0,))
    def loss_ic(model,*args,**kwargs):
        *o, = args
        fn_g1 = lambda i,j,k: model(i,j,k)[1]
        def vgx(i,j,k):
            return k*jax.grad(fn_g1, argnums=1)(i,j,k)
            
        def fn_g(i,j,k):
            return fn_g1(i,j,k) - int_g(i,j)
        
        def complus_integral(f,i,j):
            Iw = w/2
            return jnp.dot((jax.vmap(lambda k: f(i,j,k),in_axes=0)(mu_prime)).flatten(),Iw)
               
        def int_g(i,j):
            return complus_integral(fn_g1,i,j)

        def fn_intvg_x(i,j):
            return complus_integral(vgx,i,j)
        rho,*_ = model(*o)

        g = fn_g(*o)
        return rho,eps*g/sigma0

if model_type in [0 ,1,2]:
    @partial(jax.jit, static_argnums=(0,))
    def loss_bc(model, *args,**kwargs):
        *o,val = args
        rho,g,_ = model(*o)
        I = rho + eps*g/sigma0
        return I-val,rho-val
    @partial(jax.jit, static_argnums=(0,))
    def loss_ic(model, *args,**kwargs):
        *o, = args
        rho,g,_ = model(*o)
        return rho,eps*g/sigma0

@partial(jax.jit, static_argnums=(0,))
def loss_ec(model, *args,**kwargs):
    *o,Tr,T = args
    fn_Tr = lambda i,j,k: model(i,j,k)[0]
    fn_T = lambda i,j,k: model(i,j,k)[1]
    return fn_Tr(*o)-Tr,fn_T(*o)-T


def loss_func(params,graphdef,freeze_params,Xr,Xb,Xi,Xe,config,*argv):
    NN = nnx.merge(graphdef,freeze_params, params)
    
    def loss_f(x,weight=1.0):
        return jnp.sum(jnp.mean(weight*jnp.square(x),axis=0),axis=-1)

    def loss_F(loss_R):
        loss_array = jnp.stack(loss_R, axis=-1)
        return loss_f(loss_array)
    
    def loss_rba(loss_R):
        res = jnp.stack(loss_R, axis=-1)
        r_norm = config['eta']*jnp.abs(res)/jnp.max(jnp.abs(res))

        rsum = config['rba']*(config['rsum']*(config['gamma']) + r_norm) + (1-config['rba'])*1
    
        rsum = jax.lax.stop_gradient(rsum)  
        return loss_f(res,1.),loss_f(res,rsum),rsum
    

    def loss_ca(loss_R):
        res = jnp.stack(loss_R, axis=-1)
        res_Tr,*_ = vmap(lambda *o: loss_ic(NN,*o))(*list(map(jnp.squeeze,jnp.split(Xi,Xi.shape[-1],-1))))
        *o,Tr_val,_ = jnp.split(Xi,Xi.shape[-1],1)
        tr,*_ = jnp.split(Xr,Xr.shape[-1],1)
        ic_L2 = jnp.linalg.norm(res_Tr)/jnp.linalg.norm(Tr_val)


        config["rsum"] = config['rba']*(jax.lax.stop_gradient(jnp.exp(-2e3*ic_L2*(tr-lb[0])))) + (1-config['rba'])*1

        return loss_f(res,1.0),loss_f(res,config["rsum"]),config["rsum"]

    def trans_squeeze(y):
        return jax.tree_util.tree_map(lambda x: x.squeeze() ,jnp.split(y,y.shape[-1],axis=-1) )


    loss_r,loss_r_weight,config["rsum"] = loss_rba(vmap(lambda *o: loss_res(NN,*o))(*trans_squeeze(Xr)))
    
    loss_b = loss_F(vmap(lambda *o: loss_bc(NN,*o))(*trans_squeeze(Xb)))

    loss_i = loss_F(vmap(lambda *o: loss_ic(NN,*o))(*trans_squeeze(Xi)))

    try :
        loss_e = loss_F(vmap(lambda *o: loss_ec(NN,*o))(*trans_squeeze(Xe)))
    except:
        loss_e = 0.
    
    alpha = config['alpha']
    return alpha[0]*loss_r_weight+alpha[1]*loss_b+alpha[2]*loss_i+alpha[3]*loss_e,(loss_r,loss_b,loss_i,loss_e,config)