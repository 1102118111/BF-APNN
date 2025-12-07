import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np
from utils.utils import *
from functools import partial
from flax import nnx
from jax import vmap
from utils.common import gradtt, vgrad ,get_Integral_50

lb = Lb[:-2]
ub = Ub[:-2]
a = 1
c= 1
cv = 1
sigma = 1.
sigma0 = 1.
eps = 1.

if model_type in [1,2]:
    def loss_res(NN,Xr,seeds):
    
        def rho_txy(t,x,y,theta,phi):
            return a*c*(NN(t,x,y,theta,phi)[:,0:1])**4/2/np.pi
        def g1(t,x,y,u,v):
            return NN(t,x,y,u,v)[:,2:3]
            
        tr,xr,yr,thetar,phir = jnp.split(Xr,5,1)
        ur = jnp.sin(thetar)*jnp.cos(phir)
        vr = jnp.sin(thetar)*jnp.sin(phir)

        Tr,T,g,_,_ = jnp.split(NN(tr,xr,yr,thetar,phir),5,1)
        rho = a*c*Tr**4/2/np.pi
        Avg_g = gradtt(lambda t,x,y,u,v:NN(t,x,y,u,v)[:,3:4],1)(tr,xr,yr,thetar,phir) + \
                gradtt(lambda t,x,y,u,v:NN(t,x,y,u,v)[:,4:5],2)(tr,xr,yr,thetar,phir)
        rho_t,rho_x,rho_y = vgrad(lambda t,x,y:jnp.sum(rho_txy(t,x,y,thetar,phir)),tr,xr,yr)
        g_t,g_x,g_y = vgrad(lambda t,x,y:jnp.sum(g1(t,x,y,thetar,phir)),tr,xr,yr)
        T_t = gradtt(lambda t:jnp.sum(NN(t,xr,yr,thetar,phir)[:,1:2]),0)(tr)
        
        res1 = rho_t/c+Avg_g/sigma0+cv*T_t/2/np.pi
        res2 = eps**2/c*g_t + sigma0*(ur*rho_x + vr*rho_y) + eps*(ur*g_x+vr*g_y) + sigma*g - eps*(Avg_g)
        res3 = eps**2*cv*T_t-sigma*(2*np.pi*rho-a*c*T**4)
    
        return res1,res2,res3
    
    def loss_ic(NN,Xi):
        ti,xi,yi,ui,vi,rhoi,Ti,gi = jnp.split(Xi,5+3,-1)
        Tr,T,g,_,_  = jnp.split(NN(ti,xi,yi,ui,vi),5,-1)
        rho = a*c*Tr**4/2/np.pi
        return rho-rhoi,T-Ti,g-gi
    
    def loss_bc(NN,Xb):
        Nb = Xb.shape[0]//2
        tb,xb,yb,ub,vb = jnp.split(Xb,ipdim,-1)
        Trl,Tl,gl,_,_ = jnp.split(NN(tb,xb,yb,ub,vb),5,-1)
        xb = xb.at[:Nb].set(Ub[1])
        yb = yb.at[Nb:].set(Ub[2])
        Trb,Tb,gb,_,_ = jnp.split(NN(tb,xb,yb,ub,vb),5,-1)
        return Trl-Trb,Tl-Tb,gl-gb
    
    def loss_ec(NN,Xe):
        try:
            assert Xe.shape[0]!= (sharding.shape[0])**2
            te,xe,ye,ue,ve,val = jnp.split(Xe,ipdim+1,-1)
            return NN(te,xe,ye,ue,ve,)[:,0:2]-val,
        except:
            return jnp.zeros((1,1))

elif model_type in [0]:
    Ixyzw = get_Integral_50()
    def loss_res(NN,Xr,seeds):
        key = jr.PRNGKey(seeds)
        T_orth = jr.orthogonal(key,3)
        def div_omega_f(f,t,x,y,v1,v2):
            f_x,f_y = vgrad(lambda x,y:jnp.sum(f(t,x,y,v1,v2)),x,y)
            return v1*f_x+v2*f_y
    
        def rho_txy(t,x,y,v1,v2):
            return a*c*(NN(t,x,y,v1,v2)[:,0:1])**4/2/np.pi
        
        def g1(t,x,y,u,v):
            return NN(t,x,y,u,v)[:,-1:]
        
        def int_g(t,x,y):
            return jax.vmap(lambda t,x,y:complus_integral(g1,t,x,y))(t,x,y).reshape(-1,1)
        
        def g_txy(t,x,y,v1,v2):
            return g1(t,x,y,v1,v2) - int_g(t,x,y)
            
        def complus_integral(f,t,x,y,):
            Ix,Iy,Iz = jnp.split(Ixyzw[:,:3]@T_orth,3,1)
            Iw = Ixyzw[:,3:4]/4/np.pi
            return f(*jnp.broadcast_arrays(t,x,y,Ix,Iy,)).T@Iw
    
        def vgx(tr,xr,yr,ur,vr):
            g_x,g_y = vgrad(lambda x,y:jnp.sum(g1(tr,x,y,ur,vr)),xr,yr)
            return ur*g_x+vr*g_y
    
        def avg_g(tr,xr,yr):
            return jax.vmap(lambda t,x,y:complus_integral(vgx,t,x,y,))(tr,xr,yr).reshape(-1,1)
    
            
        tr,xr,yr,ur,vr = jnp.split(Xr,5,1)
        Tr,T,_ = jnp.split(NN(tr,xr,yr,ur,vr),3,1)
        g = g_txy(tr,xr,yr,ur,vr)
        rho = a*c*Tr**4/2/np.pi
        Avg_g = avg_g(tr,xr,yr)
        rho_t,rho_x,rho_y = vgrad(lambda t,x,y:jnp.sum(rho_txy(t,x,y,ur,vr)),tr,xr,yr)
        g_t,g_x,g_y = vgrad(lambda t,x,y:jnp.sum(g_txy(t,x,y,ur,vr)),tr,xr,yr)
        T_t = gradtt(lambda t:jnp.sum(NN(t,xr,yr,ur,vr)[:,1:2]),0)(tr)
        
        res1 = rho_t/c+Avg_g/sigma0+cv*T_t/2/np.pi
        res2 = eps**2/c*g_t + sigma0*(ur*rho_x + vr*rho_y) + eps*(ur*g_x+vr*g_y) + sigma*g - eps*(Avg_g)
        res3 = eps**2*cv*T_t-sigma*(2*np.pi*rho-a*c*T**4)
    
        return res1,res2,res3
    
    def loss_ic(NN,Xi):
        def complus_integral(f,t,x,y,):
            Ix,Iy,Iz = jnp.split(Ixyzw[:,:3],3,1)
            Iw = Ixyzw[:,3:4]/4/np.pi
            return f(*jnp.broadcast_arrays(t,x,y,Ix,Iy,)).T@Iw
        def g1(t,x,y,u,v):
            return NN(t,x,y,u,v)[:,-1:]
        
        def int_g(t,x,y):
            return jax.vmap(lambda t,x,y:complus_integral(g1,t,x,y))(t,x,y).reshape(-1,1)
        
        def g_txy(t,x,y,v1,v2):
            return g1(t,x,y,v1,v2) - int_g(t,x,y)
        ti,xi,yi,ui,vi,rhoi,Ti,gi = jnp.split(Xi,5+3,-1)
        Tr,T,_  = jnp.split(NN(ti,xi,yi,ui,vi),3,-1)
        g = g_txy(ti,xi,yi,ui,vi)
        rho = a*c*Tr**4/2/np.pi
        return rho-rhoi,T-Ti,g-gi
    

    def loss_bc(NN,Xb):
        Nb = Xb.shape[0]//2
        tb,xb,yb,ub,vb = jnp.split(Xb,ipdim,-1)
        Trl,Tl,gl = jnp.split(NN(tb,xb,yb,ub,vb),3,-1)
        xb = xb.at[:Nb].set(Ub[1])
        yb = yb.at[Nb:].set(Ub[2])
        Trb,Tb,gb = jnp.split(NN(tb,xb,yb,ub,vb),3,-1)
        return Trl-Trb,Tl-Tb,gl-gb

    def loss_ec(NN,Xe):
        try:
            assert (sharding.shape[0])**2
            te,xe,ye,ue,ve,val_Tr,val_Te = jnp.split(Xe,Xe.shape[-1],1)
            Tr,T,_ = jnp.split(NN(te,xe,ye,ue,ve),outdim,1)
            return Tr-val_Tr,T-val_Te
        except:
            return jnp.zeros((1,1)),
        
    

    
@jax.jit
def loss_func(NN,Xr,Xb,Xi,Xe,config={'alpha':[1.0,1.0,1.0,1.0],"gamma":0.999,"eta":1e-3,"rsum":0,"rba":0,"epoch":0},*argv):
    def loss_f(x,weight=1.0):
        return jnp.sum(jnp.mean(jnp.square(weight*x),axis=0))

    def loss_F(loss_R):
        # 将元组中的元素沿新轴堆叠，然后计算平方和的均值
        loss_array = jnp.concatenate(loss_R, axis=-1)
        return loss_f(loss_array)
    
    def loss_rba(loss_R):
        res = jnp.concatenate(loss_R, axis=-1)
        r_norm = config['eta']*jnp.abs(res)/jnp.max(jnp.abs(res))
#         rsum = config['rba']*(config['rsum']*0.999 + r_norm) + (1-config['rba'])*config['rsum']
        rsum = config['rba']*(config['rsum']*(config['gamma']) + r_norm) + (1-config['rba'])*1
    
        rsum = jax.lax.stop_gradient(rsum)  # 阻止 rsum 追踪梯度
        return loss_f(res),loss_f(res,rsum),rsum

    # 计算 loss_r 和 loss_b
    loss_r,loss_r_weight,config["rsum"] = loss_rba(loss_res(NN, Xr,config['epoch'] ))
    loss_b = loss_F(loss_bc(NN, Xb))
    loss_i = loss_F(loss_ic(NN, Xi))
    loss_e = loss_F(loss_ec(NN, Xe))

    
    alpha = config['alpha']
    return alpha[0]*loss_r_weight+alpha[1]*loss_b+alpha[2]*loss_i+alpha[3]*loss_e,(loss_r,loss_b,loss_i,loss_e,config)


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