import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np
from utils.utils import *
from functools import partial
from flax import nnx
from jax import vmap
from utils.common import vgrad,gradtt,get_Integral_50


c = 1
eps = 1
a = 1.5
left_boundry = -a;right_boundry = a
sigma0 = np.sqrt(1)
Lb = jnp.array([0.,left_boundry,left_boundry,-1,-1])
Ub = jnp.array([1.,right_boundry,right_boundry,1,1])
lb = Lb[:-2]
ub = Ub[:-2]

eps = 1.

if model_type in [1,2]:

    def loss_res(NN,Xr,seeds):
    
        def rho_txy(t,x,y,u,v):
            return NN(t,x,y,u,v)[:,0:1]
            
        def g1(t,x,y,u,v):
            return NN(t,x,y,u,v)[:,1:2]
            
        tr,xr,yr,thetar,phir,sigmar, = jnp.split(Xr,6,1)
        ur = jnp.sin(thetar)*jnp.cos(phir)
        vr = jnp.sin(thetar)*jnp.sin(phir)
        rho_t,rho_x,rho_y = vgrad(lambda t,x,y:jnp.sum(rho_txy(t,x,y,thetar,phir)),tr,xr,yr)
        g_t,g_x,g_y = vgrad(lambda t,x,y:jnp.sum(g1(t,x,y,thetar,phir)),tr,xr,yr)
        
        Avg_g = gradtt(lambda t,x,y,u,v:NN(t,x,y,u,v)[:,2:3],1)(tr,xr,yr,thetar,phir) + \
                gradtt(lambda t,x,y,u,v:NN(t,x,y,u,v)[:,3:4],2)(tr,xr,yr,thetar,phir)
        rho,g,_,_ = jnp.split(NN(tr,xr,yr,thetar,phir),4,1)
    
        res1 = rho_t+Avg_g/sigma0
        res2 = eps**2/c*g_t + sigma0*(ur*rho_x + vr*rho_y) + eps*(ur*g_x+vr*g_y) + sigmar*g - eps*(Avg_g)

        return jnp.concatenate((res1,res2),0),
    

    def loss_ic(NN,Xi):
        ti,xi,yi,ui,vi,vali, = jnp.split(Xi,6,1)
        rho,g,_,_ = jnp.split(NN(ti,xi,yi,ui,vi),4,1)
        I = rho+eps*g/sigma0
        return jnp.concatenate((I-vali,rho-vali),0),

    def loss_bc(NN,Xb):
        td,xd,yd,ud,vd,vald = jnp.split(Xb,6,1)
        rho,g,_,_ = jnp.split(NN(td,xd,yd,ud,vd),4,1)
        I = rho + eps*g/sigma0
        return I-vald,

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
            return NN(t,x,y,v1,v2)[:,0:1]
        
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
            
        tr,xr,yr,ur,vr,sigmar, = jnp.split(Xr,6,1)
        rho,_ = jnp.split(NN(tr,xr,yr,ur,vr),2,1)
        g = g_txy(tr,xr,yr,ur,vr)
        Avg_g = avg_g(tr,xr,yr)
        rho_t,rho_x,rho_y = vgrad(lambda t,x,y:jnp.sum(rho_txy(t,x,y,ur,vr)),tr,xr,yr)
        g_t,g_x,g_y = vgrad(lambda t,x,y:jnp.sum(g_txy(t,x,y,ur,vr)),tr,xr,yr)
        
        Avg_g = avg_g(tr,xr,yr)
    
        res1 = rho_t+Avg_g/sigma0
        res2 = eps**2/c*g_t + sigma0*(ur*rho_x + vr*rho_y) + eps*(ur*g_x+vr*g_y) + sigmar*g - eps*(Avg_g)
        return jnp.concatenate((res1,res2),0),
    
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
        ti,xi,yi,ui,vi,vali, = jnp.split(Xi,6,1)
        rho,_, = jnp.split(NN(ti,xi,yi,ui,vi),2,1)
        g = g_txy(ti,xi,yi,ui,vi)
        I = rho+eps*g/sigma0
        return jnp.concatenate((I-vali,rho-vali),0),

    def loss_bc(NN,Xb):
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
        td,xd,yd,ud,vd,vald = jnp.split(Xb,6,1)
        rho,_, = jnp.split(NN(td,xd,yd,ud,vd),2,1)
        g = g_txy(td,xd,yd,ud,vd)
        I = rho + eps*g/sigma0
        return I-vald,
    

    def loss_ec(NN,Xe):
        try:
            assert (sharding.shape[0])**2
            te,xe,ye,ue,ve,val_Tr,val_Te = jnp.split(Xe,Xe.shape[-1],1)
            Tr,T,_ = jnp.split(NN(te,xe,ye,ue,ve),outdim,1)
            return Tr-val_Tr,T-val_Te
        except:
            return jnp.zeros((1,1)),
if model_type in [1,2]:

    def loss_res(NN,Xr,seeds):
    
        def rho_txy(t,x,y,u,v):
            return NN(t,x,y,u,v)[:,0:1]
            
        def g1(t,x,y,u,v):
            return NN(t,x,y,u,v)[:,1:2]
            
        tr,xr,yr,thetar,phir,sigmar, = jnp.split(Xr,6,1)
        ur = jnp.sin(thetar)*jnp.cos(phir)
        vr = jnp.sin(thetar)*jnp.sin(phir)
        rho_t,rho_x,rho_y = vgrad(lambda t,x,y:jnp.sum(rho_txy(t,x,y,thetar,phir)),tr,xr,yr)
        g_t,g_x,g_y = vgrad(lambda t,x,y:jnp.sum(g1(t,x,y,thetar,phir)),tr,xr,yr)
        
        Avg_g = gradtt(lambda t,x,y,u,v:NN(t,x,y,u,v)[:,2:3],1)(tr,xr,yr,thetar,phir) + \
                gradtt(lambda t,x,y,u,v:NN(t,x,y,u,v)[:,3:4],2)(tr,xr,yr,thetar,phir)
        rho,g,_,_ = jnp.split(NN(tr,xr,yr,thetar,phir),4,1)
    
        res1 = rho_t+Avg_g/sigma0
        res2 = eps**2/c*g_t + sigma0*(ur*rho_x + vr*rho_y) + eps*(ur*g_x+vr*g_y) + sigmar*g - eps*(Avg_g)
        return jnp.concatenate((res1,res2),0),
    

    def loss_ic(NN,Xi):
        ti,xi,yi,ui,vi,vali, = jnp.split(Xi,6,1)
        rho,g,_,_ = jnp.split(NN(ti,xi,yi,ui,vi),4,1)
        I = rho+eps*g/sigma0
        return jnp.concatenate((I-vali,rho-vali),0),

    def loss_bc(NN,Xb):
        td,xd,yd,ud,vd,vald = jnp.split(Xb,6,1)
        rho,g,_,_ = jnp.split(NN(td,xd,yd,ud,vd),4,1)
        I = rho + eps*g/sigma0
        return I-vald,

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
            return NN(t,x,y,v1,v2)[:,0:1]
        
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
            
        tr,xr,yr,ur,vr,sigmar, = jnp.split(Xr,6,1)
        rho,_ = jnp.split(NN(tr,xr,yr,ur,vr),2,1)
        g = g_txy(tr,xr,yr,ur,vr)
        Avg_g = avg_g(tr,xr,yr)
        rho_t,rho_x,rho_y = vgrad(lambda t,x,y:jnp.sum(rho_txy(t,x,y,ur,vr)),tr,xr,yr)
        g_t,g_x,g_y = vgrad(lambda t,x,y:jnp.sum(g_txy(t,x,y,ur,vr)),tr,xr,yr)
        
        Avg_g = avg_g(tr,xr,yr)
    
        res1 = rho_t+Avg_g/sigma0
        res2 = eps**2/c*g_t + sigma0*(ur*rho_x + vr*rho_y) + eps*(ur*g_x+vr*g_y) + sigmar*g - eps*(Avg_g)

        return jnp.concatenate((res1,res2),0),
    
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
        ti,xi,yi,ui,vi,vali, = jnp.split(Xi,6,1)
        rho,_, = jnp.split(NN(ti,xi,yi,ui,vi),2,1)
        g = g_txy(ti,xi,yi,ui,vi)
        I = rho+eps*g/sigma0
        return jnp.concatenate((I-vali,rho-vali),0),


    def loss_bc(NN,Xb):
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
        td,xd,yd,ud,vd,vald = jnp.split(Xb,6,1)
        rho,_, = jnp.split(NN(td,xd,yd,ud,vd),2,1)
        g = g_txy(td,xd,yd,ud,vd)
        I = rho + eps*g/sigma0
        return I-vald,
    

    def loss_ec(NN,Xe):
        try:
            assert (sharding.shape[0])**2
            te,xe,ye,ue,ve,val_Tr,val_Te = jnp.split(Xe,Xe.shape[-1],1)
            Tr,T,_ = jnp.split(NN(te,xe,ye,ue,ve),outdim,1)
            return Tr-val_Tr,T-val_Te
        except:
            return jnp.zeros((1,1)),


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