import jax.numpy as jnp
import jax.scipy as jsp
import jax.random as jr
import pytreeclass as pytc
from .resblock import *
from utils.utils import *
def affine(x,Lb,Ub):
    return (x-Lb)/(Ub-Lb) * 2 - 1
def outactivate(x):
    return x
lm_pairs = []

for i in range(1, l+1):  
    lm_pairs.append([i, 0])  
for i in range(1, l+1):  
    for j in range(1, i + 1):  
        lm_pairs.append([i, j])  


lm_array = jnp.array(lm_pairs)

lm_array = jnp.concatenate((jnp.flip(lm_array[l:],0),lm_array), axis=0)
lm_array = lm_array.at[:l*(l+1)//2,1].set(lm_array[:l*(l+1)//2,1]*-1)


l_scale = lm_array[:,0]
m_scale = lm_array[(embed_size +l)//2:,1]

def spherical_harmonic(l, m, theta, phi):
    P_lm = jnp.transpose(jsp.special.lpmn_values(m, l, jnp.cos(theta).ravel(),is_normalized=True)[...,None],[1,0,2,3])
    m = ((jnp.arange(0,m+1)*jnp.ones_like(phi)).T)[...,None]

    harmonic_1 =  P_lm * jnp.exp(1j * m * phi)
    harmonic_2 = jnp.flip((-1)**(m[1:])*harmonic_1.conjugate()[:,1:],1)
    return (jnp.concatenate([harmonic_1, harmonic_2], axis=1)[lm_array[:,0],lm_array[:,1]].squeeze().T)/l_scale


class PINN(pytc.TreeClass):
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
        weight_zero_re = weight[:,:l]
        weight_zero_im = jnp.zeros_like(weight_zero_re)
        weight_postive = weight[:,l:]
        weight_postive_re,weight_postive_im = jnp.split(weight_postive,2,-1)
        weight_negative_re,weight_negative_im = jnp.flip((-1)**m_scale*weight_postive_re,1),jnp.flip(-1*(-1)**m_scale*weight_postive_im,1)
        weight_re = jnp.concatenate([weight_negative_re , weight_zero_re,weight_postive_re,],-1)
        weight_im = jnp.concatenate([weight_negative_im , weight_zero_im,weight_postive_im,],-1)


        base = spherical_harmonic(l, l, theta, phi)
        g = jnp.sum(weight_re*base.real - weight_im*base.imag,axis=-1,keepdims=True)
        
        
        int_v1g = ((weight_re[:,l*(l+1)//2-1:l*(l+1)//2]-weight_re[:,-l*(l+1)//2:-l*(l+1)//2+1])*jnp.sqrt(2*jnp.pi/3)/4/jnp.pi)
        int_v2g = ((weight_im[:,l*(l+1)//2-1:l*(l+1)//2]+weight_im[:,-l*(l+1)//2:-l*(l+1)//2+1])*jnp.sqrt(2*jnp.pi/3)/4/jnp.pi)
        return jnp.concatenate((rho,g,int_v1g,int_v2g),-1)
    

class PINN_GRTE(pytc.TreeClass):
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
        weight_zero_re = weight[:,:l]
        weight_zero_im = jnp.zeros_like(weight_zero_re)
        weight_postive = weight[:,l:]
        weight_postive_re,weight_postive_im = jnp.split(weight_postive,2,-1)
        weight_negative_re,weight_negative_im = jnp.flip((-1)**m_scale*weight_postive_re,1),jnp.flip(-1*(-1)**m_scale*weight_postive_im,1)
        weight_re = jnp.concatenate([weight_negative_re , weight_zero_re,weight_postive_re,],-1)
        weight_im = jnp.concatenate([weight_negative_im , weight_zero_im,weight_postive_im,],-1)


        base = spherical_harmonic(l, l, theta, phi)
        g = jnp.sum(weight_re*base.real - weight_im*base.imag,axis=-1,keepdims=True)
        
        
        int_v1g = ((weight_re[:,l*(l+1)//2-1:l*(l+1)//2]-weight_re[:,-l*(l+1)//2:-l*(l+1)//2+1])*jnp.sqrt(2*jnp.pi/3)/4/jnp.pi)
        int_v2g = ((weight_im[:,l*(l+1)//2-1:l*(l+1)//2]+weight_im[:,-l*(l+1)//2:-l*(l+1)//2+1])*jnp.sqrt(2*jnp.pi/3)/4/jnp.pi)
        return jnp.concatenate((rho,g,int_v1g,int_v2g),-1)