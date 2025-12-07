import jax
import jax.numpy as jnp
import numpy as np

def trans_squeeze(y):
    return jax.tree_util.tree_map(lambda x: x.squeeze() ,jnp.split(y,y.shape[-1],axis=-1) )
def shard_to_single(y): 
    return jax.tree_util.tree_map(lambda x: x.addressable_shards[0].data, y)


def gradtt(f,dim):
    return jax.grad(lambda *a,**k:jnp.sum(f(*a,**k)),dim)    
def jacott(f,dim,sumdim=0):
    return jax.jacrev(lambda *a,**k:jnp.sum(f(*a,**k),sumdim),dim)   

def vgrad(f,*x):
    y, vjp_fn = jax.vjp(f,*x)
    return vjp_fn(jnp.ones(y.shape))

def get_Integral_50():
    r,s,t = 1,np.sqrt(1/2),np.sqrt(1/3)
    u,v = np.sqrt(1/11),np.sqrt(9/11)
    B = 9216/725760*4*np.pi,16384/725760*4*np.pi,15309/725760*4*np.pi,14641/725760*4*np.pi
    Integral_50 = sum([[(s1*r,0,0,B[0]),(0,s1*r,0,B[0]),(0,0,s1*r,B[0])] for s1 in [-1,1]] +\
                      [[(s1*s,s2*s,0,B[1]),(0,s1*s,s2*s,B[1]),(s2*s,0,s1*s,B[1])] for s1 in [-1,1] for s2 in [-1,1]] +\
                      [[(s1*t,s2*t,s3*t,B[2])] for s1 in [-1,1] for s2 in [-1,1] for s3 in [-1,1]] +\
                      [[(s1*u,s2*u,s3*v,B[3]),(s1*v,s2*u,s3*u,B[3]),(s1*u,s2*v,s3*u,B[3])] for s1 in [-1,1] for s2 in [-1,1] for s3 in [-1,1]],start=[])
    return np.array(Integral_50)

def get_Integral_56():
    t,A = np.sqrt(1/3),9/560*4*np.pi
    r,s,B = np.sqrt((9-4*np.sqrt(3))/33),np.sqrt((15+8*np.sqrt(3))/33),(122+9*np.sqrt(3))/6720*4*np.pi
    v,u,C = np.sqrt((9+4*np.sqrt(3))/33),np.sqrt((15-8*np.sqrt(3))/33),(122-9*np.sqrt(3))/6720*4*np.pi
    Integral_56 = sum([[(s1*t,s2*t,s3*t,A)] for s1 in [-1,1] for s2 in [-1,1] for s3 in [-1,1]] +\
                      [[(s1*s,s2*r,s3*r,B),(s1*r,s2*s,s3*r,B),(s1*r,s2*r,s3*s,B)] for s1 in [-1,1] for s2 in [-1,1] for s3 in [-1,1]] +\
                      [[(s1*u,s2*v,s3*v,C),(s1*v,s2*u,s3*v,C),(s1*v,s2*v,s3*u,C)] for s1 in [-1,1] for s2 in [-1,1] for s3 in [-1,1]] ,start=[])
    return np.array(Integral_56)

def get_Integral_72():
    r = np.sqrt((5-np.sqrt(5))/10)
    s = np.sqrt((5+np.sqrt(5))/10)
    B = 125/10080*4*np.pi
    C = 143/10080*4*np.pi
    z = np.sqrt(np.roots(np.array([2556125,-5112250,3578575,-1043900,115115,-3562,9])))/(2*s)
    uvw = [[-z[2]+z[3],z[4]+z[5],z[0]+z[1]],
           [-z[4]+z[1],z[5]+z[3],z[0]+z[2]],
           [-z[1]+z[5],z[2]+z[4],z[0]+z[3]],
           [-z[5]+z[2],z[3]+z[1],z[0]+z[4]],
           [-z[3]+z[4],z[1]+z[2],z[0]+z[5]],]
    Integral_72 = sum([[(s1*r,s2*s,0,B),(0,s1*r,s2*s,B),(s2*s,0,s1*r,B)] for s1 in [-1,1] for s2 in [-1,1]] +\
                      [[(s1*u,s1*s2*v,s2*w,C),(s1*w,s1*s2*u,s2*v,C),(s1*v,s1*s2*w,s2*u,C)] for s1 in [-1,1] for s2 in [-1,1] for u,v,w in uvw],start=[])
    return np.array(Integral_72)