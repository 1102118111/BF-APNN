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
class Micro_Macro_Net_2d(pytc.TreeClass):
    def __init__(self,layers1_list=layers1_list,layers2_list=layers2_list,activation=activation,seed=seed):
        LA = layers1_list
        _,KA,KB = jr.split(jr.PRNGKey(seed),3)
        _,*KA = jr.split(KA,len(LA)+1)
        
        self.layers1 = Seq(*[Res2L(i,key,activation) if i==o else LinearAct(Linear(key , i, o, ),activation)
                          for key , i, o in zip(KA[:-1] , LA[:-2], LA[1:-1])])
        self.layers3 = Linear(KA[-1] , LA[-2], LA[-1], )
        LB = layers2_list
        _,*KB = jr.split(KB,len(LB)+1)
        self.layers2 = Seq(*[Res2L(i,key,activation) if i==o else LinearAct(Linear(key , i, o, ),activation)
                          for key, i, o  in zip(KB[:-1] , LB[:-2], LB[1:-1] )] + [Linear(KB[-1] , LB[-2], LB[-1] )])

    def summary(self):

        print(pytc.tree_diagram(self))
        

        def print_layer_params(layer, layer_name):
            print(f"\nLayer: {layer_name}")
            for param_name, param_value in layer.__dict__.items():
                if isinstance(param_value, jnp.ndarray):
                    print(f"  {param_name}: shape={param_value.shape}, dtype={param_value.dtype}, "
                          f"mean={jnp.mean(param_value):.4f}, std={jnp.std(param_value):.4f}, "
                          f"min={jnp.min(param_value):.4f}, max={jnp.max(param_value):.4f}")
        

        for idx, layer in enumerate(self.layers1.models):
            print_layer_params(layer, f"layers1.model[{idx}]")
        
        print_layer_params(self.layers3, "layers3")
        
        for idx, layer in enumerate(self.layers2.models):
            print_layer_params(layer, f"layers2.model[{idx}]")

    def get_layer_params(self, layer_name, layer_index=None):

        layer = getattr(self, layer_name)
        
        if layer_name in ['layers1', 'layers2'] and layer_index is not None:

            layer = layer.models[layer_index]
        
        if hasattr(layer, 'linear_i'): 
            w = layer.linear_i.w
            b = layer.linear_i.b
        elif hasattr(layer, 'linear_1'):
            w = layer.linear_1.w
            b = layer.linear_1.b
        elif hasattr(layer, 'w') and hasattr(layer, 'b'):  
            w = layer.w
            b = layer.b
        else:
            raise ValueError(f"Layer {layer_name} at index {layer_index} has no weights or biases.")
        
        print(f"Layer: {layer_name}, Index: {layer_index}")
        print(f"  Weights (w): {w}")
        print(f"  Biases (b): {b}")
        return w, b
    
    
    def __call__(self, *o):
        t,x,y,v1,v2 = o
        x1 = jnp.stack([t,x,y],-1).reshape(-1,3)
        x1 = affine(x1) 
        u = jnp.stack([v1,v2],-1).reshape(-1,2)
        X = jnp.concatenate([
                        x1, 
                        ],axis=-1)
        Y = self.layers1(X)
        TrT = outactivate(self.layers3(Y))
        g = self.layers2(jnp.concatenate((Y,u),-1))
        return jnp.concatenate((TrT,g),-1)