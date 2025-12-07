import jax
import jax.numpy as jnp
import numpy as np
import optax
from flax import nnx
from examples.ex_1.getloss import loss_func
from examples.ex_1.sample import get_data
from examples.ex_1.plotting import draw
from utils.utils import *
from utils.common import shard_to_single
from getmodel import get_model
import time
from soap_jax import soap
import orbax.checkpoint as ocp


if __name__ == '__main__':
    checkpointer = ocp.StandardCheckpointer()
    def save_checkpointer(path,model):
        abs_path = os.path.abspath(path)
        checkpointer.save(abs_path, nnx.split(model), force=True)
        if isinstance(checkpointer, ocp.AsyncCheckpointer):
            checkpointer.wait_until_finished()
    def load_checkpointer(path,model):
        abs_path = os.path.abspath(path)
        graphdef , params = checkpointer.restore(abs_path,nnx.split(model))
        model = nnx.merge(graphdef, params)
        return model
    
    loss_func_jit = jax.jit(loss_func,static_argnums=(1,))
    grad_loss_func = jax.value_and_grad(loss_func_jit,has_aux=True)
    checkpointer = ocp.StandardCheckpointer()
    @jax.jit
    def train_step(params, graphdef, freeze_params, Xr, Xb, Xi, Xe, config, stats):
        (val, (loss_r, loss_b, loss_i, loss_e, config_new)), grads = grad_loss_func(
            params,  graphdef, freeze_params , Xr, Xb, Xi, Xe, config
        )
        grads, stats = optim.update(grads, stats, params)
        params_new = optax.apply_updates(params, grads)  
        return params_new, val, stats, grads, loss_r, loss_b, loss_i, loss_e, config_new

    def result_print(i, loss, loss_r, loss_b, loss_i, loss_e, config,scd,start,opt = "Adam"):
        global header_printed
        if header_printed:
            print('|  Optim  |  Epoch  |   Loss    |  Loss_r   |  Loss_b   |  Loss_i   |  Loss_e   | Weight_add| Lambda_PDE| Learnrate |   Time   |', end='\n')
            header_printed = False
        print('\r|  %5s  |  %6d | %.3e | %.3e | %.3e | %.3e | %.3e | %.3e | %.3e | %.3e | %8.2f |' % \
            (opt,i, loss, loss_r, loss_b, loss_i, loss_e,config["alpha"][-1], config["rsum"].mean(), scd(i), time.time() - start), end='\n')
        return 0
    model_old = None
    start = time.time()
    old_T = 0.
    config ={'alpha':[1e0,1e0,1e0,1e0],"rsum":0,"gamma":0.999,"eta":1e-2,"rba":0,"epoch":0}
    freeze_list = []
    for Time in [1.]:
        PINN = get_model(exp_id,model_type)
        model = PINN(seed,layer_sizes1,layer_sizes2)
        is_freeze = lambda path, node : any(param in path for param in freeze_list)
        graphdef, freeze_params , params = nnx.split(model,is_freeze,...)
        params = jax.device_put(params,sharding.replicate())
        freeze_params = jax.device_put(freeze_params,sharding.replicate())
        graphdef = jax.device_put(graphdef,sharding.replicate())
        config["rba"] =0.
        config["rsum"] =0.
        config["gamma"]=0.999
        config["eta"]=0.001
        try :
            os.mkdir(save_path+f'/figure/Time={Time}')
            os.mkdir(save_path+f'/figure/Time={Time}/weight')
            os.mkdir(save_path+f'/figure/Time={Time}/loss')
            os.mkdir(save_path+f'/figure/Time={Time}/prediction_figure')
            os.mkdir(save_path+f'/model_params/Time={Time}')
        except : 
            pass
        # """
        L = []    
        if old_T == 0.:

            scd = optax.exponential_decay(
                        init_value=5e-3, 
                        transition_steps=200, 
                        decay_rate=0.95, 
                        end_value= 1e-6
                        )

            Epoch = adam_epoch
        else:
            scd = optax.warmup_exponential_decay_schedule(
                        1e-7, 
                        1e-3, 
                        1_000, 
                        100, 
                        0.99, 
                        end_value=5e-6, 
                        )

            Epoch = adam_epoch
        optim = optax.chain(
            soap(
            learning_rate=scd,
            b1=0.99,
            b2=0.99,
            weight_decay=0.01,
            precondition_frequency=2,
            ))

        print('Time = %.2f'%(Time))
        stats = optim.init(params)

        Xr,Xb,Xi,Xe = get_data(seed,Time,Lb,Ub.at[0].set(Time),Nr,Nb,Ni,Ne,model_old,old_T)
        Xr = jax.device_put(Xr,sharding)
        Xb = jax.device_put(Xb,sharding)
        Xi = jax.device_put(Xi,sharding)
        Xe = jax.device_put(Xe,sharding)
        header_printed = 1
        for i in range(1,Epoch +1):
            
            params,loss,stats,grads,loss_r,loss_b,loss_i,loss_e,config = train_step(params,graphdef,freeze_params,Xr,Xb,Xi,Xe,\
                                                                                                                        config,stats)
            
            L.append((loss,loss_r,loss_b,loss_i,loss_e))
            if (i%1000)==0 or i==1:
                header_printed = result_print(i, loss, loss_r, loss_b, loss_i, loss_e, config,scd,start)
            
            if(i%2000)==0:
                print()
                model =nnx.merge(graphdef,shard_to_single(freeze_params), shard_to_single(params))
                L_array = jnp.array(jax.tree_util.tree_map(lambda x: float(x),L))
                draw(model,L_array,Time,save_path)
                header_printed = True
                
                jnp.save(save_path+f'/figure/Time={Time}'+'/loss/T%04d_Loss_adam.npy'%(Time*100), L_array)
                save_checkpointer(save_path+f'/model_params/Time={Time}'+"/T%04d_net_adam"%(Time*100), model)

        Xr = jax.device_put(Xr.reshape(-1,Xr.shape[-1]),sharding)
        Xb = jax.device_put(Xb.reshape(-1,Xb.shape[-1]),sharding)
        Xi = jax.device_put(Xi.reshape(-1,Xi.shape[-1]),sharding)
        Xe = jax.device_put(Xe.reshape(-1,Xe.shape[-1]),sharding)
        np.save(save_path+f'/figure/Time={Time}'+'/T%04d_point_Xr.npy'%(Time*100),Xr)
        np.save(save_path+f'/figure/Time={Time}'+'/T%04d_point_Xb.npy'%(Time*100),Xb)
        np.save(save_path+f'/figure/Time={Time}'+'/T%04d_point_Xi.npy'%(Time*100),Xi)
        np.save(save_path+f'/figure/Time={Time}'+'/T%04d_point_Xe.npy'%(Time*100),Xe)
        print()
        
        params_adam = params
        """
        ################  L-BFGS  ############

    #     config["rba"] =0 
        config["gamma"] = 1.
        config["eta"] = 0.
        optim = jaxopt.LBFGS(grad_loss_func,value_and_grad=True,has_aux=True,
                            history_size=150,
                            tol=1e-8,
                            maxiter=10000,
                            max_stepsize=2,
        #                      linesearch_init='max',
                            jit = True,
                            linesearch='zoom')

        sta = optim.init_state(params,graphdef,freeze_params,Xr,Xb,Xi,Xe,config)
        #     L = L[:10000]
        for i in range(1,1+lbfgs_epoch):
            params,sta = optim.update(params,sta,graphdef,freeze_params,Xr,Xb,Xi,Xe,config)
            loss_r,loss_b,loss_i,loss_e,config = sta.aux
            loss = sta.value
            L.append((loss,loss_r,loss_b,loss_i,loss_e))
            if (i%1000)==0 or i==1:
                header_printed = result_print(i, loss, loss_r, loss_b, loss_i, loss_e, config,scd,start,opt = "LBFGS")
            if (i%1000)==0 or i==1000:
                print()
                model = nnx.merge(graphdef,shard_to_single(freeze_params), shard_to_single(params))
                L_array = jnp.array(jax.tree_util.tree_map(lambda x: float(x),L))
                draw(model,L_array,Time,save_path)
                header_printed = True
                jnp.save(save_path+f'/figure/Time={Time}'+'/loss/T%04d_Loss_bfgs.npy'%(Time*100), L_array)
                save_checkpointer(save_path+f'/model_params/Time={Time}'+"/T%04d_net_bfgs"%(Time*100), model)

        """
        model_old = nnx.merge(graphdef,shard_to_single(freeze_params), shard_to_single(params))
        old_T = Time
        print()