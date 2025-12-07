import jax
import jax.numpy as jnp
import argparse
import os
from jax.sharding import PositionalSharding

def parse_args():
    parser = argparse.ArgumentParser(
        description='Hyperparameters for multi-medium heat conduction model',
        add_help=True
    )

    parser.add_argument('--exp_id', type=int, default=1, help='Experiment ID')


    parser.add_argument('--seed', type=int, default=168168, help='Random seed for reproducibility')


    parser.add_argument('--dtype', type=jax._src.numpy.scalar_types._ScalarMeta, default=jnp.float64, help='Floating point type parameters')



    parser.add_argument('--model_type', type=int, default=1, help='Model type (0: Fourier, 1: Legendre, 2: Bspline 3: RT)')
    parser.add_argument('--model_width', type=int, default=40, help='Number of neurons in each layer')
    parser.add_argument('--model_depth', type=int, default=3,  help='Number of hidden layer depths')
    parser.add_argument('--embed_size', type=int, default=16,  help='Number of embed_size')
    parser.add_argument('--l', type=int, default=5,  help='Spherical order')
    

    parser.add_argument('--Nr', type=int, default=2**10, help='Number of resdual points')
    parser.add_argument('--Nb', type=int, default=2**10, help='Number of boundary points')
    parser.add_argument('--Ni', type=int, default=2**10, help='Number of initial points (None if not used)')
    parser.add_argument('--Ne', type=int, default=2**4, help='Number of extra points')
    parser.add_argument('--NS', type=int, default=10, help='Number of integrate points')


    parser.add_argument('--adam_epoch', type=int, default=10000, help='Maximum number of adam training epochs')
    parser.add_argument('--lbfgs_epoch', type=int, default=1000, help='Maximum number of lbfgs training epochs')
    
    args, _ = parser.parse_known_args()
    return args

args = parse_args()
dtype = args.dtype
if dtype == jnp.float64:
    jax.config.update("jax_enable_x64", True)
seed = args.seed
exp_id = args.exp_id
if exp_id in [1]:
    ipdim = 3
    outdim = 2
elif exp_id in [2,3]:
    ipdim = 3
    outdim = 3
elif exp_id in [4]:
    ipdim = 5
    outdim = 2
elif exp_id in [5,6]:
    ipdim = 5
    outdim = 3
embed_size = args.embed_size
model_type = args.model_type
model_width = args.model_width
layer_sizes1 = [2] + [model_width]*3 + [1]
layer_sizes2 = [model_width]*2 +[embed_size]


if exp_id in [1] and model_type == 0:
    layer_sizes1 = [2] + [model_width]*3 + [1]
    layer_sizes2 = [model_width]*2 +[embed_size]
    save_path = f'../results/ex_{exp_id}/Fourier_{embed_size}_seed{seed}'
elif exp_id in [1] and model_type == 1:
    save_path = f'../results/ex_{exp_id}/Legendre_{embed_size}_seed{seed}'
    layer_sizes1 = [2] + [model_width]*3 + [1]
    layer_sizes2 = [model_width]*2 +[embed_size+1]
elif exp_id in [1] and model_type == 2:
    layer_sizes1 = [2] + [model_width]*3 + [1]
    layer_sizes2 = [model_width]*2 +[embed_size]
    save_path = f'../results/ex_{exp_id}/Bspline_{embed_size}_seed{seed}'
elif exp_id in [1] and model_type == 3:
    layer_sizes1 = [2] + [model_width]*3 + [1]
    layer_sizes2 = [model_width+1] + [model_width]*1 +[1]
    NS = args.NS
    save_path = f'../results/ex_{exp_id}/RT_{NS}_seed{seed}'
elif exp_id in [1] and model_type == 4:
    layer_sizes1 = [2] + [model_width]*5 + [1]
    layer_sizes2 = [3] + [model_width]*5 +[1]
    NS = args.NS
    save_path = f'../results/ex_{exp_id}/APNN_{NS}_seed{seed}'
elif exp_id in [2,3] and model_type == 0:
    layer_sizes1 = [2] + [model_width]*3 + [2]
    layer_sizes2 = [model_width]*2 +[embed_size]
    save_path = f'../results/ex_{exp_id}/Fourier_{embed_size}_seed{seed}'
elif exp_id in [2,3] and model_type == 1:
    save_path = f'../results/ex_{exp_id}/Legendre_{embed_size}_seed{seed}'
    layer_sizes1 = [2] + [model_width]*3 + [2]
    layer_sizes2 = [model_width]*2 +[embed_size+1]
elif exp_id in [2,3] and model_type == 2:
    layer_sizes1 = [2] + [model_width]*3 + [2]
    layer_sizes2 = [model_width]*2 +[embed_size]
    save_path = f'../results/ex_{exp_id}/Bspline_{embed_size}_seed{seed}'
elif exp_id in [2,3] and model_type == 3:
    layer_sizes1 = [2] + [model_width]*3 + [2]
    layer_sizes2 = [model_width+1] + [model_width]*1 +[1]
    NS = args.NS
    save_path = f'../results/ex_{exp_id}/RT_{NS}_seed{seed}'

elif exp_id in [4,5,6] and model_type == 0:
    NS = args.NS
    layers1_list = [3] + [model_width]*3 + [1]
    layers2_list = [model_width + 2] +[model_width]*2 + [1]
    save_path = f'../results/ex_{exp_id}/RT_{NS}_seed{seed}'
elif exp_id in [4,5,6] and model_type == 1:
    l = args.l
    embed_size = 4*l**2
    layers1_list = [3] + [model_width]*3 + [1]
    save_path = f'../results/ex_{exp_id}/Fourier_{embed_size}_seed{seed}'
elif exp_id in [4,5,6] and model_type == 2:
    l = args.l
    embed_size = (l+1)**2-1
    layers1_list = [3] + [model_width]*3 + [1]
    layers2_list =  [model_width]*2 + [embed_size]
    save_path = f'../results/ex_{exp_id}/Spherical_{embed_size}_seed{seed}'
else:
    raise ValueError("Invalid model_type")

def create_directories(path, subdirs=["model_params", "figure"]):
    os.makedirs(path, exist_ok=True)
    for subdir in subdirs:
        os.makedirs(os.path.join(path, subdir), exist_ok=True)

create_directories(save_path)

Nr , Nb , Ni , Ne = args.Nr, args.Nb, args.Ni, args.Ne

adam_epoch = args.adam_epoch
lbfgs_epoch = args.lbfgs_epoch

num_devices = jax.local_device_count()
devices = jax.devices()[:num_devices]

sharding = PositionalSharding(jax.devices()).reshape(len(jax.devices()),1)
print(jax.devices())
print(sharding)

defult_activation = jnp.tanh

if model_type in [0 ,2]:
    Lb = jnp.array([0.,0.,0.])
    Ub = jnp.array([1.,1.,jnp.pi])
elif model_type in [1 ,3 ,4]:
    Lb = jnp.array([0,0,-1])
    Ub = jnp.array([1.,1.,1])