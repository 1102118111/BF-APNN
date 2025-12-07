from utils.utils import *

def get_model(exp_id, model_type):
    if exp_id in [1] and model_type == 0:
        from models.fourier_1d import PINN_LRTE as PINN
    elif exp_id in [1] and model_type == 1:
        from models.legendre_1d import PINN_LRTE as PINN
    elif exp_id in [1] and model_type == 2:
        from models.bspline_1d import PINN_LRTE as PINN
    elif exp_id in [1] and model_type == 3:
        from models.rt_1d import PINN_LRTE as PINN
    elif exp_id in [1] and model_type == 4:
        from models.apnn_1d import PINN_LRTE as PINN
    elif exp_id in [2,3] and model_type == 0:
        from models.fourier_1d import PINN_GRTE as PINN
    elif exp_id in [2,3] and model_type == 1:
        from models.legendre_1d import PINN_GRTE as PINN
    elif exp_id in [2,3] and model_type == 2:
        from models.bspline_1d import PINN_GRTE as PINN
    elif exp_id in [2,3] and model_type == 3:
        from models.rt_1d import PINN_GRTE as PINN
    elif exp_id in [4,5,6] and model_type == 0:
        from models.rt_2d import PINN
    elif exp_id in [4,5,6] and model_type == 1:
        from models.fourier_2d import PINN
    elif exp_id in [4,5,6] and model_type == 3:
        from models.spherical_2d import PINN
    return PINN
