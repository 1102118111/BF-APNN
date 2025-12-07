import numpy as np
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt


def l2_error(x,y):
    return np.linalg.norm(x-y)/np.linalg.norm(y)

def draw(model,L,Time,PATH):
    from matplotlib import rcParams
    from matplotlib.ticker import ScalarFormatter
    fontsize = 20
    rcParams['font.size'] = 11
    rcParams['axes.titlesize'] = 14
    rcParams['axes.labelsize'] = 14
    rcParams['legend.fontsize'] = 14
    rcParams['grid.linestyle'] = '--'
    rcParams['grid.alpha'] = 0.5
    
    nL = jnp.array(L)[:, :]  
    it = jnp.arange(len(nL)) + 1   
    lowerL = np.minimum.accumulate(nL, axis=0)  
    upperL = np.maximum.accumulate(nL[::-1], axis=0)[::-1]  
    
    colors = ["k", '#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
    names = ["L", 'L_r', 'L_b', 'L_i', 'L_e']
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6), dpi=150)
    
    axes = [(ax1, 'linear', 'log', 'Log-Y Scale'), 
            (ax2, 'log', 'log', 'Log-Log Scale')]
    
    for ax, xscale, yscale, title in axes:
        for k, (name, color) in enumerate(zip(names, colors)):
            ax.plot(it, nL[:, k], color=color, linewidth=2, linestyle='--', alpha=0.5, label=name)
            ax.plot(it, lowerL[:, k], color=color, linewidth=4, linestyle='-', alpha=0.9)
        
        ax.set_xscale(xscale)
        ax.set_yscale(yscale)
        ax.set_xlim(1, len(it))  
    
        if xscale == 'linear':
            ax.xaxis.set_major_formatter(ScalarFormatter(useMathText=True))
            ax.ticklabel_format(style='sci', axis='x', scilimits=(0, 0))
    
        ax.legend(loc='upper right', frameon=True, framealpha=0.9, edgecolor='gray')
        ax.grid(True, which="both", linestyle='--', alpha=0.5)
        ax.set_title(title, pad=10)
        ax.set_xlabel('Iteration', labelpad=5)
        ax.set_ylabel('Loss', labelpad=5)
    
    plt.tight_layout()
    try : 
        plt.savefig(f'{PATH}/figure/Time={Time}/loss/Time_{int(Time*1000):05d}_Loss.pdf', bbox_inches='tight')
    except:
        pass
    plt.clf()
    plt.close()
    data = np.loadtxt('../data/输运参考解数据/线性输运方程/Reference_intermediate_epsilon=10-02/rho_intermediate(10-02)_reference_1.txt',skiprows=0)
    rho_t0 = data[:,1:2]
    rho_t1 = data[:,2:3]
    rho_t2 = data[:,3:4]
    rho_t3 = data[:,4:5]
    rho_t4 = data[:,5:]
    xx = data[:,0:1]
    nx = xx.shape[0]

    t_values = [0.2, 0.4, 0.6, 0.8,1.0]
    rho_ref_list = [rho_t0, rho_t1, rho_t2, rho_t3,rho_t4]
    markers = ["b1", "c*", "rx", "yo","k+",]
    linestyles = ['--', '-', '-.', '--',"-"]
    colors = ['b', "c", 'r', 'y','k',]
    
    def Model(t,x,theta):
        t = t.squeeze()
        x = x.squeeze()
        theta = theta.squeeze()
        model_vmap = jax.vmap(model)
        return model_vmap(t,x,theta)
        
    def predict_rho(t, xx):
        t = jnp.ones((nx, 1)) * t
        return Model(t, xx, xx)[:, 0:1]

    rho_pred_list = [predict_rho(t, xx) for t in t_values]

    plt.figure(figsize=(8, 8))
    for i, (rho_pred, rho_ref, marker, linestyle, color) in enumerate(zip(rho_pred_list, rho_ref_list, markers, linestyles, colors)):
        plt.plot(xx, rho_pred, marker, markersize=8, markevery=200, label=f"$\\rho_{{nn}}(t={t_values[i]})$")
        plt.plot(xx, rho_ref, linestyle=linestyle, color=color, linewidth=2, label=f"$\\rho_{{ref}}(t={t_values[i]})$")

    plt.xlabel(r"$x$", fontsize=20)
    plt.ylabel(r"$\rho$", fontsize=20)
    plt.ylim(0, 1)
    plt.grid()
    plt.legend(fontsize=16)
    plt.subplots_adjust(left=0.19, right=0.94, bottom=0.13)
    try : 
        plt.savefig(PATH + f'/figure/Time={Time}/prediction_figure/Time_{int(Time * 1000)}_{len(L)}.pdf', bbox_inches='tight')
    except:
        pass
    plt.close()
    
    error_list = [f"{l2_error(rho_pred, rho_ref):4.2e}" for rho_pred, rho_ref in zip(rho_pred_list, rho_ref_list)]
    print(f"L2 error: {error_list}")