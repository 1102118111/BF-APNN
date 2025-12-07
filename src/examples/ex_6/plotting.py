import numpy as np
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt


def l2_error(x,y):
    return np.linalg.norm(x-y)/np.linalg.norm(y)

def draw(Model,L,Time,PATH):
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
    n = 601
    step = 2
    data = np.loadtxt('../data/exact_line_source_2d_plot.txt')
    x, y, Rho_True = np.split(data, 3, -1)
    X,Y,Rho_True = x.reshape(n,n)[1::step,1::step],y.reshape(n,n)[1::step,1::step],Rho_True.reshape(n,n)[1::step,1::step],
    n = (n-1)//step
    x,y,rho_True = X.reshape(-1,1),Y.reshape(-1,1),Rho_True.reshape(-1,1)
    vmin = rho_True.min()
    vmax = rho_True.max()
    t = jnp.ones_like(x)  
    rho_pred,*_ = jnp.split(Model(t, x, y, y, y), 4, axis=1)
    pred = 4 * jnp.pi * jax.device_get(rho_pred).reshape(n,n)
    
    plt.figure(figsize=(18, 5))
    plt.subplot(1, 3, 1)
    plt.contourf(X, Y, pred.reshape(n,n), cmap='jet',levels=np.linspace(vmin, vmax, 101),extend="both" )
    plt.colorbar()
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title(r'$\rho_{pred}$ - t=%.1f' % (0.2 * 5))


    plt.subplot(1, 3, 2)
    plt.contourf(X, Y, Rho_True, cmap='jet', levels=np.linspace(vmin, vmax, 101))
    plt.colorbar()
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title(r'$\rho_{true}$ - t=%.1f' % (0.2 * 5))


    plt.subplot(1, 3, 3)
    plt.contourf(X, Y,  jnp.abs(Rho_True - pred).reshape(n,n), cmap='jet', levels=101)
    plt.colorbar()
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title('error - t=%.1f' % (0.2 * 5))

    plt.tight_layout()
    try : 
        plt.savefig(PATH + f'/figure/Time={Time}/prediction_figure/Time_{int(Time * 1000)}_duibi_{len(L)}.pdf', bbox_inches='tight')
    except:
        pass

    plt.close()

    l2_error = jnp.linalg.norm(pred - Rho_True) / jnp.linalg.norm(Rho_True)
    print("Relative L2 Error:", l2_error)