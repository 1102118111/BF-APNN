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
    fig, ax = plt.subplots(2, 5, figsize=(15, 5),dpi=200)
    ax = ax.flatten()
    for i in range(5):
        path = ('../data/RS200/fort.100%d') % (i+1)
        data = np.loadtxt(path)
        x, y, Tr, T  = np.split(data, 4, -1)
        n = int(len(x)**0.5)
        X = x.reshape(n,n)
        Y = y.reshape(n,n)
        t = jnp.zeros_like(x,) + i * 0.2 + 0.2
        Pred = model(t,x,y,y,y)
        Tr_pred, T_pred, *_ = jnp.split(Pred, Pred.shape[-1], -1)
        Tr_pred, T_pred = abs(Tr_pred), abs(T_pred)
        error_Tr, error_T = abs(Tr_pred - Tr), abs(T_pred - T)


        contour = ax[i].contourf(X, Y, error_Tr.reshape(n, n), cmap="jet", levels=51)

        plt.colorbar(contour, ax=ax[i])
        ax[i].set_xlabel('x')
        ax[i].set_ylabel('y')
        ax[i].set_title('Error Tr - t=%.1f' % (i * 0.2+0.2))
        ax[i].set_xticks([])
        ax[i].set_yticks([])

        ax[i].invert_yaxis()


        contour = ax[i+5].contourf(X, Y, error_T.reshape(n, n), cmap="jet", levels=51)

        plt.colorbar(contour, ax=ax[i+5])
        ax[i + 5].set_xlabel('x')
        ax[i + 5].set_ylabel('y')
        ax[i + 5].set_title('Error T - t=%.1f' % (i * 0.2+0.2))
        ax[i + 5].set_xticks([])
        ax[i + 5].set_yticks([])
        ax[i + 5].invert_yaxis()
        print('error_Tr_t=%.1f:%.2e' % (i * 0.2+0.2, np.linalg.norm(error_Tr, 2) / np.linalg.norm(Tr, 2)))
        print('error_Te_t=%.1f:%.2e' % (i * 0.2+0.2, np.linalg.norm(error_T, 2) / np.linalg.norm(T, 2)))
    plt.tight_layout()

    try : 
        plt.savefig(PATH + f'/figure/Time={Time}/prediction_figure/Time_{int(Time * 1000)}_{len(L)}.pdf', bbox_inches='tight')
    except:
        pass
    plt.close()