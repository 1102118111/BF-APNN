import numpy as np
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt


a = 1.5
left_boundry = -a;right_boundry = a
k = 2

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

    n = 101
    x, y = jnp.linspace(left_boundry, right_boundry, n), jnp.linspace(left_boundry, right_boundry, n)
    X, Y = jnp.meshgrid(x, y, indexing='ij')

    x, y = X.reshape(-1, 1), Y.reshape(-1, 1)
    t = jnp.ones_like(x) 
    val = model(t, x, y, y, y)
    rho_pred,*_ = jnp.split(val, val.shape[-1], axis=1)


    pred = 4 * jnp.pi * jax.device_get(rho_pred).reshape(n,n)


    data = np.loadtxt(f'../data/a={k}_new.dat')
    data = np.flip(data, axis=1).reshape(n,n)

    vmin = data.min()
    vmax = data.max()


    plt.figure(figsize=(17, 5))

    plt.subplot(1, 3, 1)
    plt.contourf(X, Y, pred.reshape(n,n), cmap='jet', levels=np.linspace(vmin, vmax, 101))
    plt.plot([-1.5, 1.5], [1.5, -1.5], color='black', linestyle='--')
    plt.plot([-1.5, 1.5], [-1.5, 1.5], color='r', linestyle='--')
    plt.colorbar()
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title(r'$\rho_{pred}$ - t=%.1f' % (0.2 * 5))


    plt.subplot(1, 3, 2)
    plt.contourf(X, Y, data.reshape(n,n), cmap='jet', levels=np.linspace(vmin, vmax, 101))
    plt.plot([-1.5, 1.5], [1.5, -1.5], color='black', linestyle='--')
    plt.plot([-1.5, 1.5], [-1.5, 1.5], color='r', linestyle='--')
    plt.colorbar()
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title(r'$\rho_{true}$ - t=%.1f' % (0.2 * 5))


    plt.subplot(1, 3, 3)
    plt.contourf(X, Y,  jnp.abs(data - pred).reshape(n,n), cmap='jet', levels=101)
    plt.plot([-1.5, 1.5], [1.5, -1.5], color='black', linestyle='--')
    plt.plot([-1.5, 1.5], [-1.5, 1.5], color='r', linestyle='--')
    plt.colorbar()
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title('error - t=%.1f' % (0.2 * 5))

    plt.tight_layout()

    try : 
        plt.savefig(PATH + f'/figure/Time={Time}/prediction_figure/Time_{int(Time * 1000)}_duibi_{len(L)}.pdf', bbox_inches='tight')
    except:
        pass
    plt.clf()
    plt.close()


    anti_diag_pred = np.diag(pred)
    anti_diag_data = np.diag(data)
    main_diag_pred = np.diag(np.fliplr(pred))
    main_diag_data = np.diag(np.fliplr(data))


    plt.figure(figsize=(12, 5))


    plt.subplot(1, 2, 1)
    plt.plot(np.linspace(left_boundry,right_boundry,n),main_diag_pred, label='Pred Main Diagonal', marker='o')
    plt.plot(np.linspace(left_boundry,right_boundry,n),main_diag_data, label='True Main Diagonal', marker='x')
    plt.title('Main Diagonal Comparison')
    plt.xlabel('x')
    plt.ylabel('Value')
    plt.legend()
    plt.grid(True)


    plt.subplot(1, 2, 2)
    plt.plot(np.linspace(left_boundry,right_boundry,n),anti_diag_pred, label='Pred Anti-Diagonal', marker='o')
    plt.plot(np.linspace(left_boundry,right_boundry,n),anti_diag_data, label='True Anti-Diagonal', marker='x')
    plt.title('Anti-Diagonal Comparison')
    plt.xlabel('x')
    plt.ylabel('Value')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    try : 
        plt.savefig(PATH + f'/figure/Time={Time}/Diagonal/Time_{int(Time * 1000)}_Diagonal_{len(L)}.pdf', bbox_inches='tight')
    except:
        pass

    plt.clf()
    plt.close()

    l2_error = jnp.linalg.norm(pred - data) / jnp.linalg.norm(data)
    print("Relative L2 Error:", l2_error)
