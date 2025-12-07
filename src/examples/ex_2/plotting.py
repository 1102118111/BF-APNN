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
    data_t0 = np.loadtxt('../data/result_eps1e-3/result3240_0.10')
    data_t1 = np.loadtxt('../data/result_eps1e-3/result3240_0.20')
    data_t2 = np.loadtxt('../data/result_eps1e-3/result3240_0.30')
    data_t3 = np.loadtxt('../data/result_eps1e-3/result3240_0.40')
    data_t4 = np.loadtxt('../data/result_eps1e-3/result3240_0.50')
    idx= 1
    Tr_t0 = data_t0[::idx,1:2]**0.25
    Tr_t1 = data_t1[::idx,1:2]**0.25
    Tr_t2 = data_t2[::idx,1:2]**0.25
    Tr_t3 = data_t3[::idx,1:2]**0.25
    Tr_t4 = data_t4[::idx,1:2]**0.25
    Te_x0 = []
    for i in range(50):
        x,y,z = np.split(np.loadtxt('../data/result_eps1e-3/result3240_%.2f'%(0.01*(i+1))),3,1)
        Te_x0.append(z[3][0])
    Te_x0 = np.array(Te_x0).reshape(-1,1)
    # Te_x0 = data_x0[:,1:2]
    xx = jnp.array(data_t1[::idx,0:1])
    tt = jnp.arange(0.01,0.51,0.01).reshape(-1,1)
    nx = xx.shape[0]
    nt = tt.shape[0]
    t0 = 0.1*jnp.ones(nx,).reshape(-1,1)
    t1 = 0.2*jnp.ones(nx,).reshape(-1,1)
    t2= 0.3*jnp.ones(nx,).reshape(-1,1)
    t3= 0.4*jnp.ones(nx,).reshape(-1,1)
    t4= 0.5*jnp.ones(nx,).reshape(-1,1)
    x0 = 0.0025*jnp.ones(nt,).reshape(-1,1)


    def Model(t,x,theta):
        t = t.squeeze()
        x = x.squeeze()
        theta = theta.squeeze()
        model_vmap = jax.vmap(model)
        return model_vmap(t,x,theta)

    val_t0 = Model(t0,xx,xx)
    val_t1 = Model(t1,xx,xx)
    val_t2 = Model(t2,xx,xx)
    val_t3 = Model(t3,xx,xx)
    val_t4 = Model(t4,xx,xx)
    val_t5 = Model(tt,x0,x0)
    Tr_t0_pred,Te_t0_pred,*_ = jnp.split(val_t0,val_t0.shape[-1],1)
    Tr_t1_pred,Te_t1_pred,*_ = jnp.split(val_t1,val_t0.shape[-1],1)
    Tr_t2_pred,Te_t2_pred,*_ = jnp.split(val_t2,val_t0.shape[-1],1)
    Tr_t3_pred,Te_t3_pred,*_ = jnp.split(val_t3,val_t0.shape[-1],1)
    Tr_t4_pred,Te_t4_pred,*_ = jnp.split(val_t4,val_t0.shape[-1],1)
    Tr_x0_pred,Te_x0_pred,*_ = jnp.split(val_t5,val_t0.shape[-1],1)
   
    plt.figure(figsize=(15, 6))
    plt.subplot(1, 2, 1)
    plt.plot(tt,Te_x0_pred, "c+", markersize=12, markerfacecolor='#725E79', markevery=2, linewidth=2, label=r"$Te_{nn}$")
    plt.plot(tt,Te_x0,color='c',label=r"$Te_{ref}$")

    plt.xlabel(r"$t$",fontsize=20)
    plt.ylabel(r"$Te$",fontsize=20)
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)
    plt.grid()
    plt.legend(fontsize=26)

    ax=plt.gca()
    ax.spines['bottom'].set_linewidth(2)
    ax.spines['left'].set_linewidth(2)
    ax.spines['right'].set_linewidth(2)
    ax.spines['top'].set_linewidth(2)
    plt.text(0.5, -0.25, '(a)', transform=plt.gca().transAxes, fontsize=fontsize, fontweight='bold', ha='center')
    
    plt.subplot(1, 2, 2)
    plt.plot(xx, Tr_t0_pred, "b1", markersize=8, markevery=100, label=r"$Tr_{nn}(t=0.1)$")
    plt.plot(xx, Tr_t0, linestyle = '--', color='b', markerfacecolor=None, linewidth=2, label=r"$Tr_{ref}(t=0.1)$")

    plt.plot(xx, Tr_t2_pred, "rx", markersize=8, markevery=100, label=r"$Tr_{nn}(t=0.3)$")
    plt.plot(xx, Tr_t2, linestyle = '-.', color='r', markerfacecolor=None, linewidth=2, label=r"$Tr_{ref}(t=0.3)$")

    plt.plot(xx, Tr_t4_pred, "k+", markersize=8, markevery=100, label=r"$Tr_{nn}(t=0.5)$")
    plt.plot(xx, Tr_t4, linestyle = '-', color='k', markerfacecolor=None, linewidth=2, label=r"$Tr_{ref}(t=0.5)$")

    plt.xlabel(r"$x$",fontsize=20)
    plt.ylabel(r"$Tr$",fontsize=20)
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)
    plt.grid()
    plt.legend(fontsize=16)


    plt.subplots_adjust(left=0.19,right=0.94, bottom=0.13)


    ax=plt.gca()
    ax.spines['bottom'].set_linewidth(2)
    ax.spines['left'].set_linewidth(2)
    ax.spines['right'].set_linewidth(2)
    ax.spines['top'].set_linewidth(2)
    plt.text(0.5, -0.25, '(b)', transform=plt.gca().transAxes, fontsize=fontsize, fontweight='bold', ha='center')
    try : 
        plt.savefig(PATH + f'/figure/Time={Time}/prediction_figure/Time_{int(Time * 1000)}_{len(L)}.pdf', bbox_inches='tight')
    except:
        pass

    plt.close()
    pred = np.concatenate([Tr_t0_pred,Tr_t1_pred,Tr_t2_pred,Tr_t3_pred,Tr_t4_pred],-1)
    star = np.concatenate([Tr_t0,Tr_t1,Tr_t2,Tr_t3,Tr_t4],-1)
    error = np.abs(pred-star)
    error_Te_x0 = np.linalg.norm(Te_x0_pred-Te_x0,2)/np.linalg.norm(Te_x0,2)
    np.set_printoptions(formatter={'float_kind':'{:.2e}'.format})
    print((np.sqrt(np.mean(np.square(error),axis=0))/np.sqrt(np.mean(np.square(star),axis=0))))
    print('error_Te:%.2e'%(error_Te_x0 ))