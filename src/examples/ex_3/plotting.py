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

    a = 0.01372
    c= 29.97924580
    NX = 3240
    qp = 1297
    constant = 30
    T0 = 1e-2
    data_t0 = np.loadtxt(r'../data/marshak/examples/sigma{}/{:.0e}_{}_0.10'.format(constant,T0,NX))[:qp]
    data_t1 = np.loadtxt(r'../data/marshak/examples/sigma{}/{:.0e}_{}_0.20'.format(constant,T0,NX))[:qp]
    data_t2 = np.loadtxt(r'../data/marshak/examples/sigma{}/{:.0e}_{}_0.40'.format(constant,T0,NX))[:qp]
    data_t3 = np.loadtxt(r'../data/marshak/examples/sigma{}/{:.0e}_{}_0.60'.format(constant,T0,NX))[:qp]
    data_t4 = np.loadtxt(r'../data/marshak/examples/sigma{}/{:.0e}_{}_0.80'.format(constant,T0,NX))[:qp]
    data_t5 = np.loadtxt(r'../data/marshak/examples/sigma{}/{:.0e}_{}_1.00'.format(constant,T0,NX))[:qp]
    Tr_t0 = (data_t0[:,1:2]/a/c)**0.25
    Tr_t1 = (data_t1[:,1:2]/a/c)**0.25
    Tr_t2 = (data_t2[:,1:2]/a/c)**0.25
    Tr_t3 = (data_t3[:,1:2]/a/c)**0.25
    Tr_t4 = (data_t4[:,1:2]/a/c)**0.25
    Tr_t5 = (data_t5[:,1:2]/a/c)**0.25
    Te_t0 = data_t0[:,2:3]
    Te_t1 = data_t1[:,2:3]
    Te_t2 = data_t2[:,2:3]
    Te_t3 = data_t3[:,2:3]
    Te_t4 = data_t4[:,2:3]
    Te_t5 = data_t5[:,2:3]

    nx = Tr_t0.shape[0]
    xx = jnp.array(data_t0[:,0:1])
    t0 = 0.0*jnp.ones((nx,1))
    t1 = 0.1*jnp.ones((nx,1))
    t2 = 0.3*jnp.ones((nx,1))
    t3 = 0.5*jnp.ones((nx,1))
    t4 = 0.7*jnp.ones((nx,1))
    t5 = 0.9*jnp.ones((nx,1))

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
    val_t5 = Model(t5,xx,xx)
    Tr_t0_pred,Te_t0_pred,*_ = jnp.split(val_t0,val_t0.shape[-1],1)
    Tr_t1_pred,Te_t1_pred,*_ = jnp.split(val_t1,val_t0.shape[-1],1)
    Tr_t2_pred,Te_t2_pred,*_ = jnp.split(val_t2,val_t0.shape[-1],1)
    Tr_t3_pred,Te_t3_pred,*_ = jnp.split(val_t3,val_t0.shape[-1],1)
    Tr_t4_pred,Te_t4_pred,*_ = jnp.split(val_t4,val_t0.shape[-1],1)
    Tr_t5_pred,Te_t5_pred,*_ = jnp.split(val_t5,val_t0.shape[-1],1)
   
    np.set_printoptions(formatter={'float_kind':'{:.2e}'.format})
    markever = 50
    plt.figure(figsize=(14, 6))
    ax1 = plt.subplot(1, 2, 1)

    pred = np.concatenate([Tr_t1_pred,Tr_t2_pred,Tr_t3_pred,Tr_t4_pred,Tr_t5_pred],-1)
    star = np.concatenate([Tr_t1,Tr_t2,Tr_t3,Tr_t4,Tr_t5],-1)
    plt.plot(xx, star, 'k', label=['Ref']+['_']*4)
    plt.plot(xx, pred, 'r', label=['Pred']+['_']*4,linewidth = 0.8)

    plt.xlabel(r"$x$",fontsize=20)
    plt.ylabel(r"$Tr$",fontsize=20)
    plt.grid()
    plt.legend(fontsize=10)
    ax=plt.gca()
    ax.spines['bottom'].set_linewidth(2)
    ax.spines['left'].set_linewidth(2)
    ax.spines['right'].set_linewidth(2)
    ax.spines['top'].set_linewidth(2)

    ax2 = plt.subplot(1, 2, 2)

    pred = np.concatenate([Te_t1_pred,Te_t2_pred,Te_t3_pred,Te_t4_pred,Te_t5_pred],-1)
    star = np.concatenate([Te_t1,Te_t2,Te_t3,Te_t4,Te_t5],-1)
    plt.plot(xx, star, 'k', label=['Ref']+['_']*4)
    plt.plot(xx, pred, 'r', label=['Pred']+['_']*4,linewidth = 0.8)

    plt.xlabel(r"$x$",fontsize=20)
    plt.ylabel(r"$Tm$",fontsize=20)
    plt.grid()
    plt.legend(fontsize=10)


    plt.subplots_adjust(left=0.19,right=0.94, bottom=0.13)


    ax=plt.gca()
    ax.spines['bottom'].set_linewidth(2)
    ax.spines['left'].set_linewidth(2)
    ax.spines['right'].set_linewidth(2)
    ax.spines['top'].set_linewidth(2)
    try : 
        plt.savefig(PATH + f'/figure/Time={Time}/prediction_figure/Time_{int(Time * 1000)}_{len(L)}.pdf', bbox_inches='tight')
    except:
        pass

    plt.close()
    pred = np.concatenate([Te_t0_pred,Te_t1_pred,Te_t2_pred,Te_t3_pred,Te_t4_pred,Te_t5_pred],-1)
    star = np.concatenate([Te_t0,Te_t1,Te_t2,Te_t3,Te_t4,Te_t5],-1)
    error = np.abs(pred-star)
    print("error_Te")
    print(np.sqrt(np.mean(np.square(error),axis=0))/np.sqrt(np.mean(np.square(star),axis=0)))
    pred = np.concatenate([Tr_t0_pred,Tr_t1_pred,Tr_t2_pred,Tr_t3_pred,Tr_t4_pred,Tr_t5_pred],-1)
    star = np.concatenate([Tr_t0,Tr_t1,Tr_t2,Tr_t3,Tr_t4,Tr_t5],-1)
    error = np.abs(pred-star)
    print("error_Tr")
    print(np.sqrt(np.mean(np.square(error),axis=0))/np.sqrt(np.mean(np.square(star),axis=0)))