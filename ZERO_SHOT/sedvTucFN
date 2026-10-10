import numpy as np, scipy.io as sio
from scipy.stats import spearmanr
D='benchmark'
fp=sio.loadmat(f'{D}/Freq_Phase.mat'); f=fp['freqs'].ravel(); ph=fp['phases'].ravel()
t=np.arange(1250)/250; E=np.exp(-2j*np.pi*np.outer(f,t))
taus=np.arange(0,0.4001,0.002); K=np.exp(2j*np.pi*np.outer(taus,f))
acc=[0.762,0.796,0.967,0.654,0.767,0.842,0.854,0.683,0.700,0.850,0.158,0.467,0.412,0.042,0.717,0.708,0.725,0.713,0.433,0.771,0.767,0.892,0.263,0.883,0.696,0.746,0.733,0.946,0.375,0.596,0.871,0.988,0.329,0.896,0.792]
out=[]
for s in range(1,36):
    x=sio.loadmat(f'{D}/S{s}.mat')['data'][61,125:1375]
    x=x-x.mean(0)
    z=np.einsum('kt,tkb->kb',E,x)*np.exp(-1j*ph)[:,None]
    zm=z.mean(1); sc=np.abs(K@zm)/np.abs(zm).sum()
    cons=np.abs(np.exp(1j*np.angle(z)).mean(1)).mean()
    out.append((s,taus[sc.argmax()]*1e3,sc.max(),cons))
    print('S%02d tau=%3.0f ms  R=%.2f  block-consistency=%.2f  CNN_ref acc=%.3f'%(*out[-1],acc[s-1]))
tau=np.array([o[1] for o in out]); print('median tau',np.median(tau))
print('Spearman(|tau-median|, acc):',spearmanr(np.abs(tau-np.median(tau)),acc))
