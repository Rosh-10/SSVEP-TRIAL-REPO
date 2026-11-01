import subprocess,sys,json,time
from pathlib import Path
Z=Path(__file__).resolve().parent; N=Z/'results'; R=Path('/mnt/f/ssvep_backup/results'); S='--sl 1.0 --subjects 1 '
def g(s,t,*v): return (s,S+'--variant '+' '.join(v),[f'{t}_{x}_s01_1p0' for x in v])
J=[('zs_runner.py',S+'--methods fbcca fbttcca_yf',['fold_s01_fbcca_1p0','fold_s01_fbttcca_yf_1p0']),('zs_cnn_ref.py',S,['cnn_ref_s01_1p0']),
 g('zs_gnn_v3b_freq2.py','gnnv3b3','fg2'),g('zs_gnn_v3c_freq3.py','gnnv3c','ng','h5','ngh5'),g('zs_gnn_v3d_unif.py','gnnv3d','fgu'),
 g('zs_gnn_v3e_contrast.py','gnnv3e','cm','lr'),g('zs_gnn_v3g_graph.py','gnnv3g','knn9','knn19','dense19'),g('zs_gnn_v3h_anat.py','gnnv3h','fga'),
 g('zs_gnn_v3i_cap.py','gnnv3i','wide19'),g('zs_gnn_v3j_adapt.py','gnnv3j','aw9','aw19'),g('zs_gnn_v3k_seed.py','gnnv3k','s42d19','s43d19'),
 g('zs_gnn_v3l_bank5.py','gnnv3l','b5d9','b5d19'),g('zs_gnn_v3m_bank5seed.py','gnnv3m','s42b5d19','s43b5d19')]
assert R.exists(); t0=time.time(); out=[]
for s,a,st in J:
    print('>>',s,a,flush=True); rc=subprocess.run([sys.executable,s,*a.split()],cwd=Z).returncode
    for x in st:
        f=N/f'{x}.json'; r=R/f'{x}.json'
        try:
            n,o=json.load(open(f)),json.load(open(r)); new=f.stat().st_mtime>t0
            ok='PASS' if new and n['acc']==o['acc'] and n['preds']==o['preds'] and n.get('n_params')==o.get('n_params') else ('DIFF' if new else 'NOT RERUN')
            out.append((x,rc,ok,o['acc'],n['acc']))
        except Exception as e: out.append((x,rc,'ERR '+type(e).__name__,0,0))
print('\n%-34s rc  result     ref    new'%'file')
for x,rc,ok,a,b in out: print('%-34s %-3d %-10s %.4f %.4f'%(x,rc,ok,a,b))
print('minutes',round((time.time()-t0)/60,1),'| PASS',sum(o[2]=='PASS' for o in out),'of',len(out))
