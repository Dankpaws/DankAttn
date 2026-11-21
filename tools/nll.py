import sys, numpy as np, torch
V=248320
ids=torch.from_numpy(np.fromfile(sys.argv[1],dtype=np.uint32).astype(np.int64))
for f in sys.argv[2:]:
    l=torch.from_numpy(np.fromfile(f,dtype=np.float16).astype(np.float32)).view(-1,V)[:len(ids)]
    lp=torch.log_softmax(l.double(),-1)
    print(f.split('/')[-1], round(float(-lp[:-1].gather(1,ids[1:,None]).mean()),4))
