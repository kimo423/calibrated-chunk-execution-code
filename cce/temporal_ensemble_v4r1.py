"""ACT-style oldest-first weights; pose-aware averaging for absolute EE chunks."""
import numpy as np

class TemporalEnsemble:
    def __init__(self,m=.1):
        if m<0 or not np.isfinite(m):raise ValueError('invalid decay')
        self.m=m;self.chunks=[]

    def add(self,start,poses,grips):
        poses=np.asarray(poses,float);grips=np.asarray(grips,float)
        if poses.shape!=(30,7) or grips.shape!=(30,) or not np.isfinite(poses).all() or not np.isfinite(grips).all():raise ValueError('expected finite 30-step chunk')
        if self.chunks and start<=self.chunks[-1][0]:raise ValueError('chunk starts must increase')
        if np.any(np.linalg.norm(poses[:,3:],axis=1)<1e-9):raise ValueError('invalid quaternion')
        self.chunks.append((start,poses.copy(),grips.copy()))

    def target(self,t):
        self.chunks=[x for x in self.chunks if t<x[0]+30]
        active=[x for x in self.chunks if x[0]<=t]
        if not active:raise ValueError('no active prediction')
        weights=np.exp(-self.m*np.arange(len(active)));weights/=weights.sum()
        ps=np.array([p[t-s] for s,p,g in active]);gs=np.array([g[t-s] for s,p,g in active])
        if len(active)==1:return ps[0].copy(),float(gs[0])
        # Markley quaternion mean is invariant to the sign of each quaternion.
        q=ps[:,3:]/np.linalg.norm(ps[:,3:],axis=1)[:,None]
        matrix=np.einsum('i,ij,ik->jk',weights,q,q)
        values,vectors=np.linalg.eigh(matrix)
        if values[-1]-values[-2]<1e-12:
            qm=q[0].copy()  # prespecified ambiguous-orientation fallback
        else:qm=vectors[:,-1]
        if np.dot(qm,q[0])<0:qm=-qm
        pose=np.r_[weights@ps[:,:3],qm]
        grip=1. if float(weights@gs)>0 else -1.
        return pose,grip

def selftest():
    a=np.tile([0.,0.,0.,1.,0.,0.,0.],(30,1));g=np.full(30,-1.)
    te=TemporalEnsemble();te.add(0,a,g)
    assert np.array_equal(te.target(0)[0],a[0]) # zeros are valid predictions
    b=a.copy();b[:,0]=1;b[:,3:]*=-1
    te.add(10,b,-g);p,gr=te.target(10)
    expected=np.exp(-.1)/(1+np.exp(-.1))
    assert abs(p[0]-expected)<1e-12 and gr==-1 and np.allclose(p[3:],[1,0,0,0])
    c=a.copy();c[:,0]=2;te.add(20,c,g)
    assert len(te.chunks)==3
    p,gr=te.target(30);assert len(te.chunks)==2 and p[0]>1
    try:te.target(60)
    except ValueError:pass
    else:raise AssertionError('stale chunk used')
    print('TE checks passed: identity, zero target, overlap weights, quaternion signs, gripper, expiry')

if __name__=='__main__':selftest()
