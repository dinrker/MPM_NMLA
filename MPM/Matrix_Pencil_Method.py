import numpy as np

def NtoL(N):
    return int(np.ceil(( np.ceil(N/3.0) + np.floor(N/2.0) )/2))


## Matrix Pencil Method
def Matrix_Pencil(y,tol=1e-3,L=0):
    sh = y.shape
    N = sh[0]
    if sh[1]>1:
        print 'Error: y should be a column vector'
    if tol<0:
        print 'Error: tol should be positive'
    elif tol>=1:
        tol = int(tol)
    if L==0:
        L = NtoL(N)
    
    Y = np.zeros((L+1,N-L),dtype=np.complex_) 
    for j in range(L+1):
        Y[j] = y[j:N-L+j].T
    Y = Y.T
    U, s, V = np.linalg.svd(Y)
    if tol>=1:
        M = tol
    else:
        for m in range(len(s)):
            if abs(s[m+1])/s[0] <= tol:
                M = m+1
                break
    
    V = V.T
    S = np.diag(s)
    SM = S[:,:M]
    VM = V[:,:M]
    V1 = VM[:L,:]
    V2 = VM[1:L+1,:]

    Y1 = np.dot( np.dot(U,SM),V1.T )
    Y2 = np.dot( np.dot(U,SM),V2.T )
    A = np.dot( np.linalg.pinv(Y1),Y2 )
    ev = np.linalg.eigvals(A)
    z = np.sort(ev)
    z = sorted(z,reverse=True)
    return z[:M]