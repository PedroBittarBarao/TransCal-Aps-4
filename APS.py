import numpy as np

def gauss_seidel_solver(max_i,tol,K,F):
    n = len(F)
    x = np.array(n*[0])
    x_new = np.zeros(n)
    for i in range(max_i):
        for j in range(n):
            x_new[j] = (F[j] - np.dot(K[j,:],x_new) + K[j,j]*x_new[j])/K[j,j]
            
        all_under = True
        for j in range(n):
            erro = abs((x_new[j]-x[j])/x_new[j])
            if erro > tol:
                all_under = False
                break
        if all_under:
            break
        x = x_new.copy()

    return x_new, erro,i

def jacobi_solve(max_i,tol,K,F):
    n = len(F)
    x = np.array(n*[0])
    x_new = np.zeros(n)
    for i in range(max_i):
        for j in range(n):
            x_new[j] = (F[j] - np.dot(K[j,:],x) + K[j,j]*x[j])/K[j,j]
            
        all_under = True
        for j in range(n):
            erro = abs((x_new[j]-x[j])/x_new[j])
            if erro > tol:
                all_under = False
                break
        if all_under:
            break
        x = x_new.copy()

    return x_new, erro,i


