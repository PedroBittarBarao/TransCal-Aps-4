import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import funcoesTermosol as ft
from APS import *
import sys

def angulo(No1,No2):
    x_1= N[0][No1-1]
    y_1= N[1][No1-1]
    x_2= N[0][No2-1]
    y_2= N[1][No2-1]

    if x_2-x_1 == 0:
        if y_2-y_1 > 0:
            ang = 0.5*np.pi
        else:
            ang = 1.5*np.pi
    else:
        if x_2-x_1 > 0:
            ang = np.arctan((y_2-y_1)/(x_2-x_1))
        else:
            ang = np.pi + np.arctan((y_2-y_1)/(x_2-x_1))
    return ang

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python3 solve.py <input_file> <output_file>")
        exit(1)

    [nn,N,nm,Inc,nc,F,nr,R] = ft.importa(f'{sys.argv[1]}')
    # nn = numero de nos
    # N = matriz dos nos
    # nm = numero de membros
    # Inc = matriz de incidencia
    # nc = numero de cargas
    # F = vetor de carregamento
    # nr = numero de restricoes
    # R = vetor com os graus de liberdade restritos
    ks =[]
    dof_indices_list = []
    for i in range(0,nm):
        dof_indices_list.append([int(Inc[i][0]),int(Inc[i][1])])
        [x_1,y_1]= N[:,int(Inc[i][0])-1]
        [x_2,y_2]= N[:,int(Inc[i][1])-1]
        l = np.sqrt((x_2-x_1)**2+(y_2-y_1)**2)
        k = Inc[i][2] * Inc[i][3] / l
        ang = angulo(int(Inc[i][0]),int(Inc[i][1]))
        c = np.cos(ang)
        s = np.sin(ang)

        M = np.array([[c**2,c*s,-c**2,-c*s],[c*s,s**2,-c*s,-s**2],[-c**2,-c*s,c**2,c*s],[-c*s,-s**2,c*s,s**2]]) 
        ke = np.multiply(k,M)
        ks.append(ke)

    global_stiffness_matrix = np.zeros((nn*2, nn*2))

    dof_indices_list

    for i in range (len(dof_indices_list)):
        dof1 = (dof_indices_list[i][0]-1)*2
        dof2 = (dof_indices_list[i][0]-1)*2+1
        dof3 = (dof_indices_list[i][1]-1)*2
        dof4 = (dof_indices_list[i][1]-1)*2+1

        global_stiffness_matrix[np.ix_([dof1,dof2,dof3,dof4],[dof1,dof2,dof3,dof4])] += ks[i]
    
    global_stiffness_matrix_uncut = global_stiffness_matrix.copy()
    u = np.zeros((nn*2,1))
    dropped = 0
    for i in range (len(R)):
        global_stiffness_matrix = np.delete(global_stiffness_matrix, int(R[i][0])-dropped, 0)
        global_stiffness_matrix = np.delete(global_stiffness_matrix, int(R[i][0])-dropped, 1)
        u = np.delete(u, int(R[i][0]) - dropped, None)
        F = np.delete(F, int(R[i][0]) - dropped, None)
        dropped += 1
    
    result_a,max_e,i = gauss_seidel_solver(500,1e-5,global_stiffness_matrix,F)

    u_completo = np.zeros((nn*2,1))
    count = 0
    for i in range (nn*2):
        if i not in R:
            u_completo[i] = result_a[count]
            count += 1

    deform = []
    tens = []

    for el in range(nm):
        u_1 = int(Inc[el][0]-1)*2
        v_1 = int(Inc[el][0]-1)*2+1
        u_2 = int(Inc[el][1]-1)*2
        v_2 = int(Inc[el][1]-1)*2+1

        ang = angulo(int(Inc[el][0]),int(Inc[el][1]))
        c = np.cos(ang)
        s = np.sin(ang)
        [x_1,y_1]= N[:,int(Inc[el][0])-1]
        [x_2,y_2]= N[:,int(Inc[el][1])-1]
        l = np.sqrt((x_2-x_1)**2+(y_2-y_1)**2)
        e = np.dot(np.array([-c,-s,c,s]),np.array([u_completo[u_1],u_completo[v_1],u_completo[u_2],u_completo[v_2]]))
        e = np.multiply(e,1/l)
        deform.append(e[0])
        tens.append(Inc[el][2]*e[0])
    
    Reac_uncut = ((np.matmul(global_stiffness_matrix_uncut,u_completo)))
    Reac =[]
    for i in range (nn*2):
        if i in R:
            Reac.append(Reac_uncut[i][0])
    
    Fi = tens*Inc[:,3]

    ft.geraSaida(sys.argv[2],Reac,u_completo,deform,Fi,tens)

    N2 = N.copy()
    for i in range (nn):
        N2[0][i] += (u_completo[i*2][0])*1000
        N2[1][i] += (u_completo[i*2+1][0])*1000



    ft.plota_to_file(N2,Inc,f"{sys.argv[2]}")