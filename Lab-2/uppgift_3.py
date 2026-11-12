import matplotlib.pyplot as plt
import numpy as np
import scipy as sp

def main():
    
    k = 2 # konstant
    Ta = Tb = 2 # ändpunkter
    q = lambda x: 50*x**3 * np.log(x + 1) # funktion i högerled
    
    N = 100 # antal del-intevall
    
    D(N, Ta, Tb, k, q)
    E(k, Ta, Tb, q)
    F(N, k, Ta, Tb, q)
    
def diskretisering_temperatur(N, q, k, Ta, Tb):
    
    h = 1/N # beräkna steglängd
    
    v2 = (-2)*np.ones(N-1) # mitten diagonal
    v1 = np.ones(N-2) # sido diagonal
    v3 = np.ones(N-2) # sido diagonal
    A = (k/h**2) * sp.sparse.diags_array((v1, v2, v3), offsets=(-1, 0, 1), format='csr') # skapa tridiogonal matris A
    
    x_all = np.linspace(0, 1, N+1) # ta fram alla x-värden
    x_in = x_all[1:-1] # slica ut de yttersta punkterna
    
    b = -q(x_in) # beräkna högerledet b
    b[0] = b[0] - Ta*k/h**2 # överför första randvillkoret
    b[-1] = b[-1] - Tb*k/h**2 # överför andra randvillkoret
    
    return A, b

def D(N, Ta, Tb, k, q):
    
    A, b = diskretisering_temperatur(N, q, k, Ta, Tb) # ta fram A och b
    
    T = sp.sparse.linalg.spsolve(A, b) # lös för T(x) (innersta punkterna)
    T_all = np.concatenate(([Ta], T, [Tb])) # lägg till yttersta punkterna
    x_all = np.linspace(0, 1, N+1) # ta fram alla x-värden
    
    T_02 = T_all[int(round(0.2*N))] # ta fram värde på T(0.2)
    
    plt.plot(x_all, T_all, label='T(x)') # plotta T(x)
    plt.grid(True)
    plt.xlabel('x')
    plt.ylabel('T(x)')
    plt.legend()
    
    print(f'T(0.2) ≈ {T_02}')
    
def E(k, Ta, Tb, q):
    
    N_lista = [50 * (2**i) for i in range(10)] # lista med dubblerande värden (50 - 25600)
    T_lista = [] # lista av T(0.7)
    
    for N in N_lista: # för varje N
        
        A, b = diskretisering_temperatur(N, q, k, Ta, Tb) # ta fram A och b
        
        T = sp.sparse.linalg.spsolve(A, b) # lös för T(x) (innersta punkterna)
        T_all = np.concatenate(([Ta], T, [Tb])) # lägg till yttersta punkterna
        
        T_07 = T_all[int(round(0.7*N))] # ta fram värde på T(0.7)
        T_lista.append(T_07) # spara värdet
        
    T_lista = np.array(T_lista) # omvandla till numpy array
    
    fel = np.abs(T_lista[1:] - T_lista[:-1]) # beräkna array av fel
    
    p = np.log(fel[:-1] / fel[1:]) * (1 / np.log(2)) # beräkna noggranhetsordning
    
    print(f'T(0.7) = {np.round(T_lista, 7)}')
    print(f'p = {np.round(p, 4)}')
    
def F(N, k, Ta, Tb, q):
    
    plt.figure(2)
    x_all = np.linspace(0, 1, N+1) # ta fram alla x-värden
    
    # Ta = 2, Tb = 2
    A, b = diskretisering_temperatur(N, q, k, Ta, Tb)
    T = np.concatenate(([Ta], sp.sparse.linalg.spsolve(A, b), [Tb]))
    v0 = T[int(round(0.7*N))]
    
    plt.plot(x_all, T, label='$T_a = 2, T_b = 2$')
    
    # Ta = 2.2, Tb = 2
    A, b = diskretisering_temperatur(N, q, k, Ta + 0.2, Tb)
    T = np.concatenate(([Ta + 0.2], sp.sparse.linalg.spsolve(A, b), [Tb]))
    v1 = T[int(round(0.7*N))]
    
    plt.plot(x_all, T, label='$T_a = 2.2, T_b = 2$')
    
    # Ta = 2, Tb = 2.3
    A, b = diskretisering_temperatur(N, q, k, Ta, Tb + 0.3)
    T = np.concatenate(([Ta], sp.sparse.linalg.spsolve(A, b), [Tb + 0.3]))
    v2 = T[int(round(0.7*N))]
    
    plt.plot(x_all, T, label='$T_a = 2, T_b = 2.3$')
    
    osäkerhet = np.abs(v1 - v0) + np.abs(v2 - v0) # beräkna osäkerhet för T(0.7)
    
    print(f'osäkerhet = {osäkerhet}')
    
    plt.grid(True)
    plt.xlabel('x')
    plt.ylabel('T(x)')
    plt.legend()

if __name__ == '__main__':
    main()
    plt.show()