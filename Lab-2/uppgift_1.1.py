import numpy as np
import matplotlib.pyplot as plt

def main():
    
    f = lambda t, y: 1 + t - y # diffekvations (DE) högerled
    exakt_lösning = lambda t: np.exp(-t) +t # exakta lösningen till DE
    
    y0 = 1 # start värde y
    t0 = 0 # start värde t
    h = 0.1 # steglängd
    T = 1.2 # slut värde t
    N = round((T - t0) / h) # antal punkter
    
    t = np.zeros(N+1) # lista av t-värden
    y = np.zeros(N+1) # lista av y-värden
    
    t[0] = t0 # lägg in t-värde
    y[0] = y0 # lägg in y-värde
    
    A(t, y, h, N, f, exakt_lösning)
    B(T, y, exakt_lösning)

def A(t, y, h, N, f, exakt_lösning):
    
    for j in range(N):
        t[j+1] = t[j] + h # lägg till nästa t-värde
        y[j+1] = y[j] + h * f(t[j], y[j]) # lägg till nästa y-värde (euler-först metoden)
    
    plt.figure()
    plt.plot(t, y, label='$yh(t)$') # plotta approximering
    plt.plot(t, exakt_lösning(t), label='$y(t)$') # plotta exakta
    plt.xlabel('$t$')
    plt.ylabel('$y(t)$')
    plt.grid(True)
    plt.legend()
    
def B(T, y, exakt_lösning):
    
    eh = np.abs(y[-1] - exakt_lösning(T)) # beräkna felet vid t = T
    
    print(f'eh = {eh}')

if __name__ == '__main__':
    main()
    plt.show()