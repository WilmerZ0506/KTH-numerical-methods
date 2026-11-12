import numpy as np
import matplotlib.pyplot as plt

def main():
    
    f = lambda t, y: 1 + t - y # diffekvations (DE) högerled
    exakt_lösning = lambda t: np.exp(-t) +t # exakta lösningen till DE
    
    y0 = 1 # start värde y
    t0 = 0 # start värde t
    h = np.array([0.2, 0.1, 0.05, 0.025, 0.0125]) # lista av halverade steglängder
    T = 1.2 # slut värde t
    
    yh_T = np.zeros(h.size) # lista av sparade yh(T) för varje h
    eh = np.zeros(h.size) # lista av fel för yh(T)
    
    A(t0, y0, T, h, yh_T, f)
    B(T, yh_T, eh, exakt_lösning)
    C(eh)

def A(t0, y0, T, h, yh_T, f):
    
    for i in range(len(h)): # för varje h
        
        N = round((T - t0) / h[i]) # beräkna antal punkter
        t = np.zeros(N+1) # lista av t-värden
        y = np.zeros(N+1) # lista av y-värden
    
        t[0] = t0 # lägg in t-värde
        y[0] = y0 # lägg in y-värde
        
        for j in range(N):
            t[j+1] = t[j] + h[i] # lägg till nästa t-värde
            y[j+1] = y[j] + h[i] * f(t[j], y[j]) # lägg till nästa y-värde (euler-först metoden)
            
        yh_T[i] = y[-1] # spara yh(T)
    
def B(T, yh_T, eh, exakt_lösning):
    
    for i in range(len(yh_T)): # för varje yh(T)
        
        eh[i] = np.abs(yh_T[i] - exakt_lösning(T)) # spara felet
        
    print(f'eh = {eh}')
        
def C(eh):
        
    p = np.log(eh[:-1] / eh[1:]) * (1 / np.log(2)) # beräkna noggranhetsordningen
    
    print(f'\np = {p}')

if __name__ == '__main__':
    main()
    plt.show()