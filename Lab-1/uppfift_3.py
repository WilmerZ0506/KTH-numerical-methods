import numpy as np
import scipy as sp
import matplotlib.pyplot as plt

def main():
    
    f_a = lambda x: x**3 * np.exp(x) # funktion för uppgift A och B
    
    t = np.array([2014, 2015, 2016, 2017, 2018, 2019, 2020, 2021, 2022]) # array av t-värden
    ft = np.array([12.00, 15.10, 19.01, 23.92, 30.11, 37.90, 47.70, 60.03, 75.56]) # array
    
    A(f_a)
    B(f_a)
    C(ft)
    Th, Th2 = D(ft)
    E(ft, Th, Th2)
    a, b = F(t, ft)
    G(t, ft, a, b)

def trapets(f, n, interval):

    a, b = interval # ta ut ändpunkter
    h = (b - a)/n # beräkna steg längd
    x = np.linspace(a, b, n + 1) # x-värden (n+1 punkter ger n delintervall)
    fx = f(x) # y-värden
    Th = h * (fx[0]/2 + np.sum(fx[1:-1]) + fx[-1]/2) # trapetsregeln
    return Th

def A(f_a):
    
    Th = trapets(f_a, 200, [0, 2]) #trapets av f med 200 punkter mellan 0 och 2
    
    print(f'\nA: Th = {Th}')

def B(f_a):
    
    I = sp.integrate.quad(f_a, 0, 2, epsabs=1e-13, epsrel=1e-13)[0] # jämförelse integral (typ exakt)
    
    n_vektor = [5, 10, 20, 40, 80, 160, 320, 640, 1280, 2560, 5120] # lista av punkter, dubblas varje steg, n, 2n, 4n, ... => h, h/2, h/4, ...
    fel = np.array([]) # array av fel

    for n in n_vektor: # för vaje värde av n:
        Th = trapets(f_a, n, [0, 2]) # beräkna med trapets
        eh = abs(Th - I) # beräkna felet
        fel = np.append(fel, eh) # lägg till felet

    kvot = fel[:-1] / fel[1:] # fel kvoter n / n+1 (eh / eh/2 = 2^p, p = noggranhetsordning)
    
    p = np.log2(kvot) # beräkna p
    
    print('\nB: Noggranhetsordning för trapets metod:')
    for i in p:
        print(i)

def C(ft):
    Th = ft[0]/2 + np.sum(ft[1:-1]) + ft[-1]/2 # uppskatta med trapets metod för h = 1
    print(f'\nC: h = 1 => Th = {Th}')

def D(ft):

    trapets = lambda f, h: h * (f[0]/2 + np.sum(f[1:-1]) + f[-1]/2) # lambda funktion för trapets metoden
    
    Th = trapets(ft, 1) # h = 1
    T2h = trapets(ft[::2], 2) # h = 2, ft[::2] = varannat tal i ft
    T4h = trapets(ft[::4], 4) # h = 4, ft[::4] = var fjärde tal i ft
    T8h = trapets(ft[::8], 8) # h = 8
    
    eh = np.abs(Th - T2h) # eh/2 = Th/2 - Th => eh = Th - T2h (fel med steglängd h = uppskattning med steglängd h - uppskattning med dubbla steglängden)
    e2h = np.abs(T2h - T4h)
    e4h = np.abs(T4h - T8h)
    
    kvot_1 = e2h / eh # fel kvot (eh / eh/2 = e2h / eh = 2^p)
    kvot_2 = e4h / e2h
    
    p1 = np.log2(kvot_1) # beräkna noggranhetsorning
    p2 = np.log2(kvot_2)
    
    print(f'\nD: Noggranhetsordning (p1, p2): p1 = {p1}, p2 = {p2}')
    
    return Th, T2h #returnera mest noggranna skattningar av trapets regeln

def E(ft, Th, T2h):

    R = (4*Th - T2h) / 3 # funktion av Richardsonextrapolation för trapets regeln (Rh/2 = (2^p*Vh/2 - Vh) / 2^p -1)

    simpson = lambda h: (h/3) * (ft[0] + 4*np.sum(ft[1:-1:2]) + 2*np.sum(ft[2:-1:2]) + ft[-1]) # funktion för simpsons metod
    
    print(f'\nE: Richardonextrapolation = {R}, Simpson (h = 1) = {simpson(1)}') # h = 1 används för båda fallen

def F(t, ft):
    
    A = np.column_stack((t**0, t - 2014)) # överbestämt system av den linjära funktionen ln(a) + b(t - 214) = ln(f(t))
    c = np.linalg.lstsq(A, np.log(ft))[0] # lös ut konstanter ln(a) och b
    
    atilde = c[0] # ln(a)
    b = c[1] # b
    a = np.exp(atilde) # lös ut a
    
    print(f'\nF: a = {a}, b = {b}')
    
    return a, b # returnera konstanter

def G(t, ft, a, b):
    
    f = lambda t: a * np.exp(b * (t - 2014)) # definera icke-linjära funktionen med konstanterna från uppgift F
    
    ft_2023 = f(2023) # räkna ut approximerat värde för 2023
    
    t = np.append(t, 2023) # lägg till år 2023 i t
    ft = np.append(ft, ft_2023) # lägg till värdet för 2024 i f(t)
    
    Th = ft[0]/2 + np.sum(ft[1:-1]) + ft[-1]/2 # approximera med trapets metoden (h = 1)
    
    print(f'\nG: Effekt under 2023 = {ft_2023} kW, Total mängd energi under 2014-2023 = {Th} kWår')

if __name__ == '__main__':
    main() 
    plt.show()