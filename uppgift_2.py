import numpy as np
import matplotlib.pyplot as plt

def main():
    y = np.loadtxt('Lab_1/KPI.csv', delimiter=',', skiprows=3, usecols=1)
    t = np.arange(y.size) / 12
    
    N = 360 # antal punkter
    
    setup_plots(t, y) # skapa figurer
    
    I = (t+1980 >= 1991) & (t+1980 < 2021) # ändra intervallet till 1991 - 2020
    t = t[I] # updatera t
    y = y[I] # updatera y
    
    c_a, f_a = A(t, y, N) 
    d_b, f_b = B(t, y, N)
    c_c, f_c = C(t, y, N, d_b)
    D(f_a, f_b, f_c, c_a, d_b, c_c)
    
    finalize_plots()
    
def RMS(N, y, funktion): # räkna ut RMS
    return np.sqrt(1/N*np.sum((funktion - y)**2)) # Detsamma som  roten ur (felkvadratsumman / N) eller residualvectorns norm / roten ur N
    
def A(t, y, N):
    
    V_a = np.column_stack((t**0, t**1)) # överbestämt system på linjär funktion
    c = np.linalg.lstsq(V_a, y)[0] # beräkna parametrar
    
    f_a = lambda t, c: c[0] + c[1]*t # anpassa linjär funktion
    
    print(f'\nc0 = {c[0]}, c1 = {c[1]}') # skriv parametrar
    RMS_a = RMS(N, y, f_a(t, c)) # beräkna RMS för linjär funktion
    print(f'RMS_A = {RMS_a}') #skriv RMS
    
    plt.figure(1)
    plt.plot(t+1980, f_a(t, c), label='$f(t)_A$') # plotta linjär funktion
    plt.legend()

    plt.figure(2)
    r_a = lambda t, c: f_a(t, c) - y # anpassa funktion för linjära ekvationens mätningsfel
    plt.plot(t+1980, r_a(t, c), label='$Modelfel_A$') # plotta mätningsfel
    
    return c, f_a
    
def B(t, y, N):
    
    V_b = np.column_stack((t**0, t**1, np.sin((2*np.pi*t)/8), np.cos((2*np.pi*t)/8))) # överbestämt system på periodisk funktion
    d =  np.linalg.lstsq(V_b, y)[0] # beräkna parametrar
    
    f_b = lambda t, d: d[0] + d[1]*t**1 + d[2]*np.sin((2*np.pi*t)/8) + d[3]*np.cos((2*np.pi*t)/8) # anpassa periodisk funktion
    
    print(f'\nd0 = {d[0]}, d1 = {d[1]}, d2 = {d[2]}, d3 = {d[3]}') # skriv parametrar
    RMS_b = RMS(N, y, f_b(t, d)) # beräkna RMS för periodisk funktion
    print(f'RMS_B = {RMS_b}') # plotta RMS
    
    plt.figure(1)
    plt.plot(t+1980, f_b(t, d), label='$f(t)_B$') # plotta periodisk funktion
    plt.legend()
    
    plt.figure(3)
    r_b = lambda t, d: f_b(t, d) - y # anpassa funktion för periodiska funktionens mätningsfel
    plt.plot(t+1980, r_b(t, d), label='$Modelfel_B$') # plotta mätningsfel
    
    return d, f_b
    
def C(t, y, N, d):
    
    c = np.array([d[0], d[1], d[2], d[3], 8.0]) # startgissning från B
    
    f_c = lambda t, c: c[0] + c[1]*t**1 + c[2]*np.sin((2*np.pi*t)/c[4]) + c[3]*np.cos((2*np.pi*t)/c[4]) # anpassa periodisk (icke-linjär) funktion
    F = lambda t, c: f_c(t, c) - y # residual vektor, ska approximera F(c)≈0
    J = lambda t, c: np.column_stack(( # jakobian till f
        t**0,
        t**1,
        np.sin((2*np.pi*t)/c[4]),
        np.cos((2*np.pi*t)/c[4]),
        ((2*np.pi*t)/(c[4]**2))*(-c[2]*np.cos((2*np.pi*t)/c[4]) + c[3]*np.sin((2*np.pi*t)/c[4]))))
    
    korrektion = 1 # initiera differens
    tol = 1e-10 # tolerans
    
    # Gaus-Newton metoden
    while korrektion > tol:
        delta = np.linalg.lstsq(J(t, c), -F(t, c))[0] # beräkna korrektionsvektor
        c_new = c + delta # justera parametrar
        korrektion = np.linalg.norm(delta) # updatera korrektion
        c = c_new # updatera parametrar
        
    print(f'\nd0 = {c[0]}, d1 = {c[1]}, d2 = {c[2]}, d3 = {c[3]}, L = {c[4]}') # skriv parametrar
    RMS_c = RMS(N, y, f_c(t, c)) # beräkna rms för periodisk icke-linjär funktion
    print(f'RMS_C = {RMS_c}') # skriv RMS
        
    plt.figure(1)
    plt.plot(t+1980, f_c(t, c), label='$f(t)_C$') # plotta periodisk icke-linjär funktion
    
    plt.figure(4)
    r_c = lambda t, c: f_c(t, c)-y # anpassa funktion för periodisk icke-linjära funktionens mätningsfel
    plt.plot(t+1980, r_c(t, c), label='$Modelfel_C$') # plotta mätningsfel
    
    return c, f_c
    
def D(f_a, f_b, f_c, c_a, d_b, c_c): # srkiv ut alla modellernas genomsnittliga ökning
    
    t_start = 1991 - 1980
    t_slut = 2020 - 1980
    antal_ar = 2020 - 1991
    
    stigning_A = f_a(t_slut, c_a) - f_a(t_start, c_a)
    stigning_B = f_b(t_slut, d_b) - f_b(t_start, d_b)
    stigning_C = f_c(t_slut, c_c) - f_c(t_start, c_c)
    
    print('\nGenomsnittlig ökning')
    print(f'Model A: {stigning_A/antal_ar} KPI/år')
    print(f'Model B: {stigning_B/antal_ar} KPI/år')
    print(f'Model C: {stigning_C/antal_ar} KPI/år')
        
def setup_plots(t, y):  # initierar alla fyra figurer med titlar, axelnamn och rutnät.

    plt.figure(1)
    plt.plot(t+1980, y, label='$KPI$ (Mätdata)')
    plt.title('KPI och Kurvanpassningsmodeller (1991–2020)')
    plt.xlabel('År ($t$)')
    plt.ylabel('KPI') 
    plt.grid(True)
    
    titles = [
        (2, 'Modellfel A (Rät linje)', '$f(t)_A - y$'),
        (3, 'Modellfel B (Fast period L=8)', '$f(t)_B - y$'),
        (4, 'Modellfel C (Optimerad period L)', '$f(t)_C - y$')
    ]
    
    for fig_id, title, ylabel in titles:
        plt.figure(fig_id)
        plt.title(title)
        plt.xlabel('År ($t$)')
        plt.ylabel(ylabel)
        plt.grid(True)

def finalize_plots(): #aktiverar legend för alla öppna figurer
    
    for fig_id in range(1, 5):
        plt.figure(fig_id)
        plt.legend()

if __name__ == '__main__':
    main()
    plt.show()