import numpy as np
import matplotlib.pyplot as plt

def main():
    
    f = lambda x, L: 8*x/(3*L) - 3*(x/L)**2 + 1/3*(x/L)**3 - 2/3*np.sin(np.pi*x/L) #funktion f(x)
    g = lambda x, L: 3*L/8*(3*(x/L)**2 - 1/3*(x/L)**3 + 2/3*np.sin(np.pi*x/L)) #funktion g(x), f(x)=0 => g(x)=x
    fp = lambda x, L: 8/(3*L) - 6*x/L**2 + x**2/L**3 - 2*np.pi/(3*L)*np.cos(np.pi*x/L) #funktion f´(x)
    
    L = 1 #längd parameter
    x0 = 0.3 #startgissning
    tol = 1e-10 #tolerans

    plotta_f(f, L)
    print('Fixpunkt')
    diffs_fix = fixpunkt(g, L, x0, tol)
    print('Newton')
    diffs_new = newton(f, fp, L, x0, tol)
    plotta_konvergens(diffs_fix, diffs_new)

def plotta_f(f, L): #plottar grafen f(x)
    
        plt.figure()
        x = np.linspace(0, L) #intervall från 0 till L
        plt.plot(x, f(x, L), label='$f(x)$') #plotta f(x)
        plt.grid(True)
        plt.xlabel('$x$')
        plt.ylabel('$y$')
        plt.legend()
        
def fixpunkt(g, L, x0, tol): #hitta f(x)=0 genom fixpunktsmetoden
    
    diff = 1 #differens
    it = 0 #iteration
    diffs = [] #lista av differenser
    
    while diff > tol:
        xnew = g(x0, L) #fixpunktsiteration
        diff = np.abs(xnew - x0) #uppdatera differens
        diffs.append(diff) #spara differens
        x0 = xnew #uppdatera x0
        it += 1 #öka iteration
        print(f"{it}, x = {xnew}, diff = {diff}")
        
    print(f"f(x)≈0 för x={xnew}")
    return diffs
    
def newton(f, fp, L, x0, tol): #hitta f(x)=0 genom newtons metod
    
    diff = 1 #differens
    it = 0 #iteration
    diffs = [] #lista av differenser
    
    while diff > tol:
        xnew = x0 - f(x0, L)/fp(x0, L) #newtons metod
        diff = np.abs(xnew - x0) #uppdatera differens
        diffs.append(diff) #spara differens
        x0 = xnew #uppdatera x0
        it += 1 #öka iteration
        print(f"{it}, x = {xnew}, diff = {diff}")
        
    print(f"f(x)≈0 för x={xnew}")
    return diffs

def plotta_konvergens(diffs_fix, diffs_new): #plotta funktionernas konvergens
    
    n_fix = range(len(diffs_fix)) #listor av iterationer (0-n)
    n_new = range(len(diffs_new)) 
    
    plt.figure()
    plt.semilogy(n_fix, diffs_fix, label='Fixpunkt') #plotta fixpunktskonvergens
    plt.semilogy(n_new, diffs_new, label='Newton') #plotta newtonskonvergens
    plt.grid(True)
    plt.xlabel('$n$')
    plt.ylabel('$diffs$')
    plt.legend()
    
    
if __name__ == '__main__':
    main()
    plt.show()