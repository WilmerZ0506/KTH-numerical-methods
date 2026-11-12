import matplotlib.pyplot as plt
import numpy as np
import scipy as sp

def main():
    q0, i0 = 1, 0 # start värden på q och i
    C, L = 0.5, 2 # parametrar C, L
    R_dämpad, R_odämpad = 1, 0 # parametrar R för dämpad och odämpad variant
    t_span = (0, 20) # t intervall
    y0 = np.array([q0, i0]) # star vektor med start värden

    N_list = [20, 40, 80, 160] # lista med olika antal punkter
    N_studie = [80, 160, 320, 640, 1280] # lista med olika antal punkter (20, 40 ger instabila lösningar)

    sol_i, sol_ii = solve_scipy(t_span, y0, C, L, R_dämpad, R_odämpad) # lös ut systemet för både dämpat och odämpat

    plot_framat_euler(C, L, R_dämpad, t_span, N_list, y0, sol_i) # plotta de olika lösningarna för framåt euler
    konvergens_studie(t_span, y0, C, L, R_dämpad, sol_i, N_studie) # konvergens studie

def F(t, y, R, L, C):
    q, i = y # ta ut q och i
    dq_dt = i # q´ = i
    di_dt = -(1 / (C * L)) * q - (R / L) * i # i´ = -(1/CL)q - (R/L)i
    return np.array([dq_dt, di_dt]) # returnera F(t, y)

def solve_scipy(t_span, y0, C, L, R_dämpad, R_odämpad):

    sol_i = sp.integrate.solve_ivp(F, t_span, y0, args=(R_dämpad, L, C)) # beräkna system för dämpad
    sol_ii = sp.integrate.solve_ivp(F, t_span, y0, args=(R_odämpad, L, C)) # beräkna system för odämpad
    return sol_i, sol_ii # returnera lösningar

def euler_forward(t_span, y0, N, R, L, C): # funktion för euler framåt metoden
    t0, T = t_span # ta ut start och slut värde på t
    h = (T - t0) / N # beräkna steglängden h

    t = np.linspace(t0, T, N + 1) # skapa array av t med N punkter
    y = np.zeros((y0.size, N + 1)) # skapa matris för y värden med N punkter
    y[:, 0] = y0 # lägg in start värdet

    for j in range(N): # för varje N
        y[:, j + 1] = y[:, j] + h * F(t[j], y[:, j], R, L, C) # beräkna nästa y genom euler framåt

    return t, y # returnera t och y

def plot_framat_euler(C, L, R_dämpad, t_span, N_list, y0, sol_i):
    plt.figure(figsize=(12, 8)) # skapa en figur

    for idx, N in enumerate(N_list): # för varje steg längd N
        t, y = euler_forward(t_span, y0, N, R_dämpad, L, C) # ta ut t och y

        plt.subplot(2, 2, idx + 1) # skapa en 2x2 subplot
        plt.plot(sol_i.t, sol_i.y[0], label="$q(t)$ exakt") # plotta exakta kurvor för q
        plt.plot(sol_i.t, sol_i.y[1], label="$i(t)$ exakt") # plotta exakta kurvor för i

        plt.plot(t, y[0], label="$q(t)$ Euler") # plotta eulers q kurva
        plt.plot(t, y[1], label="$i(t)$ Euler") # plotta eulers i kurva

        plt.title(f"Framåt Euler med N = {N}") # skriv titel
        plt.xlabel("t") # lodrät label
        plt.grid(True) # aktivera rutnät
        plt.legend() # visa nanm på kurvor
        plt.ylim(-3, 3) # begränsa y ledet så att man ser

def konvergens_studie(t_span, y0, C, L, R_dämpad, sol_i, N_studie):

    q_ref = sol_i.y[0, -1] # ta ut referens för q(20)
    i_ref = sol_i.y[1, -1] # ta ut referens för i(20)

    q_error = [] # lista av q fel
    i_error = [] # lista av i fel

    for N in N_studie: # för varje N
        _, y = euler_forward(t_span, y0, N, R_dämpad, L, C) # ta ut y värden

        err_q = np.abs(y[0, -1] - q_ref) # räkna q fel
        err_i = np.abs(y[1, -1] - i_ref) # räkna i fel

        q_error.append(err_q) # lägg till q fel
        i_error.append(err_i) # lägg till i fel

    q_error = np.array(q_error) # omvandla till array
    i_error = np.array(i_error) # omvandla till array

    q_p = np.log(q_error[:-1] / q_error[1:]) * (1 / np.log(2)) # beräkna noggranhetsordning för q
    i_p = np.log(i_error[:-1] / i_error[1:]) * (1 / np.log(2)) # beräkna noggranhetsordning för i

    print(f"q: p = {q_p}")
    print(f"i: p = {i_p}")


if __name__ == "__main__":
    main()
    plt.show()