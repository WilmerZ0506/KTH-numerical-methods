# KTH-numerical-methods
Python implementations of numerical methods, ODE solvers, and finite difference schemes developed at KTH.

Technologies Used
* Python (NumPy, SciPy, Matplotlib)

Key Projects & Features
* Non-Linear Least Squares: Implemented the Gauss-Newton method with an analytical Jacobian to evaluate trends in time-series data.
* Numerical Integration & Extrapolation: Built solvers for the Trapezoidal rule, Simpson's rule, and Richardson extrapolation, verifying convergence rates via step-size halving.
* ODE Solvers: Implemented Forward Euler for ordinary differential equations and systems (such as damped LC circuits) and benchmarked against SciPy's solve_ivp.
* Finite Difference Method (FDM): Developed sparse matrix solvers (scipy.sparse) for elliptic boundary value problems (heat conduction) including perturbation analysis.
