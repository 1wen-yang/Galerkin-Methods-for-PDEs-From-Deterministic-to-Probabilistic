# -*- coding: utf-8 -*-
"""
Created on Tue Mar 11 13:30:18 2025

@author: willi
"""

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
import scipy.integrate as integrate
import matplotlib.pyplot as plt

# exact solution for -u''(x)=exp(x), u(0)=alpha, u(1)=beta
def u_exact(x, alpha=0, beta=0.00001):
    return -np.exp(x) + (beta - alpha - 1 + np.exp(1)) * x + alpha + 1

# different N values
N_list = [4, 8, 16, 32]

# plot exact solution
x_fine = np.linspace(0, 1, 1000)
plt.plot(x_fine, u_exact(x_fine), label="Exact")

for N in N_list:
    # parameter
    a = 1
    h = 1 / N  # step size adjusted for [0,1]

    # segmentation
    x = np.linspace(0, 1, N+1)

    f = lambda xx: np.exp(xx)  # choose a f(x)

    # non-homogeneous Dirichlet boundary conditions
    alpha = 0  # u(0) = alpha
    beta = 0.00001
    # u(1) = beta

    # compute b using quadrature
    b = np.zeros(N-1)
    for j in range(1, N):
        integral_1, _ = integrate.quad(lambda xx: f(xx)*(xx - x[j-1]) / h, x[j-1], x[j])
        integral_2, _ = integrate.quad(lambda xx: f(xx)*(x[j+1] - xx) / h, x[j], x[j+1]) if j < N-1 else (0, 0)
        b[j-1] = integral_1 + integral_2

    # adjust b for boundary conditions
    b[0]  += a/h * alpha
    b[-1] += a/h * beta

    # tridiagonal matrix
    main_diag = np.full(N-1, 2*a/h)
    off_diag = np.full(N-2, -a/h)
    A = sp.diags([off_diag, main_diag, off_diag], [-1, 0, 1])

    # solving linear equation
    ξ = spla.spsolve(A, b) 

    # solve for u and apply boundary conditions
    u = np.zeros(N+1)
    u[1:N] = ξ
    u[0] = alpha
    u[N] = beta

    # plot FEM solution
    plt.plot(x, u, "-o", label=f"FEM (N={N})")

plt.xlabel("x"), plt.ylabel("u(x)")
plt.legend()
plt.grid()
plt.savefig("figure.pdf")
plt.show()