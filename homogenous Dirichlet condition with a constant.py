import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
import scipy.integrate as integrate
import matplotlib.pyplot as plt

# exact solution for -u''(x)=exp(x), u(0)=0, u(1)=0
def u_exact(x):
    return -np.exp(x) + (np.e - 1) * x + 1

x_fine = np.linspace(0, 1, 1000)
plt.plot(x_fine, u_exact(x_fine), label="Exact")

N_list = [2, 4, 8, 16]

for N in N_list:
    h = 1 / N
    x = np.linspace(0, 1, N+1)

    f = lambda x_: np.exp(x_)
    a = 1
    
    # segmentation
    x = np.linspace(0, 1, N+1)

    f = lambda x: np.exp(x) # choose a f(x)

    # compute b using quadrature
    b = np.zeros(N-1)
    for j in range(N-1):  # since dirichlet
        integral_1, _ = integrate.quad(lambda x_: f(x_) * (x_ - x[j]) / h, x[j], x[j+1])
        integral_2, _ = integrate.quad(lambda x_: f(x_) * (x[j+2] - x_) / h, x[j+1], x[j+2]) if j < N-2 else (0, 0)
        b[j] = integral_1 + integral_2

    # tridiagonal matrix
    main_diag = np.full(N-1, 2*a/h)
    off_diag = np.full(N-2, -a/h)
    A = sp.diags([off_diag, main_diag, off_diag], [-1, 0, 1])

    # solving linear equation
    ξ = spla.spsolve(A, b) 

    # solve for u
    u = np.zeros(N+1)
    u[1:N] = ξ
    plt.plot(x, u, "-o", label=f"N={N}")
    
    
# plot
plt.xlabel("x"), plt.ylabel("u(x)"), plt.legend(), plt.grid()
plt.savefig("figure.pdf")
plt.show()
