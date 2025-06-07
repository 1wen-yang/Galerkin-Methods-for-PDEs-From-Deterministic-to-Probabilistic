import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
import scipy.integrate as integrate
import matplotlib.pyplot as plt

# different N values
N_list = [4, 8, 16, 32]

# loop over N
for N in N_list:
    h = 1 / N  # step size
    x = np.linspace(0, 1, N+1)  # grid points

    # source function
    f = lambda x_: np.exp(x_)

    # coefficient a(x)
    a = lambda x_: 1 + 0.5 * np.sin(np.pi * x_)

    # compute RHS vector b
    b = np.zeros(N-1)
    for i in range(N-1):
        integral_1, _ = integrate.quad(lambda xx: f(xx) * (xx - x[i]) / h, x[i], x[i+1])
        integral_2, _ = integrate.quad(lambda xx: f(xx) * (x[i+2] - xx) / h, x[i+1], x[i+2]) if i < N-2 else (0, 0)
        b[i] = integral_1 + integral_2

    # assemble tridiagonal matrix A
    main_diag = np.array([integrate.quad(lambda xx: a(xx), x[i], x[i+1])[0] / h**2 +
                          integrate.quad(lambda xx: a(xx), x[i+1], x[i+2])[0] / h**2
                          for i in range(N-1)])
    off_diag = np.array([-integrate.quad(lambda xx: a(xx), x[i+1], x[i+2])[0] / h**2
                         for i in range(N-2)])
    A = sp.diags([off_diag, main_diag, off_diag], [-1, 0, 1])

    # solve linear system
    ξ = spla.spsolve(A, b)

    # apply boundary conditions
    u = np.zeros(N+1)
    u[1:N] = ξ

    # plot FEM solution
    plt.plot(x, u, "-o", label=f"N={N}")

# plot settings
plt.xlabel("x")
plt.ylabel("u(x)")
plt.legend()
plt.grid()
plt.savefig("figure.pdf")
plt.show()