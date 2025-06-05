import numpy as np
import matplotlib.pyplot as plt

# different N values
N_list = [1, 2, 3]

# source function
f = lambda x: np.exp(x)

# grid for integration and plotting
x_nodes = np.linspace(0, 1, 1000)
x_plot  = np.linspace(0, 1, 20)


# Galerkin solution for each N
for N in N_list:
    b  = np.zeros(N)
    xi = np.zeros(N)
    # compute b_i = ∫ f(x) sin(jπx) dx
    for i in range(N):
        j = i + 1
        integrand = f(x_nodes) * np.sin(j * np.pi * x_nodes)
        b[i] = np.trapz(integrand, x_nodes)
    # xi_i = b_i / A_ii, with A_ii = (j^2 π^2)/2
    for i in range(N):
        j = i + 1
        A_ii = 0.5 * j**2 * (np.pi**2)
        xi[i] = b[i] / A_ii
    # build Galerkin solution
    y_plot = np.zeros_like(x_plot)
    for i in range(N):
        j = i + 1
        y_plot += xi[i] * np.sin(j * np.pi * x_plot)
    # plot Galerkin solution
    plt.plot(x_plot, y_plot,"-o", label=f'N = {N}')

# exact solution
u_exact = lambda x: -np.exp(x) + (np.e - 1) * x + 1
y_exact = u_exact(x_plot)

# plot exact solution
plt.plot(x_plot, y_exact, label='Exact')

# plot settings
plt.xlabel('x')
plt.ylabel('u(x)')
plt.legend()
plt.grid()
plt.savefig("figure.pdf")
plt.show()