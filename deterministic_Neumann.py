import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
import scipy.integrate as integrate
import matplotlib.pyplot as plt

# List of N values to test
N_list = [2, 3, 5]

# Exact solution (u'(0)=0, u'(1)=1)
def exact_solution(x):
    return 1 + x + (np.exp(x) + np.exp(2 - x)) / (np.exp(2) - 1)

# Plot exact solution
x_fine = np.linspace(0, 1, 400)
u_exact = exact_solution(x_fine)
plt.plot(x_fine, u_exact, label="Exact solution", color="blue", lw=2)

# Source term f(x) = f1(x) + f2(x) = (1+x) + 0 = 1+2x
f1 = lambda xx: 1.0 + xx
f2 = lambda xx: 0.0
f = lambda xx: f1(xx) + f2(xx)

# FEM solver for each N
for N in N_list:
    h = 1.0 / N
    x = np.linspace(0, 1, N+1)  # N+1 nodes

    # Initialize global load vector and stiffness matrix data
    b = np.zeros(N+1)
    data, rows, cols = [], [], []
    
    # Assemble local contributions
    for e in range(N):
        xL, xR = x[e], x[e+1]
        he = xR - xL

        # Local stiffness matrix (from -u'')
        K_local = (1.0/he) * np.array([[1, -1],
                                       [-1, 1]])
        # Local mass matrix (from +u)
        M_local = (he/6.0) * np.array([[2, 1],
                                       [1, 2]])
        A_local = K_local + M_local

        # Local load vector
        phi0 = lambda xx: (xR - xx) / he
        phi1 = lambda xx: (xx - xL) / he
        b0, _ = integrate.quad(lambda xx: f(xx) * phi0(xx), xL, xR)
        b1, _ = integrate.quad(lambda xx: f(xx) * phi1(xx), xL, xR)
        b_local = np.array([b0, b1])
        
        # Assemble global system
        for i_local, i_global in enumerate([e, e+1]):
            b[i_global] += b_local[i_local]
            for j_local, j_global in enumerate([e, e+1]):
                data.append(A_local[i_local, j_local])
                rows.append(i_global)
                cols.append(j_global)
    
    # Apply Neumann BCs
    b[-1] += 1.0  # u'(1)=1
    # u'(0)=0 gives no contribution

    # Build global stiffness matrix
    A = sp.coo_matrix((data, (rows, cols)), shape=(N+1, N+1)).tocsr()
    
    # Solve linear system
    u = spla.spsolve(A, b)
    
    # Plot solution
    plt.plot(x, u, '-o', label=f"FEM, N={N}")

plt.xlabel("x")
plt.ylabel("u(x)")
plt.legend(loc="best")
plt.grid(True)
plt.savefig("figure.pdf")
plt.show()
