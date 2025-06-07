import numpy as np
from scipy.special import roots_hermitenorm
import matplotlib.pyplot as plt
from scipy.sparse import lil_matrix
from scipy.sparse.linalg import spsolve


# MC section
np.random.seed(0)
N_MC_elem = 512
N_MC_nodes = N_MC_elem + 1
nodes_mc = np.linspace(-1.0, 1.0, N_MC_nodes)
h_mc = 2.0 / N_MC_elem
N_MC = 10000
ω1_s = np.random.normal(1.0, 0.1, N_MC)[:, None]
ω2_s = np.random.normal(0.5, 0.05, N_MC)[:, None]
ω3_s = np.random.normal(np.pi, 0.1 * np.pi, N_MC)[:, None]
ξ = 1.0 / np.sqrt(3)
x_gp1 = 0.5 * ((1 - ξ) * nodes_mc[:-1] + (1 + ξ) * nodes_mc[1:])
x_gp2 = 0.5 * ((1 + ξ) * nodes_mc[:-1] + (1 - ξ) * nodes_mc[1:])
a_gp1 = ω1_s + ω2_s * np.sin(ω3_s * x_gp1)
a_gp2 = ω1_s + ω2_s * np.sin(ω3_s * x_gp2)
I_all = 0.5 * h_mc * (a_gp1 + a_gp2)
N_int = N_MC_nodes - 2
Diag = (I_all[:, :N_int] + I_all[:, 1:N_int + 1]) / (h_mc ** 2)
Upper = np.zeros((N_MC, N_int))
Lower = np.zeros((N_MC, N_int))
Upper[:, :-1] = -I_all[:, 1:N_int] / (h_mc ** 2)
Lower[:, 1:] = -I_all[:, 1:N_int] / (h_mc ** 2)
b_int = np.zeros(N_int)
gl_pts = [-1/np.sqrt(3), 1/np.sqrt(3)]
gl_wts = [1.0, 1.0]
for elem in range(N_MC_elem):
    x_L, x_R = nodes_mc[elem], nodes_mc[elem+1]
    for ξv, w in zip(gl_pts, gl_wts):
        x_gp = 0.5 * ((1 - ξv) * x_L + (1 + ξv) * x_R)
        dx = 0.5 * (x_R - x_L)
        weight_x = w * dx
        φ_L = (x_R - x_gp) / h_mc
        φ_R = (x_gp - x_L) / h_mc
        f_val = np.exp(x_gp)
        if 0 < elem < N_MC_nodes - 1:
            b_int[elem-1] += weight_x * f_val * φ_L
        if 0 < elem+1 < N_MC_nodes - 1:
            b_int[elem] += weight_x * f_val * φ_R
RHS = np.tile(b_int, (N_MC, 1))
for i in range(1, N_int):
    lam = Lower[:, i] / Diag[:, i - 1]
    Diag[:, i] -= lam * Upper[:, i - 1]
    RHS[:, i] -= lam * RHS[:, i - 1]
u_MC_int = np.zeros_like(RHS)
u_MC_int[:, -1] = RHS[:, -1] / Diag[:, -1]
for i in range(N_int - 2, -1, -1):
    u_MC_int[:, i] = (RHS[:, i] - Upper[:, i] * u_MC_int[:, i + 1]) / Diag[:, i]
u_MC = np.zeros((N_MC, N_MC_nodes))
u_MC[:, 1:-1] = u_MC_int
u_mean_mc = u_MC.mean(axis=0)
u_var_mc = u_MC.var(axis=0)

# SGFEM section
N_list = [4, 8, 16]
sgfem_mean_all = []
sgfem_var_all = []
nodes_all = []

for N_elem in N_list:
    N_nodes = N_elem + 1
    nodes = np.linspace(-1.0, 1.0, N_nodes)
    h = 2.0 / N_elem
    order = 3
    # multi-index construction
    multi_index = [(i, j, k)
                   for i in range(order + 1)
                   for j in range(order + 1)
                   for k in range(order + 1)
                   if i + j + k <= order]
    K = len(multi_index)
    gh_order = 7
    xi_1d, w_1d = roots_hermitenorm(gh_order)
    w_1d /= np.sqrt(2 * np.pi)
    Xi_points = np.array([(a, b, c)
                          for a in xi_1d for b in xi_1d for c in xi_1d])
    Xi_weights = np.array([wa * wb * wc
                           for wa in w_1d for wb in w_1d for wc in w_1d])
    N_q = Xi_points.shape[0]
    ξ1, ξ2, ξ3 = Xi_points.T
    # Hermite basis
    He_vals = {
        0: np.ones((N_q, 3)),
        1: Xi_points,
        2: Xi_points ** 2 - 1,
        3: Xi_points ** 3 - 3 * Xi_points,
        4: Xi_points**4 - 6*Xi_points**2 + 3,      # 添加4阶
        5: Xi_points**5 - 10*Xi_points**3 + 15*Xi_points,  # 添加5阶
    }
    Psi_vals = np.zeros((N_q, K))
    for n, (i, j, k) in enumerate(multi_index):
        Psi_vals[:, n] = (He_vals[i][:, 0] / np.sqrt(np.math.factorial(i))) * \
                         (He_vals[j][:, 1] / np.sqrt(np.math.factorial(j))) * \
                         (He_vals[k][:, 2] / np.sqrt(np.math.factorial(k)))
    N_dof = N_nodes * K
    K_sg = lil_matrix((N_dof, N_dof))
    F_sg = np.zeros(N_dof)
    const_mode = 0
    gl_pts = [-1 / np.sqrt(3), 1 / np.sqrt(3)]
    gl_wts = [1.0, 1.0]
    ω1_vals = 1.0 + 0.1 * ξ1      # 15% 变异，增加椭圆性边距
    ω2_vals = 0.5 + 0.05 * ξ2     # 50% 变异，增强非线性
    ω3_vals = np.pi + 0.1*np.pi * ξ3  # 25% 变异
    for elem in range(N_elem):
        x_L, x_R = nodes[elem], nodes[elem + 1]
        M_elem = np.zeros((K, K))
        F_elem = [0.0, 0.0]
        for gp, gw in zip(gl_pts, gl_wts):
            x_gp = 0.5 * ((1 - gp) * x_L + (1 + gp) * x_R)
            dx = 0.5 * (x_R - x_L)
            weight_x = gw * dx
            a_vals = ω1_vals + ω2_vals * np.sin(ω3_vals * x_gp)
            integrand = a_vals * Xi_weights
            M_elem += weight_x * (Psi_vals.T @ (integrand[:, None] * Psi_vals))
            φ_L = (x_R - x_gp) / h
            φ_R = (x_gp - x_L) / h
            f_val = np.exp(x_gp)
            F_elem[0] += weight_x * f_val * φ_L
            F_elem[1] += weight_x * f_val * φ_R
        node_L, node_R = elem, elem + 1
        idx_L, idx_R = node_L * K, node_R * K
        fac = 1.0 / (h ** 2)
        K_sg[idx_L:idx_L + K, idx_L:idx_L + K] += fac * M_elem
        K_sg[idx_R:idx_R + K, idx_R:idx_R + K] += fac * M_elem
        K_sg[idx_L:idx_L + K, idx_R:idx_R + K] += -fac * M_elem
        K_sg[idx_R:idx_R + K, idx_L:idx_L + K] += -fac * M_elem
        F_sg[idx_L + const_mode] += F_elem[0]
        F_sg[idx_R + const_mode] += F_elem[1]
    # apply BCs
    boundary_dofs = np.r_[0:K, (N_nodes - 1) * K: N_nodes * K]
    mask = np.ones(N_dof, dtype=bool)
    mask[boundary_dofs] = False
    K_int = K_sg.tocsr()[mask][:, mask]
    F_int = F_sg[mask]
    U_int = spsolve(K_int, F_int)
    U_full = np.zeros(N_dof)
    U_full[mask] = U_int
    U_full = U_full.reshape(N_nodes, K)
    mode0 = multi_index.index((0, 0, 0))
    u_mean_sg = U_full[:, mode0]
    u_var_sg = (U_full ** 2).sum(axis=1) - u_mean_sg ** 2
    sgfem_mean_all.append(u_mean_sg)
    sgfem_var_all.append(u_var_sg)
    nodes_all.append(nodes)

# Plotting section
plt.figure(figsize=(8, 5))
plt.plot(nodes_mc, u_mean_mc, 'k-', label='MC mean (N=128)')
plt.fill_between(
    nodes_mc, u_mean_mc - np.sqrt(u_var_mc), u_mean_mc + np.sqrt(u_var_mc),
    color='gray', alpha=0.25, label='MC ±1σ'
)
colors = ['b', 'r', 'g']
for i, N_elem in enumerate(N_list):
    mean = sgfem_mean_all[i]
    var = sgfem_var_all[i]
    nodes = nodes_all[i]
    plt.plot(nodes, mean, color=colors[i], linestyle='--', label=f'SGFEM mean (N={N_elem})')
    plt.fill_between(
        nodes, mean - np.sqrt(var), mean + np.sqrt(var),
        color=colors[i], alpha=0.18, label=f'SGFEM ±1σ (N={N_elem})'
    )
plt.xlabel('x')
plt.ylabel('u(x)')
plt.title('SGFEM Mean & Variance vs MC')
plt.legend()
plt.tight_layout()
plt.savefig("figure.pdf")
plt.show()

def solve_sgfem(N_elem: int, order: int):

    # Mesh and basic variables
    N_nodes = N_elem + 1
    nodes   = np.linspace(-1.0, 1.0, N_nodes)
    h       = 2.0 / N_elem

    # Multi-index (PCE polynomials)
    multi_index = [(i, j, k)
                   for i in range(order + 1)
                   for j in range(order + 1)
                   for k in range(order + 1)
                   if i + j + k <= order]
    K = len(multi_index)  # number of random DoFs per node

    # 3D Gauss-Hermite quadrature points
    gh_order      = 2*order + 1  # recommended quadrature order
    ξ1d, w1d      = roots_hermitenorm(gh_order)
    w1d          /= np.sqrt(2*np.pi)  # normalization
    Xi_points     = np.array([(a,b,c) for a in ξ1d for b in ξ1d for c in ξ1d])
    Xi_weights    = np.array([wa*wb*wc for wa in w1d for wb in w1d for wc in w1d])
    ξ1, ξ2, ξ3    = Xi_points.T
    N_q           = Xi_points.shape[0]

    # 3D Hermite polynomial values Psi_{α}(ξ_q)
    He_vals = {  # explicit Hermite polynomials up to order 3
        0: np.ones((N_q, 3)),
        1: Xi_points,
        2: Xi_points**2 - 1,
        3: Xi_points**3 - 3*Xi_points,
        4: Xi_points**4 - 6*Xi_points**2 + 3,      
        5: Xi_points**5 - 10*Xi_points**3 + 15*Xi_points,
    }
    Psi_vals = np.zeros((N_q, K))
    for n, (i,j,k) in enumerate(multi_index):
        Psi_vals[:, n] = (He_vals[i][:,0]/np.sqrt(np.math.factorial(i)) *
                          He_vals[j][:,1]/np.sqrt(np.math.factorial(j)) *
                          He_vals[k][:,2]/np.sqrt(np.math.factorial(k)))

    # System matrix assembly
    N_dof = N_nodes * K
    K_sg  = lil_matrix((N_dof, N_dof))
    F_sg  = np.zeros(N_dof)
    const_mode = 0  # index of (0,0,0) in multi_index

    # Precompute random coefficient a(ξ_q)
    ω1_vals = 1.0 + 0.1      * ξ1
    ω2_vals = 0.5 + 0.05     * ξ2
    ω3_vals = np.pi + 0.1*np.pi * ξ3

    gl_pts = [-1/np.sqrt(3), 1/np.sqrt(3)]
    gl_wts = [1.0, 1.0]

    for elem in range(N_elem):
        xL, xR = nodes[elem], nodes[elem+1]
        M_elem = np.zeros((K, K))
        F_elem = [0.0, 0.0]

        for gp, gw in zip(gl_pts, gl_wts):
            x_gp  = 0.5*((1-gp)*xL + (1+gp)*xR)
            dx    = 0.5*(xR - xL)
            w_x   = gw * dx

            a_vals   = ω1_vals + ω2_vals * np.sin(ω3_vals * x_gp)  # (N_q,)
            integrnd = a_vals * Xi_weights  # combined weight

            M_elem  += w_x * (Psi_vals.T @ (integrnd[:,None] * Psi_vals))

            φL = (xR - x_gp)/h
            φR = (x_gp - xL)/h
            f  = np.exp(x_gp)
            F_elem[0] += w_x * f * φL
            F_elem[1] += w_x * f * φR

        # Insert into global matrix (linear elements, off-diagonal negative)
        nodeL, nodeR = elem, elem+1
        iL, iR = nodeL*K, nodeR*K
        fac = 1.0/(h**2)
        K_sg[iL:iL+K, iL:iL+K] +=  fac * M_elem
        K_sg[iR:iR+K, iR:iR+K] +=  fac * M_elem
        K_sg[iL:iL+K, iR:iR+K] += -fac * M_elem
        K_sg[iR:iR+K, iL:iL+K] += -fac * M_elem
        F_sg[iL + const_mode] += F_elem[0]
        F_sg[iR + const_mode] += F_elem[1]

    # Dirichlet BCs u=0
    bdry = np.r_[0:K, (N_nodes-1)*K : N_nodes*K]
    mask = np.ones(N_dof, dtype=bool); mask[bdry] = False
    K_int = K_sg.tocsr()[mask][:,mask]
    F_int = F_sg[mask]

    U_int = spsolve(K_int, F_int)
    U_all = np.zeros(N_dof); U_all[mask] = U_int
    U_all = U_all.reshape(N_nodes, K)
        
    # Mean and node coordinates
    mode0  = multi_index.index((0,0,0))
    u_mean = U_all[:, mode0]

    return u_mean, nodes

# A simple error function
def abs_L2(u_ref, x_ref, u_hat, x_hat):
    u_interp = np.interp(x_ref, x_hat, u_hat)
    num = np.trapz((u_interp - u_ref)**2, x_ref)
    return np.sqrt(num)

#  ——  Case 1: fixed h, increase PCE order K  ——

order_list = [1, 2, 3, 4, 5]  
N_elem     = 512

errors   = []  # relative L2 errors
means_sg = []  # store mean solutions for plotting
nodes_sg = None  # placeholder

for order in order_list:
    # SGFEM solve (order as variable)
    u_mean, nodes_sg = solve_sgfem(N_elem, order)
    means_sg.append(u_mean)

    # Compute relative L2 error
    err = abs_L2(u_mean_mc, nodes_mc, u_mean, nodes_sg)
    errors.append(err)
    print(f"order = {order:2d} ->  rel-L2 error = {err:.3e}")

# Convergence of error vs PCE order K
import matplotlib.pyplot as plt

plt.figure(figsize=(5, 4))
plt.semilogy(order_list, errors, 'o-')
plt.xlabel('PCE order $k$')
plt.ylabel(r' $L^2$ error')
plt.title('Error decay vs PCE Order (k)')
plt.grid(True, which='both', ls=':')
plt.tight_layout()
plt.savefig("figure.pdf")
plt.show()