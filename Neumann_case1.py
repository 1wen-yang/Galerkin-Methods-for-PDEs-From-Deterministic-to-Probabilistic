import numpy as np
from numpy.polynomial.hermite import hermgauss
from numpy.polynomial.legendre import leggauss
from scipy.sparse import lil_matrix
from scipy.sparse.linalg import spsolve
import math
import matplotlib.pyplot as plt
from functools import lru_cache

# Problem & PCE settings
mu_means = [1.0, 0.5, math.pi, 1.0]
mu_stds  = [0.1, 0.05, 0.1*math.pi, 0.2]
PCE_ORDER = 5    # chaos order

# Generate multi‐indices for Hermite chaos
def generate_multi_indices(dim, max_degree):
    multi_idx = []
    def recurse(current):
        if sum(current) <= max_degree:
            if len(current) == dim:
                multi_idx.append(tuple(current))
            else:
                for d in range(max_degree - sum(current) + 1):
                    recurse(current + [d])
    recurse([])
    return sorted(multi_idx, key=lambda idx: sum(idx))

multi_indices = generate_multi_indices(4, PCE_ORDER)
N_p = len(multi_indices)
index_from_multi = {mi: i for i,mi in enumerate(multi_indices)}

# Hermite polynomials (probabilists', normalized)
@lru_cache(None)
def hermite_prob(n, x):
    if n == 0: return 1.0
    if n == 1: return x
    return x*hermite_prob(n-1,x) - (n-1)*hermite_prob(n-2,x)

def hermite_basis_values(x):
    return [ hermite_prob(n, x)/math.sqrt(math.factorial(n)) for n in range(PCE_ORDER+1) ]

# Spatial mesh & FE matrices
def create_mesh(num_elements):
    return np.linspace(-1.0, 1.0, num_elements+1)

def assemble_spatial_matrices(nodes):
    N = len(nodes)
    h = nodes[1] - nodes[0]
    K = np.zeros((N,N)); M = np.zeros((N,N))
    for e in range(N-1):
        i,j = e,e+1
        K[i,i] +=  1/h; K[j,j] +=  1/h
        K[i,j] += -1/h; K[j,i] += -1/h
        M[i,i] += 2*h/6; M[j,j] += 2*h/6
        M[i,j] += 1*h/6; M[j,i] += 1*h/6
    return K,M

# Gauss-Hermite quadrature
gh_deg = 50
gh_x, gh_w = hermgauss(gh_deg)

# expected sin term
def expected_sin_term(i3, j3, x_val):
    m3,s3 = mu_means[2], mu_stds[2]
    tot = 0.0
    for y,w in zip(gh_x, gh_w):
        X = math.sqrt(2)*y
        H = hermite_basis_values(X)
        mu3 = m3 + s3*X
        tot += w * H[i3]*H[j3] * math.sin(mu3*x_val)
    return (math.sqrt(2)/math.sqrt(2*math.pi)) * tot

# assemble global SG system
def assemble_global_system(nodes):
    N = len(nodes)
    h = nodes[1] - nodes[0]
    K_sp, M_sp = assemble_spatial_matrices(nodes)
    total_dofs = N * N_p
    A = lil_matrix((total_dofs, total_dofs))
    b = np.zeros(total_dofs)
    gl_x, gl_w = leggauss(6)
    const_idx = index_from_multi[(0,0,0,0)]

    for p_idx, pm in enumerate(multi_indices):
        n1p,n2p,n3p,n4p = pm
        for q_idx, qm in enumerate(multi_indices):
            n1q,n2q,n3q,n4q = qm

            # μ1 diffusion
            diff1 = 0.0
            if (n2p,n3p,n4p)==(n2q,n3q,n4q):
                if   n1p==n1q:   diff1 += mu_means[0]
                if   n1p==n1q+1: diff1 += mu_stds[0]*math.sqrt(n1q+1)
                elif n1q==n1p+1: diff1 += mu_stds[0]*math.sqrt(n1p+1)

            # μ4 reaction
            react = 0.0
            if (n1p,n2p,n3p)==(n1q,n2q,n3q):
                if   n4p==n4q:   react += mu_means[3]
                if   n4p==n4q+1: react += mu_stds[3]*math.sqrt(n4q+1)
                elif n4q==n4p+1: react += mu_stds[3]*math.sqrt(n4p+1)

            # μ2·sin(μ3 x)
            sin_int = None
            if (n1p,n4p)==(n1q,n4q):
                B2=0.0
                if   n2p==n2q:     B2+=mu_means[1]
                if   n2p==n2q+1:   B2+=mu_stds[1]*math.sqrt(n2q+1)
                elif n2q==n2p+1:   B2+=mu_stds[1]*math.sqrt(n2p+1)
                if abs(B2)>1e-14:
                    sin_int = lambda x,B2=B2,i3=n3p,j3=n3q: B2*expected_sin_term(i3,j3,x)

            # fill SG matrix
            for i in range(N):
                for j in range(N):
                    if diff1 and K_sp[i,j]!=0:
                        A[i*N_p+p_idx, j*N_p+q_idx] += diff1*K_sp[i,j]
                    if react and M_sp[i,j]!=0:
                        A[i*N_p+p_idx, j*N_p+q_idx] += react*M_sp[i,j]

            if sin_int:
                for e in range(N-1):
                    xl,xr = nodes[e],nodes[e+1]
                    c_aa=1/h**2; c_bb=1/h**2; c_ab=-1/h**2
                    val=0.0
                    for xi,wi in zip(gl_x,gl_w):
                        xv=0.5*(xl+xr)+0.5*(xr-xl)*xi
                        wv=wi*0.5*(xr-xl)
                        val+=wv*sin_int(xv)
                    i0,j0=e,e+1
                    A[i0*N_p+p_idx, i0*N_p+q_idx] += c_aa*val
                    A[j0*N_p+p_idx, j0*N_p+q_idx] += c_bb*val
                    A[i0*N_p+p_idx, j0*N_p+q_idx] += c_ab*val
                    A[j0*N_p+p_idx, i0*N_p+q_idx] += c_ab*val

    # RHS: source + Neumann
    for e in range(len(nodes)-1):
        xl,xr = nodes[e],nodes[e+1]
        nl,nr = e,e+1
        for xi,wi in zip(gl_x,gl_w):
            xv=0.5*(xl+xr)+0.5*(xr-xl)*xi
            wv=wi*0.5*(xr-xl)
            fv=math.exp(xv)
            ph_l=(xr-xv)/h; ph_r=(xv-xl)/h
            b[nl*N_p+const_idx] += fv*ph_l*wv
            b[nr*N_p+const_idx] += fv*ph_r*wv

    b[0   *N_p+const_idx] += 7.0
    b[-1  *N_p+const_idx] += 5.0

    return A.tocsr(), b

# MC reference
def alpha(x, mu):
    return mu[0] + mu[1]*math.sin(mu[2]*x)

def f_source(x):
    return math.exp(x)
np.random.seed(12345)
def monte_carlo_solution(num_elements, N_mc=2000):
    nodes = create_mesh(num_elements)
    N = len(nodes)
    h = nodes[1] - nodes[0]
    sols = np.zeros((N_mc, N))
    for k in range(N_mc):
        mu = [np.random.normal(mu_means[i], mu_stds[i]) for i in range(4)]
        K = np.zeros((N,N)); M = np.zeros((N,N))
        for e in range(N-1):
            xm = 0.5*(nodes[e]+nodes[e+1])
            Ke = alpha(xm,mu)*np.array([[1,-1],[-1,1]])/h
            Me = mu[3]*np.array([[2,1],[1,2]])*h/6
            i,j=e,e+1
            K[i:i+2,i:i+2]+=Ke
            M[i:i+2,i:i+2]+=Me
        A = K+M
        F = np.zeros(N)
        for i,x in enumerate(nodes):
            F[i] = f_source(x)*(h if 0<i<N-1 else h/2)
        F[0]  += 7.0
        F[-1] += 5.0
        A[np.diag_indices(N)] += 1e-10
        sols[k] = np.linalg.solve(A, F)
    mean_mc = sols.mean(axis=0)
    std_mc  = sols.std(axis=0)
    return nodes, mean_mc, std_mc

# Solve & plot
def main(N_list=[4,8,16,32], mc_elements=None, mc_samples=2000):
    plt.figure(figsize=(8,5))

    for N in N_list:
        nodes = create_mesh(N)
        A, b = assemble_global_system(nodes)
        U = spsolve(A, b)
        mean_sg = np.zeros(len(nodes))
        var_sg = np.zeros(len(nodes))
        idx0 = index_from_multi[(0,0,0,0)]
        for j in range(len(nodes)):
            coeffs = U[j*N_p:(j+1)*N_p]
            mean_sg[j] = coeffs[idx0]
            var_sg[j] = np.sum(coeffs[1:]**2)
        std_sg = np.sqrt(var_sg)
        plt.plot(nodes, mean_sg, label=f'SG N={N}')
        plt.fill_between(nodes, mean_sg-std_sg, mean_sg+std_sg, alpha=0.1)

    if mc_elements is None:
        mc_elements = max(N_list)
    x_mc, m_mc, s_mc = monte_carlo_solution(mc_elements, N_mc=mc_samples)
    plt.plot(x_mc, m_mc, 'k--', label='MC mean')
    plt.fill_between(x_mc, m_mc-s_mc, m_mc+s_mc, color='gray', alpha=0.2, label='MC ±1σ')

    plt.xlabel("x")
    plt.ylabel("u(x)")
    plt.title("SG vs MC comparison")
    plt.legend()
    plt.grid(True)
    plt.savefig("figure.pdf")
    plt.show()

def test_normalization():
    for n in range(PCE_ORDER + 1):
        tot = sum(w * hermite_basis_values(np.sqrt(2)*x)[n]**2 for x, w in zip(gh_x, gh_w))
        norm = tot / np.sqrt(np.pi)
        print(f"E[psi_{n}^2] = {norm}")


def error_vs_pce_order(N=64, max_order=5, mc_samples=5000):
    x_mc, m_mc, s_mc = monte_carlo_solution(N, N_mc=mc_samples)
    errors = []
    orders = list(range(1, max_order+1))
    for order in orders:
        global PCE_ORDER, multi_indices, N_p, index_from_multi
        PCE_ORDER = order
        multi_indices = generate_multi_indices(4, PCE_ORDER)
        N_p = len(multi_indices)
        index_from_multi = {mi: i for i,mi in enumerate(multi_indices)}
        nodes = create_mesh(N)
        A, b = assemble_global_system(nodes)
        U = spsolve(A, b)
        mean_sg = np.zeros(len(nodes))
        idx0 = index_from_multi[(0,0,0,0)]
        for j in range(len(nodes)):
            coeffs = U[j*N_p:(j+1)*N_p]
            mean_sg[j] = coeffs[idx0]
        err = np.linalg.norm(mean_sg - m_mc)
        errors.append(err)
        print(f"PCE_ORDER={order}, L2 error={err:.3e}")
    plt.figure()
    plt.plot(orders, errors, 'o-', label='log(L2 error)')
    plt.yscale('log')
    plt.xlabel('PCE order $k$')
    plt.ylabel(r'$L^2$ error')
    plt.title('Error decay vs PCE Order (k)')
    plt.grid(True)
    plt.savefig("figure.pdf", bbox_inches="tight")
    plt.show()
    
if __name__ == "__main__":
    main([4,8,16], mc_elements=128, mc_samples=5000)
    test_normalization()
    error_vs_pce_order(N=256, max_order=5, mc_samples=8000)
    
