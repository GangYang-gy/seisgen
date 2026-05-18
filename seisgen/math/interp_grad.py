import numpy as np
from seisgen.util_SPECFEM3D.xyz_reader import DEnquire_XYZ_anchors_Element

def grad_xyz(idx_processor, idx_element, xi, eta, gamma, NSPEC, data_dir):
    """
    Vectorized computation of the Jacobian between reference coordinates (xi, eta, gamma)
    and physical coordinates (x,y,z) for an element.

    Returns:
        dxi_dx, dxi_dy, dxi_dz,
        deta_dx, deta_dy, deta_dz,
        dgamma_dx, dgamma_dy, dgamma_dz
    """
    # 1. get anchor coordinates
    x_alpha, y_alpha, z_alpha = DEnquire_XYZ_anchors_Element(data_dir, idx_processor, idx_element, NSPEC)
    nodes = np.stack([x_alpha, y_alpha, z_alpha], axis=1)  # shape (8,3)

    # 2. Lagrange coefficients for 2x2x2 anchors
    xctrl = np.array([-1, 1])
    h_xi, hprime_xi = lagrange_poly(xi, xctrl)
    h_eta, hprime_eta = lagrange_poly(eta, xctrl)
    h_gamma, hprime_gamma = lagrange_poly(gamma, xctrl)

    # 3. build weight tensors for derivatives (8 anchors)
    # mapping from ix, iy, iz -> flat index
    idx = np.array([[0,1,2,3,4,5,6,7]]).flatten()
    h_xi_full = np.array([h_xi[0], h_xi[1], h_xi[1], h_xi[0], h_xi[0], h_xi[1], h_xi[1], h_xi[0]])
    h_eta_full = np.array([h_eta[0], h_eta[0], h_eta[1], h_eta[1], h_eta[0], h_eta[0], h_eta[1], h_eta[1]])
    h_gamma_full = np.array([h_gamma[0], h_gamma[0], h_gamma[0], h_gamma[0], h_gamma[1], h_gamma[1], h_gamma[1], h_gamma[1]])

    hprime_xi_full = np.array([hprime_xi[0], hprime_xi[1], hprime_xi[1], hprime_xi[0], hprime_xi[0], hprime_xi[1], hprime_xi[1], hprime_xi[0]])
    hprime_eta_full = np.array([hprime_eta[0], hprime_eta[0], hprime_eta[1], hprime_eta[1], hprime_eta[0], hprime_eta[0], hprime_eta[1], hprime_eta[1]])
    hprime_gamma_full = np.array([hprime_gamma[0], hprime_gamma[0], hprime_gamma[0], hprime_gamma[0], hprime_gamma[1], hprime_gamma[1], hprime_gamma[1], hprime_gamma[1]])

    # 4. compute Jacobian entries using einsum
    dx_dxi = np.einsum('i,ij->j', hprime_xi_full * h_eta_full * h_gamma_full, nodes)
    dx_deta = np.einsum('i,ij->j', h_xi_full * hprime_eta_full * h_gamma_full, nodes)
    dx_dgamma = np.einsum('i,ij->j', h_xi_full * h_eta_full * hprime_gamma_full, nodes)

    dy_dxi, dy_deta, dy_dgamma = dx_dxi[1], dx_deta[1], dx_dgamma[1]
    dz_dxi, dz_deta, dz_dgamma = dx_dxi[2], dx_deta[2], dx_dgamma[2]

    # 5. assemble Jacobian matrix
    J = np.array([[dx_dxi[0], dx_deta[0], dx_dgamma[0]],
                  [dy_dxi, dy_deta, dy_dgamma],
                  [dz_dxi, dz_deta, dz_dgamma]])

    # 6. invert Jacobian
    J_inv = np.linalg.inv(J)

    dxi_dx, dxi_dy, dxi_dz = J_inv[0]
    deta_dx, deta_dy, deta_dz = J_inv[1]
    dgamma_dx, dgamma_dy, dgamma_dz = J_inv[2]

    return dxi_dx, dxi_dy, dxi_dz, deta_dx, deta_dy, deta_dz, dgamma_dx, dgamma_dy, dgamma_dz


def lagrange_poly(xi, xctrl):
    nctrl = len(xctrl)
    h = np.zeros(nctrl)
    hprime = np.zeros(nctrl)
    
    for i in range(nctrl):
        x0 = xctrl[i]
        prod2 = np.prod([x0 - xctrl[j] for j in range(nctrl) if j != i])
        h[i] = np.prod([xi - xctrl[j] for j in range(nctrl) if j != i]) / prod2
        hprime[i] = np.sum([np.prod([xi - xctrl[k] for k in range(nctrl) if k != i and k != j]) for j in range(nctrl) if j != i]) / prod2
    return h, hprime

def grad_sgt(idx_element, NSPEC, data_dir, idx_processor, xi, eta, gamma, sgt_arr):
    '''
    Vectorized grad_sgt function using einsum
    sgt_arr: list of 27 arrays with shape (n_step, n_dim, n_para)
    '''
    # Step 1: Lagrange coefficients
    xctrl = np.array([-1, 0, 1])
    h_xi, h_prime_xi = lagrange_poly(xi, xctrl)
    h_eta, h_prime_eta = lagrange_poly(eta, xctrl)
    h_gamma, h_prime_gamma = lagrange_poly(gamma, xctrl)
    
    # Step 2: grad_xyz
    dxi_dx, dxi_dy, dxi_dz, deta_dx, deta_dy, deta_dz, dgamma_dx, dgamma_dy, dgamma_dz = \
        grad_xyz(idx_processor, idx_element, xi, eta, gamma, NSPEC, data_dir)
    
    # Step 3: reshape sgt_arr -> (3,3,3,n_step,n_dim,n_para)
    sgt_arr_np = np.array(sgt_arr)  # shape (27, n_step, n_dim, n_para)
    n_step, n_dim, n_para = sgt_arr_np.shape[1:]
    sgt_arr_reshaped = sgt_arr_np.reshape(3,3,3,n_step,n_dim,n_para)
    
    # Step 4: weight tensors
    h_xi = h_xi[:, None, None]
    h_eta = h_eta[None, :, None]
    h_gamma = h_gamma[None, None, :]
    h_prime_xi = h_prime_xi[:, None, None]
    h_prime_eta = h_prime_eta[None, :, None]
    h_prime_gamma = h_prime_gamma[None, None, :]
    
    W_dx = h_prime_xi * dxi_dx * h_eta * h_gamma + h_xi * h_prime_eta * deta_dx * h_gamma + h_xi * h_eta * h_prime_gamma * dgamma_dx
    W_dy = h_prime_xi * dxi_dy * h_eta * h_gamma + h_xi * h_prime_eta * deta_dy * h_gamma + h_xi * h_eta * h_prime_gamma * dgamma_dy
    W_dz = h_prime_xi * dxi_dz * h_eta * h_gamma + h_xi * h_prime_eta * deta_dz * h_gamma + h_xi * h_eta * h_prime_gamma * dgamma_dz
    
    # Step 5: einsum to sum over 3x3x3 GLL points
    dsgt_dx = np.einsum('ijk,ijk...->...', W_dx, sgt_arr_reshaped)
    dsgt_dy = np.einsum('ijk,ijk...->...', W_dy, sgt_arr_reshaped)
    dsgt_dz = np.einsum('ijk,ijk...->...', W_dz, sgt_arr_reshaped)
    
    return dsgt_dx, dsgt_dy, dsgt_dz
