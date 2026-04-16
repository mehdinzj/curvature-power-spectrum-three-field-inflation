import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter
# constants:

V0 = 18.1e-10
Mp = 1
mchi = np.sqrt(V0 / 500)
mpsi = np.sqrt(V0 / 80000)
phi0 = np.sqrt(6) * Mp
b0 = 7.6 / Mp
y0 = 7.6 / Mp
p0 = 0
mp = 2.435515 * 1e18
mpc = (6.394970897 * 1e-39) / mp   # Mpc^{-1} =  2.63 * 1e-57 Mp (arXiv: 2110.12251)
k0 = 0.05 * mpc
def power(a,b):     # a ^ b
    s = 1
    for i in range(b):
        s *= a
    return s
# potential V(phi,chi,psi):

def V(phi,chi,psi):
    return V0 * power(phi,2) / (power(phi0,2) + power(phi,2)) + 0.5 * power(mchi,2) * power(chi,2) + 0.5 * power(mpsi,2) * power(psi,2)

# derivates of the potential:


def V_phi(phi,chi,psi):
   h = 1e-5
   return (V(phi + h,chi,psi) - V(phi - h,chi,psi)) / (2 * h)

def V_chi(phi,chi,psi):
   h = 1e-5
   return (V(phi,chi + h,psi) - V(phi,chi - h,psi)) / (2 * h)

def V_psi(phi,chi,psi):
   h = 1e-5
   return (V(phi,chi,psi + h) - V(phi,chi,psi - h)) / (2 * h)

def V_phiphi(phi,chi,psi):
   h = 1e-5
   return (V_phi(phi + h , chi,psi) - V_phi(phi - h , chi,psi)) / (2 * h)

def V_chichi(phi,chi,psi):
   h = 1e-5
   return (V_chi(phi,chi + h,psi) - V_chi(phi,chi - h,psi)) / (2 * h)

def V_psipsi(phi,chi,psi):
   h = 1e-5
   return (V_psi(phi,chi,psi + h) - V_psi(phi,chi,psi - h)) / (2 * h)

def V_phichi(phi,chi,psi):
   h = 1e-5
   return (V_phi(phi,chi + h,psi) - V_phi(phi,chi - h,psi)) / (2 * h)

def V_phipsi(phi,chi,psi):
   h = 1e-5
   return (V_phi(phi,chi,psi + h) - V_phi(phi,chi,psi - h)) / (2 * h)

def V_chipsi(phi,chi,psi):
   h = 1e-5
   return (V_chi(phi,chi,psi + h) - V_chi(phi,chi,psi - h)) / (2 * h)
# noncanonical coefficients:

def b(phi):
    return b0 * phi
def y(phi):
    return y0 * phi
def p(chi):
    return p0 * chi

# derivatives:

def b_phi(phi):
    h = 1e-5
    return (b(phi + h) - b(phi - h)) / (2 * h)
def y_phi(phi):
    h = 1e-5
    return (y(phi + h) - y(phi - h)) / (2 * h)
def p_chi(chi):
    h = 1e-5
    return (p(chi + h) - p(chi - h)) / (2 * h)

# second derivatives for perturbations:

def b_phiphi(phi):
    h = 1e-5
    return (b_phi(phi + h) - b_phi(phi - h)) / (2 * h)
def y_phiphi(phi):
    h = 1e-5
    return (y_phi(phi + h) - y_phi(phi - h)) / (2 * h)
def p_chichi(chi):
    h = 1e-5
    return (p_chi(chi + h) - p_chi(chi - h)) / (2 * h)
def G_ab(phi,chi):     # field-space metric
    return np.array([[1,0,0],[0, np.exp(2 * b(phi)) ,0],[0,0, np.exp(2 * y(phi) + 2 * p(chi))]])
def H(phi,chi,psi,dphi_dN,dchi_dN,dpsi_dN):
    G = G_ab(phi,chi)
    value = np.sqrt((2 * V(phi,chi,psi)) / (6 * power(Mp,2) - G[0][0] * power(dphi_dN,2) - G[1][1] * power(dchi_dN ,2) - G[2][2] * power(dpsi_dN ,2)))
    return value

def dH_dN(phi,chi,psi,dphi_dN,dchi_dN,dpsi_dN):
    G = G_ab(phi,chi)
    Hub = H(phi,chi,psi,dphi_dN,dchi_dN,dpsi_dN)
    value = - 0.5 * (1 / power(Mp,2)) * Hub * (G[0][0] * power(dphi_dN ,2) + G[1][1] * power(dchi_dN,2) + G[2][2] * power(dpsi_dN,2))
    return value

def numeps(phi,chi,psi,dphi_dN,dchi_dN,dpsi_dN):
    Hub = H(phi,chi,psi,dphi_dN,dchi_dN,dpsi_dN)
    dH = dH_dN(phi,chi,psi,dphi_dN,dchi_dN,dpsi_dN)
    return - dH / Hub
# Background equations of motion:

def background(N,y):
    phi, dphi_dN, chi, dchi_dN, psi, dpsi_dN = y
    H_val = H(phi,chi,psi,dphi_dN,dchi_dN,dpsi_dN)
    eps = numeps(phi,chi,psi,dphi_dN,dchi_dN,dpsi_dN)
    G = G_ab(phi,chi)
    d2phi_dN2 = -(3 - eps) * dphi_dN + b_phi(phi) * G[1][1] * power(dchi_dN,2) + y_phi(phi) * G[2][2] * power(dpsi_dN,2) - V_phi(phi,chi,psi) / power(H_val,2)
    d2chi_dN2 = -(3 + 2 * b_phi(phi) * dphi_dN - eps) * dchi_dN + p_chi(chi) * (G[2][2] / G[1][1]) * power(dpsi_dN,2) - (1 / G[1][1]) * V_chi(phi,chi,psi) / power(H_val,2)
    d2psi_dN2 = -(3 + 2 * b_phi(phi) * dphi_dN + 2 * p_chi(chi) * dchi_dN - eps) * dpsi_dN - (1 / G[2][2]) * V_psi(phi,chi,psi) / power(H_val,2)
    return np.array([dphi_dN, d2phi_dN2, dchi_dN, d2chi_dN2, dpsi_dN, d2psi_dN2])
# Initial condition + Integration:

phi_i = 7
chi_i = 7.31
psi_i = 7.31
phi_dot_i = 0#- V_phi(phi_i,chi_i,psi_i) / V(phi_i,chi_i,psi_i)
chi_dot_i = 0#- V_chi(phi_i,chi_i,psi_i) / V(phi_i,chi_i,psi_i)
psi_dot_i = 0#- V_psi(phi_i,chi_i,psi_i) / V(phi_i,chi_i,psi_i)
Inc = np.array([phi_i, phi_dot_i, chi_i, chi_dot_i, psi_i, psi_dot_i])
 
sample_size = 10000 # to pass the stiffness
NEnd = 99.81 # estimated end for inflation

NE = np.linspace(0 , NEnd, sample_size)
sol_background = solve_ivp(background, [0 , NEnd], Inc,t_eval=NE,
                          method='LSODA', rtol=1e-6, atol=1e-8)
N_vals = sol_background.t
phi_vals = sol_background.y[0]
dphidN = sol_background.y[1]
chi_vals = sol_background.y[2]
dchidN = sol_background.y[3]
psi_vals = sol_background.y[4]
dpsidN = sol_background.y[5]

Hubble_vals = [H(phi_vals[i],chi_vals[i],psi_vals[i],dphidN[i],dchidN[i],dpsidN[i]) for i in range(len(N_vals))]
epsilon = np.array([-(1 / H(phi_vals[i],chi_vals[i],psi_vals[i],dphidN[i],dchidN[i],dpsidN[i])) * dH_dN(phi_vals[i],chi_vals[i],psi_vals[i],dphidN[i],dchidN[i],dpsidN[i]) for i in range(len(N_vals))])
Eta = np.gradient(epsilon,N_vals) / epsilon
end_inf = np.argmin(np.abs(1 - epsilon))
N_end = N_vals[end_inf]
N_pivot = N_end - 50 
pivot_idx = np.argmin(np.abs(N_pivot - N_vals))
a0 = (k0 / Hubble_vals[pivot_idx]) * np.exp(-N_vals[pivot_idx])    # normalize a(N) so that k∗ = 0.05 crosses the Hubble radius N∗ = 50 e-folds before the end of inflation.
a_vals = a0 * np.exp(N_vals)
K = a_vals * Hubble_vals  # Comoving Hubble scale
def sigma_dot(Ne):
    idx = np.argmin(np.abs(Ne - N_vals))
    phi = phi_vals[idx]
    dphi = dphidN[idx]
    chi = chi_vals[idx]
    dchi = dchidN[idx]
    psi = psi_vals[idx]
    dpsi = dpsidN[idx]
    Hub = H(phi,chi,psi,dphi,dchi,dpsi)
    G = G_ab(phi,chi)
    val = power(Hub,2) * (dphi ** 2 + G[1][1] * dchi ** 2 + G[2][2] * dpsi ** 2)
    return np.sqrt(val)
sigma_vals = [sigma_dot(ne) for ne in N_vals]
deltaN = 3    # this value is numerically stable in 3-field system

def N_subhorizon(k):
    exit_idx = np.argmin(np.abs(k - K))
    N_init = N_vals[exit_idx] - deltaN
    N_init_idx = np.argmin(np.abs(N_init - N_vals))
    if N_init_idx >= len(N_vals) - 1:
        return np.nan
    else:
        return N_vals[N_init_idx]
    def BD_Inc1(k):
    N_init = N_subhorizon(k)
    idx = np.argmin(np.abs(N_vals - N_init))
    a , Hub = a_vals[idx] , Hubble_vals[idx]
    aH , eps , phi , chi, psi , phi_p , chi_p, psi_p = K[idx] , epsilon[idx] , phi_vals[idx] , chi_vals[idx], psi_vals[idx] , dphidN[idx] , dchidN[idx], dpsidN[idx]
    G = G_ab(phi,chi)
    BD_state = 1 / (a * np.sqrt(2 * k))
    dBD_dtau = -BD_state * (1j * k + aH)
    dBD_dN = dBD_dtau / aH
    inc1 = BD_state
    dinc1 = dBD_dN
    inc2 , dinc2 = 0 , 0
    inc3 , dinc3 = 0 , 0
    Phi = (phi_p * dinc1 + G[1][1] * chi_p * dinc2 + G[2][2] * psi_p * dinc3 + (3 * phi_p + b_phi(phi) * G[1][1] * chi_p ** 2 + y_phi(phi) * G[2][2] * psi_p ** 2 + V_phi(phi,chi,psi) / Hub ** 2) * inc1 + (3 * G[1][1] * chi_p + p_chi(chi) * G[2][2] * psi_p ** 2 + V_chi(phi,chi,psi) / Hub ** 2) * inc2 + (3 * G[2][2] * psi_p + V_psi(phi,chi,psi) / Hub ** 2) * inc3) / (2 * power(Mp,2) *(eps - k ** 2 / aH ** 2))
    Phi_p = - Phi + (phi_p * inc1 + G[1][1] * chi_p * inc2 + G[2][2] * psi_p * inc3) / (2 * power(Mp,2))
    return [inc1 , inc2 , inc3 , dinc1 , dinc2 , dinc3 , Phi , Phi_p]

def BD_Inc2(k):
    N_init = N_subhorizon(k)
    idx = np.argmin(np.abs(N_vals - N_init))
    a , Hub = a_vals[idx] , Hubble_vals[idx]
    aH , eps , phi , chi, psi , phi_p , chi_p, psi_p = K[idx] , epsilon[idx] , phi_vals[idx] , chi_vals[idx], psi_vals[idx] , dphidN[idx] , dchidN[idx], dpsidN[idx]
    G = G_ab(phi,chi)
    BD_state = 1 / (a * np.sqrt(2 * k))
    dBD_dtau = -BD_state * (1j * k + aH)
    dBD_dN = dBD_dtau / aH
    inc1 , dinc1 = 0 , 0
    inc2 = BD_state / np.sqrt(G[1][1])
    dinc2 = (dBD_dtau - b_phi(phi) * phi_p * BD_state * aH) / (aH * np.sqrt(G[1][1]))
    inc3 , dinc3 = 0 , 0
    Phi = (phi_p * dinc1 + G[1][1] * chi_p * dinc2 + G[2][2] * psi_p * dinc3 + (3 * phi_p + b_phi(phi) * G[1][1] * chi_p ** 2 + y_phi(phi) * G[2][2] * psi_p ** 2 + V_phi(phi,chi,psi) / Hub ** 2) * inc1 + (3 * G[1][1] * chi_p + p_chi(chi) * G[2][2] * psi_p ** 2 + V_chi(phi,chi,psi) / Hub ** 2) * inc2 + (3 * G[2][2] * psi_p + V_psi(phi,chi,psi) / Hub ** 2) * inc3) / (2 * power(Mp,2) *(eps - k ** 2 / aH ** 2))
    Phi_p = - Phi + (phi_p * inc1 + G[1][1] * chi_p * inc2 + G[2][2] * psi_p * inc3) / (2 * power(Mp,2))
    return [inc1 , inc2 , inc3 , dinc1 , dinc2 , dinc3 , Phi , Phi_p]

def BD_Inc3(k):
    N_init = N_subhorizon(k)
    idx = np.argmin(np.abs(N_vals - N_init))
    a , Hub = a_vals[idx] , Hubble_vals[idx]
    aH , eps , phi , chi, psi , phi_p , chi_p, psi_p = K[idx] , epsilon[idx] , phi_vals[idx] , chi_vals[idx], psi_vals[idx] , dphidN[idx] , dchidN[idx], dpsidN[idx]
    G = G_ab(phi,chi)
    BD_state = 1 / (a * np.sqrt(2 * k))
    dBD_dtau = -BD_state * (1j * k + aH)
    dBD_dN = dBD_dtau / aH
    inc1 , dinc1 = 0 , 0
    inc2 , dinc2 = 0 , 0
    inc3 = BD_state / np.sqrt(G[2][2])
    dinc3 = (dBD_dtau - BD_state * aH * (y_phi(phi) * phi_p + p_chi(chi) * chi_p)) / (aH * np.sqrt(G[2][2]))
    Phi = (phi_p * dinc1 + G[1][1] * chi_p * dinc2 + G[2][2] * psi_p * dinc3 + (3 * phi_p + b_phi(phi) * G[1][1] * chi_p ** 2 + y_phi(phi) * G[2][2] * psi_p ** 2 + V_phi(phi,chi,psi) / Hub ** 2) * inc1 + (3 * G[1][1] * chi_p + p_chi(chi) * G[2][2] * psi_p ** 2 + V_chi(phi,chi,psi) / Hub ** 2) * inc2 + (3 * G[2][2] * psi_p + V_psi(phi,chi,psi) / Hub ** 2) * inc3) / (2 * power(Mp,2) *(eps - k ** 2 / aH ** 2))
    Phi_p = - Phi + (phi_p * inc1 + G[1][1] * chi_p * inc2 + G[2][2] * psi_p * inc3) / (2 * power(Mp,2))
    return [inc1 , inc2 , inc3 , dinc1 , dinc2 , dinc3 , Phi , Phi_p]

# Mukhanov-Sasaki equations:

def Mukhanov_Sasaki(N,state,k):
    X1 , X2 , X3 , dX1_dN , dX2_dN , dX3_dN , Phi , dPhi = state 
    n = np.argmin(np.abs(N - N_vals))
    a , Hub , eps = a_vals[n] , Hubble_vals[n] , epsilon[n]
    aH = a * Hub
    phi , chi , psi , dphi , dchi , dpsi = phi_vals[n] , chi_vals[n] , psi_vals[n] , dphidN[n] , dchidN[n] , dpsidN[n]
    G = G_ab(phi,chi)
    d2X1dN2 = 4 * dPhi * dphi - 2 * Phi * V_phi(phi,chi,psi) / power(Hub,2) - (3 - eps) * dX1_dN - (power(k / aH,2) + V_phiphi(phi,chi,psi) / power(Hub,2) - (2 * b_phi(phi) ** 2 + b_phiphi(phi)) * G[1][1] * power(dchi,2) - (2 * y_phi(phi) ** 2 + y_phiphi(phi)) * G[2][2] * power(dpsi,2)) * X1 - (V_phichi(phi,chi,psi) / Hub ** 2 - 2 * y_phi(phi) * p_chi(chi) * G[2][2] * dpsi ** 2) * X2 + 2 * b_phi(phi) * G[1][1] * dchi * dX2_dN - V_phipsi(phi,chi,psi) * X3 / Hub ** 2 + 2 * y_phi(phi) * G[2][2] * dpsi * dX3_dN
    d2X2dN2 = 4 * dPhi * dchi - 2 * Phi * V_chi(phi,chi,psi) / (power(Hub,2) * G[1][1]) - (3 + 2 * b_phi(phi) * dphi - eps) * dX2_dN - (power(k / aH,2) + V_chichi(phi,chi,psi) / (power(Hub,2) * G[1][1]) - (2 * p_chi(chi) ** 2 + p_chichi(chi)) * G[2][2] * dpsi ** 2 / G[1][1]) * X2 - (2 * b_phiphi(phi) * dphi * dchi + V_phichi(phi,chi,psi) / (power(Hub,2) * G[1][1]) - 2 * b_phi(phi) * V_chi(phi,chi,psi) / (power(Hub,2) * G[1][1]) + 2 * (b_phi(phi) - y_phi(phi)) * p_chi(chi) * dpsi ** 2 * G[2][2] / G[1][1]) * X1 - 2 * b_phi(phi) * dchi * dX1_dN + 2 * p_chi(chi) * (G[2][2] / G[1][1]) * dpsi * dX3_dN - V_chipsi(phi,chi,psi) * X3 / (power(Hub,2) * G[1][1])
    d2X3dN2 = 4 * dPhi * dpsi - 2 * Phi * V_psi(phi,chi,psi) / (power(Hub,2) * G[2][2]) - (3 + 2 * y_phi(phi) * dphi + 2 * p_chi(chi) * dchi - eps) * dX3_dN - (power(k / aH,2) + V_psipsi(phi,chi,psi) / (power(Hub,2) * G[2][2])) * X3 - (2 * y_phiphi(phi) * dphi * dpsi - 2 * y_phi(phi) * V_psi(phi,chi,psi) / (power(Hub,2) * G[2][2]) + V_phipsi(phi,chi,psi) / (power(Hub,2) * G[2][2])) * X1 - (2 * p_chichi(chi) * dchi * dpsi - 2 * p_chi(chi) * V_psi(phi,chi,psi) / (power(Hub,2) * G[2][2]) + V_chipsi(phi,chi,psi) / (power(Hub,2) * G[2][2])) * X2 - 2 * y_phi(phi) * dpsi * dX1_dN - 2 * p_chi(chi) * dpsi * dX2_dN
    d2PhidN2 = -(7 - eps) * dPhi - (2 * V(phi,chi,psi) / power(Mp * Hub,2) + power(k / aH,2)) * Phi - (V_phi(phi,chi,psi) * X1 + V_chi(phi,chi,psi) * X2 + V_psi(phi,chi,psi) * X3) / power(Hub * Mp,2)
    return np.array([dX1_dN , dX2_dN , dX3_dN , d2X1dN2 , d2X2dN2 , d2X3dN2 , dPhi , d2PhidN2])
def Runge_Kutta(k, init):
    A = init
    N_init = N_subhorizon(k)
    NEP = [x for x in N_vals if x >= N_init]
    solutions = np.zeros((len(NEP), len(A)), dtype = complex)
    dN = NEP[1] - NEP[0]
    for i, Ne in enumerate(NEP):
        solutions[i, :] = A 
        k1 = Mukhanov_Sasaki(Ne, A, k)
        k2 = Mukhanov_Sasaki(Ne + 0.5 * dN, A + 0.5 * k1 * dN, k)
        k3 = Mukhanov_Sasaki(Ne + 0.5 * dN, A + 0.5 * k2 * dN, k)
        k4 = Mukhanov_Sasaki(Ne + dN, A + k3 * dN, k)
        A += (dN / 6) * (k1 + 2 * k2 + 2 * k3 + k4)
    return NEP , solutions[:, 0] , solutions[:,1] , solutions[:,2] , solutions[:,6]
# power spectrum calculator

def CPS(k):
    T1 , Delta_phi1 , Delta_chi1 , Delta_psi1, Phi_B1 = Runge_Kutta(k,BD_Inc1(k))
    T2 , Delta_phi2 , Delta_chi2 , Delta_psi2, Phi_B2 = Runge_Kutta(k,BD_Inc2(k))
    T3 , Delta_phi3 , Delta_chi3 , Delta_psi3, Phi_B3 = Runge_Kutta(k,BD_Inc3(k))
    end_idx = np.argmin(np.abs(1 - epsilon))
    sol_idx = np.argmin(np.abs(N_vals[end_idx] - T1))
    Hub = Hubble_vals[end_idx]
    sigma = sigma_vals[end_idx]
    phi , chi , psi , dphi , dchi , dpsi = phi_vals[end_idx] , chi_vals[end_idx] , psi_vals[end_idx] , dphidN[end_idx] , dchidN[end_idx] , dpsidN[end_idx]
    G = G_ab(phi,chi)
    R1 = Phi_B1[sol_idx] + power(Hub / sigma,2) * (dphi * Delta_phi1[sol_idx] + G[1][1] * dchi * Delta_chi1[sol_idx] + G[2][2] * dpsi * Delta_psi1[sol_idx])
    R2 = Phi_B2[sol_idx] + power(Hub / sigma,2) * (dphi * Delta_phi2[sol_idx] + G[1][1] * dchi * Delta_chi2[sol_idx] + G[2][2] * dpsi * Delta_psi2[sol_idx])
    R3 = Phi_B3[sol_idx] + power(Hub / sigma,2) * (dphi * Delta_phi3[sol_idx] + G[1][1] * dchi * Delta_chi3[sol_idx] + G[2][2] * dpsi * Delta_psi3[sol_idx])
    Sol = np.abs(R1) ** 2 + np.abs(R2) ** 2 + np.abs(R3) ** 2 
    return (k ** 3 / (2 * power(np.pi,2))) * Sol
c_idx = np.argmin(np.abs(0.05 * mpc - K))
n_elements = len(K) 
indices = np.linspace(np.argmin(np.abs(5 - N_vals)), n_elements - 1, 300, dtype = int)
k_sample = [K[i] for i in indices]
N_sample = [N_vals[i] for i in indices]

POW = np.array([CPS(k) for k in k_sample])
highlight_N = N_vals[c_idx]
highlight_POW = CPS(K[c_idx])
plt.figure(figsize=(8, 5))
plt.title("Curvature Power Spectrum in triple inflation", fontsize = 14.75)
plt.loglog(np.array(k_sample) / mpc,POW, color ='b')
plt.scatter(K[c_idx] / mpc, highlight_POW, color='brown', s=50, zorder=5)
plt.annotate('CMB scale',
             xy=(K[c_idx]/mpc, highlight_POW),
             xytext=(0.001, highlight_POW * 2.5),color='brown',fontsize=11.5)
plt.xlabel(r'$k\;\;[\mathrm{Mpc}]^{-1}$',fontweight='bold', fontsize=14)
plt.ylabel(r'$\mathcal{P}_\mathcal{R}(k)$',fontweight='bold', fontsize=14)
#plt.savefig('3fiels_power_inK.pdf',format = 'pdf' ,  bbox_inches="tight", dpi = 600)
plt.show()






