"""
Combined script: Worker–Entry Curves under Following Scenarios

1) Endogenous thetas with strong IRS in manufacturing.
2) Endogenous prices with CES production + IRS.
3) Cobb-Douglas Production and Endogenous prices + IRS.

"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import fsolve

# === Scenario 1: Worker Entry with Endogenous Thetas ===

# Parameters
beta      = 0.5     # bargaining weight
r         = 0.05    # discount rate
s         = 0.10    # separation rate
phi       = 4.0     # IRS exponent in manufacturing
alpha_match = 0.6   # matching elasticity
p_m       = 1.0     # manufacturing price
p_s       = 1.0     # service price
A_m       = 8.0     # manufacturing TFP
A_s       = 0.002   # service TFP
k_m       = 1.5     # vacancy cost in manufacturing
k_s       = 0.05    # vacancy cost in services
epsilon0  = 6.0     # slope shift for service curve
delta_s   = 0.5     # intercept shift for service curve

def q1(theta):
    """Matching function for scenario 1."""
    return max(theta, 1e-6)**(-alpha_match)

def rJ_v1(theta, Pi):
    """Value of a filled vacancy."""
    return ((1-beta)*q1(theta)*Pi) / (r + s + (1-beta)*q1(theta) + beta*theta*q1(theta))

def rJ_u1(theta, Pi, k):
    """Unemployment value for a match."""
    return (beta*theta*q1(theta)*(Pi - r*k)) / (r + s + beta*theta*q1(theta))

def Pi_m1(L_s, th):
    """Per-match profit in manufacturing."""
    fill = th * q1(th)
    E_m  = (fill / (s + fill)) * (1 - L_s)
    return p_m * A_m * E_m**phi

# 1) Solve for service tightness θ_s
theta_s    = fsolve(lambda th: rJ_v1(th, p_s*A_s) - r*k_s, 1.0)[0]
rJ_s_base  = rJ_u1(theta_s, p_s*A_s, k_s)

# 2) Sweep manufacturing share L_m and back out θ_m
L_m_1      = np.linspace(0.01, 0.99, 200)
L_s_1      = 1 - L_m_1
rJ_m_1     = np.empty_like(L_m_1)
th_guess1  = 1.0

for i, Ls in enumerate(L_s_1):
    def entry_m(th):
        return rJ_v1(th, Pi_m1(Ls, th)) - r*k_m
    th_star      = fsolve(entry_m, th_guess1)[0]
    th_guess1    = th_star
    rJ_m_1[i]    = rJ_u1(th_star, Pi_m1(Ls, th_star), k_m)

# 3) Build shifted service curve
rJ_s_1 = (rJ_s_base - delta_s) + epsilon0 * L_m_1

# 4) Plot intersections
plt.figure(figsize=(8,5))
plt.plot(L_m_1, rJ_m_1,   label=r'$rJ^U_m(L_m)$', lw=2)
plt.plot(L_m_1, rJ_s_1, '--',label=r'$rJ^U_s + \varepsilon_0\,L_m$', lw=2)
plt.xlabel(r'$L_m$ (Manufacturing share)')
plt.ylabel(r'$rJ^U$')
plt.title('Two‐Intersection Worker–Entry Curves (Endogenous Thetas)')
plt.legend()

diff1 = rJ_m_1 - rJ_s_1
for idx in np.where(np.sign(diff1[:-1]) != np.sign(diff1[1:]))[0]:
    xm = 0.5*(L_m_1[idx] + L_m_1[idx+1])
    ym = 0.5*(rJ_m_1[idx] + rJ_s_1[idx+1])
    plt.plot(xm, ym, 'ro')

plt.tight_layout()
plt.show()


# === Scenario 2: Worker's Entry with Endogenous Prices ===

# Parameters
alpha      = 0.5     # CES share
rho        = 0.5     # CES elasticity
A_m        = 5.0     # manufacturing TFP
A_s        = 1.0     # service TFP
phi_IRS    = 1.5     # IRS exponent in manufacturing

# Reuse r, s from above
m_match    = 1.0     # matching scale
eta        = 0.5     # matching elasticity exponent
beta       = 0.5     # bargaining weight
k_m        = 1.0     # vacancy cost in manufacturing
k_s        = 0.60    # vacancy cost in services
epsilon_h  = 0.20    # high‐skill education cost
epsilon_l  = 0.10    # low‐skill education cost
edu_diff   = r * (epsilon_h - epsilon_l)

def q2(theta):
    """Matching function for scenario 2."""
    return m_match * theta**(eta - 1.0)

def free_entry(vars, L_s):
    """Free‐entry conditions for θ_m and θ_s."""
    th_m, th_s = vars
    fill_m = th_m * q2(th_m)
    fill_s = th_s * q2(th_s)
    u_m    = s / (s + fill_m)
    u_s    = s / (s + fill_s)
    L_m    = 1 - L_s

    # Employment and output
    E_m = (1-u_m)*L_m
    E_s = (1-u_s)*L_s
    Y_m = A_m * E_m**(1 + phi_IRS)
    Y_s = A_s * E_s
    Y   = (alpha*Y_m**rho + (1-alpha)*Y_s**rho)**(1/rho)

    # Endogenous prices
    p_m_fe = alpha   * Y_m**(rho-1) * Y**(1-rho)
    p_s_fe = (1-alpha) * Y_s**(rho-1) * Y**(1-rho)

    # Match‐profits
    Pi_m = p_m_fe * A_m * E_m**phi_IRS
    Pi_s = p_s_fe * A_s

    # Free‐entry equations
    eq_m = ((1-beta)*q2(th_m)*Pi_m) / (r + s + (1-beta)*q2(th_m) + beta*th_m*q2(th_m)) - r*k_m
    eq_s = ((1-beta)*q2(th_s)*Pi_s) / (r + s + (1-beta)*q2(th_s) + beta*th_s*q2(th_s)) - r*k_s

    return [eq_m, eq_s]

# 1) Sweep L_s, solve for thetas, compute rJ^U
L_s_grid   = np.linspace(0.01, 0.99, 100)
rJ_m_vals  = np.empty_like(L_s_grid)
rJ_s_vals  = np.empty_like(L_s_grid)
guess      = np.array([1.0, 1.0])

for i, Ls in enumerate(L_s_grid):
    th_m, th_s           = fsolve(free_entry, guess, args=(Ls,))
    guess                = [th_m, th_s]
    tau_m                = th_m * q2(th_m)
    tau_s                = th_s * q2(th_s)
    u_m                  = s / (s + tau_m)
    u_s                  = s / (s + tau_s)
    L_m                  = 1 - Ls
    E_m                  = (1-u_m)*L_m
    E_s                  = (1-u_s)*Ls
    Y_m                  = A_m * E_m**(1 + phi_IRS)
    Y_s                  = A_s * E_s
    Y                    = (alpha*Y_m**rho + (1-alpha)*Y_s**rho)**(1/rho)
    p_m_val              = alpha   * Y_m**(rho-1) * Y**(1-rho)
    p_s_val              = (1-alpha) * Y_s**(rho-1) * Y**(1-rho)
    Pi_m                 = p_m_val * A_m * E_m**phi_IRS
    Pi_s                 = p_s_val * A_s

    # Unemployment‐value
    rJ_m_vals[i] = (beta * tau_m * (Pi_m - r*k_m)) / (r + s + beta*tau_m)
    rJ_s_vals[i] = (beta * tau_s * (Pi_s - r*k_s)) / (r + s + beta*tau_s) + edu_diff

# 2) Plot and mark intersections
L_m_grid = 1 - L_s_grid
plt.figure(figsize=(8,5))
plt.plot(L_m_grid, rJ_m_vals, lw=2, label=r'$rJ^U_m(L_m)$')
plt.plot(L_m_grid, rJ_s_vals, '--', lw=2,
         label=r'$rJ^U_s(L_m) + r(\epsilon_h-\epsilon_l)$')
plt.axhline(0, color='k', ls=':')

diff2 = rJ_m_vals - rJ_s_vals
for idx in np.where(np.sign(diff2[:-1]) != np.sign(diff2[1:]))[0]:
    xm = 0.5*(L_m_grid[idx] + L_m_grid[idx+1])
    ym = 0.5*(rJ_m_vals[idx] + rJ_s_vals[idx+1])
    plt.plot(xm, ym, 'ro')

plt.xlabel(r'$L_m$ (Manufacturing share)')
plt.ylabel(r'$rJ^U$')
plt.title('Unemployment‐Value Curves vs. $L_m$ with Stepwise IRS')
plt.legend()
plt.tight_layout()
plt.show()


# === Scenario 3: Cobb-Douglas Production Function ===

# 1. Parameters

alpha, rho    = 0.5, 0.5      
A_m, A_s      = 20.0, 1.0     
phi_IRS       = 3.0           

r, s          = 0.05, 0.10    
m_match, eta  = 1.0, 0.5      
beta, z       = 0.5, 0.10     

k_m, k_s      = 0.5, 0.6      

epsilon_h, epsilon_l = 10.0, 0.0
edu_coeff    = r * (epsilon_h - epsilon_l)

γm, γs = 0.6, 0.6

def q(theta):
    return m_match * np.maximum(theta,1e-6)**(eta - 1.0)

def free_entry(vars, L_s):
    th_m, th_s = vars
    # Beveridge fill rates
    fill_m = th_m * q(th_m)
    fill_s = th_s * q(th_s)
    u_m = s / (s + fill_m)
    u_s = s / (s + fill_s)
    L_m = 1 - L_s

    # outputs with IRS + Cobb–Douglas capital
    E_m = (1-u_m)*L_m
    E_s = (1-u_s)*L_s
    # note IRS exponent in E_m
    Y_m = A_m * (E_m**(1+phi_IRS))**γm * (k_m**(1-γm))
    Y_s = A_s * (E_s**γs)           * (k_s**(1-γs))

    # aggregate CES output & prices
    Y   = (alpha*Y_m**rho + (1-alpha)*Y_s**rho)**(1/rho)
    p_m = alpha   * Y_m**(rho-1) * Y**(1-rho)
    p_s = (1-alpha)*Y_s**(rho-1) * Y**(1-rho)

    # —— ⟵ HERE’S THE CHANGE ⟶——
    # use total output value as match surplus
    Pi_m = p_m * Y_m
    Pi_s = p_s * Y_s

    # free‐entry zero‐profit: (1-β)qΠ/(r+s+…) = r k
    fe_m = ((1-beta)*q(th_m)*Pi_m) / (r + s + (1-beta)*q(th_m) + beta*th_m*q(th_m)) \
           - r*k_m
    fe_s = ((1-beta)*q(th_s)*Pi_s) / (r + s + (1-beta)*q(th_s) + beta*th_s*q(th_s)) \
           - r*k_s

    return [fe_m, fe_s]

# 2. Sweep L_s, solve thetas, compute rJ^U’s 
L_s_grid = np.linspace(0.01, 0.99, 200)
rJ_m_vals = np.empty_like(L_s_grid)
rJ_s_vals = np.empty_like(L_s_grid)

guess = np.array([1.0, 1.0])
for i, Ls in enumerate(L_s_grid):
    th_m, th_s = fsolve(free_entry, guess, args=(Ls,), xtol=1e-8)
    guess = [th_m, th_s]

    tau_m = th_m * q(th_m)
    tau_s = th_s * q(th_s)
    u_m = s / (s+tau_m)
    u_s = s / (s+tau_s)
    L_m = 1 - Ls

    # recompute outputs & prices
    E_m = (1-u_m)*L_m
    E_s = (1-u_s)*Ls
    Y_m = A_m * (E_m**(1+phi_IRS))**γm * (k_m**(1-γm))
    Y_s = A_s * (E_s**γs)           * (k_s**(1-γs))
    Y   = (alpha*Y_m**rho + (1-alpha)*Y_s**rho)**(1/rho)
    p_m = alpha   * Y_m**(rho-1) * Y**(1-rho)
    p_s = (1-alpha)*Y_s**(rho-1) * Y**(1-rho)

    # new match surplus for rJ^U:
    Pi_m = p_m * Y_m
    Pi_s = p_s * Y_s

    # unemployment‐value now uses total output surplus
    rJ_m_vals[i] = beta * tau_m * (Pi_m - r*k_m) / (r + s + beta*tau_m)
    rJ_s_vals[i] = beta * tau_s * (Pi_s - r*k_s) / (r + s + beta*tau_s) \
                   + edu_diff

# 3. Plot & highlight intersections
L_m_grid = 1 - L_s_grid
plt.figure(figsize=(8,5))
plt.plot(L_m_grid, rJ_m_vals,   lw=2, label=r'$rJ^U_m(L_m)$')
plt.plot(L_m_grid, rJ_s_vals, '--',lw=2,
         label=r'$rJ^U_s(L_m)+r(\epsilon_h-\epsilon_l)$')
plt.axhline(0, color='k', ls=':')

diff = rJ_m_vals - rJ_s_vals
for idx in np.where(np.sign(diff[:-1])!=np.sign(diff[1:]))[0]:
    xm = 0.5*(L_m_grid[idx]+L_m_grid[idx+1])
    ym = 0.5*(rJ_m_vals[idx]+rJ_s_vals[idx+1])
    plt.plot(xm, ym, 'ro', ms=6)

plt.xlabel(r'$L_m$ (Manufacturing share)')
plt.ylabel(r'$rJ^U$')
plt.title('Multiplicity under Cobb–Douglas+IRS with Total Output Surplus')
plt.legend()
plt.tight_layout()
plt.show()
