#   Falkner-Skan equation with suction/blowing.
#   Solution using a simple shooting method (secant) and Runge-Kutta RK4 integration.

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit


#   Functions to plot the results
def plot_f():

    plt.plot(f[:,0],eta, label=r"f($\eta$) vs $\eta$", color='red', linewidth=1.5)
    plt.ylabel(r"($\eta$)", fontweight='bold')
    plt.xlabel(r"f($\eta$)", fontweight='bold')
    plt.title(r"f($\eta$) vs $\eta$")
    plt.grid()
    plt.savefig('f.png', dpi=300)
    plt.legend()
    plt.show()

def plot_df():

    plt.plot(f[:,1],eta, label=r"f'($\eta$) vs $\eta$", color='red', linewidth=1.5)
    plt.ylabel(r"($\eta$)", fontweight='bold')
    plt.xlabel(r"f'($\eta$)", fontweight='bold')
    plt.title(r"f'($\eta$) vs $\eta$")
    plt.grid()
    plt.axvline(x=f[-1,1], color='black', linestyle='--', linewidth=1, label=r"f'($\infty$) = 1")
    plt.savefig('f_prime.png', dpi=300)
    plt.legend()
    plt.show()

def plot_d2f():

    plt.plot(f[:,2],eta, label=r"f''($\eta$) vs $\eta$", color='red', linewidth=1.5)
    plt.ylabel(r"($\eta$)", fontweight='bold')
    plt.xlabel(r"f''($\eta$)", fontweight='bold')
    plt.title(r"f''($\eta$) vs $\eta$")
    plt.grid()
    plt.axvline(x=f[0,2], color='black', linestyle='--', linewidth=1, label=r"f''(0) = %.5f" % f[0,2])
    plt.savefig('f_double_prime.png', dpi=300)
    plt.legend()
    plt.show()

def plot_d3f():
    
    plt.plot(f3,eta, label=r"f'''($\eta$) vs $\eta$", color='red', linewidth=1.5)
    plt.ylabel(r"($\eta$)", fontweight='bold')
    plt.xlabel(r"f'''($\eta$)", fontweight='bold')
    plt.title(r"f'''($\eta$) vs $\eta$")
    plt.grid()
    plt.savefig('f_triple_prime.png', dpi=300)
    plt.legend()
    plt.show()

def plot_normalized_v():

    plt.plot(nv,eta, label="Normalized Y-Velocity", color='red', linewidth=1.5)
    plt.xlabel(r"V$\sqrt{Re}$/$U_\infty$", fontweight='bold')
    plt.ylabel(r"($\eta$)", fontweight='bold')
    plt.title(r"V$\sqrt{Re}$/$U_\infty$ vs $\eta$")
    plt.grid()
    plt.axvline(x=nv[-1], color='black', linestyle='--', linewidth=1, label=r"V$\sqrt{Re}$/$U_\infty$($\infty$) = %.5f" % nv[-1])
    plt.savefig('normalized_y-velocity.png', dpi=300)
    plt.legend()
    plt.show()

def plot_howarth_f():

    plt.plot(f[:,0], eta, label=r"Computed f($\eta$) Solution", color='red', linewidth=1.5)
    plt.scatter(f_h, Eta, color='black', s=10, label="Data Points")
    plt.xlabel(r"f($\eta$)", fontweight='bold')
    plt.ylabel(r"($\eta$)", fontweight='bold')
    plt.title(r"Validation of f($\eta$) with Howarth Data")
    plt.grid()
    plt.legend()
    plt.savefig('howarth_f.png', dpi=300)
    plt.show()

def plot_howarth_df():

    plt.plot(f[:,1], eta, label=r"Computed f'($\eta$) Solution", color='red', linewidth=1.5)
    plt.scatter(f_prime_h, Eta, color='black', s=10, label="Data Points")
    plt.xlabel(r"f'($\eta$)", fontweight='bold')
    plt.ylabel(r"($\eta$)", fontweight='bold')
    plt.title(r"Validation of f'($\eta$) with Howarth Data")
    plt.grid()
    plt.legend()
    plt.savefig('howarth_f_prime.png', dpi=300)
    plt.show()

def plot_howarth_d2f():

    plt.plot(f[:,2], eta, label=r"Computed f''($\eta$) Solution", color='red', linewidth=1.5)
    plt.scatter(f_dprime_h, Eta, color='black', s=10, label="Data Points")
    plt.xlabel(r"f''($\eta$)", fontweight='bold')
    plt.ylabel(r"($\eta$)", fontweight='bold')
    plt.title(r"Validation of f''($\eta$) with Howarth Data")
    plt.grid()
    plt.legend()
    plt.savefig('howarth_f_double_prime.png', dpi=300)
    plt.show()    


#   Functions to solve the Falkner-Skan equation using RK4 and shooting method
def ff(m,u):

    f0 =   u[1]
    f1 =   u[2]
    f2 = - (((m+1)/2)*u[0]*u[2]) - (m*(1-(u[1]**2)))
    return np.array([f0,f1,f2])

def RK4(g,h,L,m,Bf):

    i_max = int(L/h)
    X = np.arange(i_max+1)*h
    U = np.zeros((i_max+1,3))
    U[0,:] = [(-(2*Bf)/(m+1)),0,g]

    for i in range(i_max):
        K1 = ff(m, U[i,:])
        K2 = ff(m, U[i,:] + K1*h/2)
        K3 = ff(m, U[i,:] + K2*h/2)
        K4 = ff(m, U[i,:] + K3*h)
    
        U[i+1,:] = U[i,:] + (h/6)*(K1+2*K2+2*K3+K4)

    # Break early if values blow up to prevent overflow/nan propagation
        if np.any(np.isnan(U[i + 1, :])) or np.any(np.abs(U[i + 1, :]) > 1e5):
            U[i + 1, :] = 1e5     # Penalty value for shooting method
            break 

    return X,U

def shoot(g,h,L,m,Bf):

    i_max = int(L/h)
    X,U = RK4(g,h,L,m,Bf)
    return 1.0 - U[i_max, 1]

def falkner_skan(h,L,tol,n_max,m,Bf):

    # Initial guess for f''(0) based on m
    if m <= 0:
        z = 0.2
    elif m <= 1:
        z = 0.33 + 0.9 * m
    else:
        z = 1.23 + 0.2 * m

    # Evaluate the shooting function at the initial guess
    fz = shoot(z,h,L,m,Bf) 

    # Check if the initial guess is already a solution
    if abs(fz) < tol:
        return z

    # Initial perturbation for the secant method
    delta = 1e-5;
    z_d = z + delta; fz_d = shoot(z_d,h,L,m,Bf)

    # Condition to start iterations
    n = 0; interval = tol+1;

    # Iterate using the secant method until convergence or maximum iterations reached
    while (interval >= tol and n <= n_max):
        denominator = fz_d - fz

    #  Check for small denominator to avoid division by zero
        if abs(denominator) < 1e-12:
            if abs(fz) < tol:
                break
            else:
                delta *= 0.5
                z_d = z + delta
                fz_d = shoot(z_d,h,L,m,Bf)
                continue

        #  Update the estimate using the secant method
        df2_old = z
        df2 = z - ((fz*delta)/(fz_d-fz)); 
        interval = abs(df2-df2_old)
        z = df2 
        z_d = z + delta
        fz = shoot(z,h,L,m,Bf)
        fz_d = shoot(z_d,h,L,m,Bf)
        n = n + 1
        print(" df2 = %14.7e (Estimated error %13.7e at iteration %d)" % (df2,abs(delta),n))   

    #   Check if the solution has converged
        if abs(fz) < tol:
            break

    #   Check if the maximum number of iterations was exceeded        
    if (n > n_max):
        raise RuntimeError('Too many iterations.')
    return df2

#  Functions to compute the boundary layer thickness, delta, and the normalized y-velocity
def X(rho, mu, Re):
    return ((mu*Re)/(rho))**(1/(m+1))

def bf(Vw,u_inf, rho, mu, x):
    return (Vw*((rho*u_inf*x)**(0.5)))/((mu**(0.5))*u_inf)

def delta(mu, rho, x, u_inf, m):
    return (((2)*mu*x)/((rho*u_inf*(1+m))))**(0.5)

def v(Re,u_inf,eta,f):
    X,U = eta,f
    fn = U[:,0]
    df = U[:,1]
    d2f = U[:,2]
    V = np.zeros(len(X))
    NV = np.zeros(len(X))
    for i in range(len(X)):
        V[i] = (u_inf/(2*(Re**(0.5)))*((X[i]*df[i])-(fn[i])))
        NV[i] = ((V[i]*(Re**(0.5)))/u_inf)
    return V,NV

def d3(eta,f):
    X,U = eta,f
    fn = U[:,0]
    df = U[:,1]
    d2f = U[:,2]
    d3f = np.zeros(len(X))
    for i in range(len(X)):
        d3f[i] = -((m+1)/2)*fn[i]*d2f[i] - (m*(1-(df[i]**2)))
    return d3f

def model_function(x, a, b, c, d, e):
    return a * (1 - np.exp(-(b * x**3 +c * x**2 + d * x + e)))


# Parameters
rho = 1.225;
mu = 1.81e-5;
tol = 1e-8;

n_max = int(input("Enter the maximum number of iterations: "))           # n_max = 100
h   = float(input("Enter the step size: "))                              # 0.01                             
m   = float(input("Enter the parameter m: "))                            # 0.5
Vw  = float(input("Enter the parameter Vw: "))                           # 1.0
Re   = float(input("Enter the value of Re: "))                           

# Boundary layer thickness, delta, is computed based on the value of m
if m <= 0:
    L = 15
elif m <= 1:
    L = 10
else:
    L = 5

x = X(rho, mu, Re)
print("\nx = %.5f" % x)

u_inf = (1.0*x)**(m);  
print("\nU_inf = %.5f" % u_inf)

Bf = bf(Vw,u_inf,rho,mu,x)
print("\nBf = %.5f" % Bf)

Delta = delta(mu,rho,x,u_inf,m)
print("\nBoundary layer thickness, delta = %.5f" % Delta)

# Solve the Falkner-Skan equation using the secant method
g = falkner_skan(h,L,tol,n_max,m,Bf)

# Recompute the Falkner-Skan solution with the optimised value for f''(0)                                                  
eta,f = RK4(g,h,L,m,Bf)     
print("\nFinal value for f''(0) = %.10f " % f[0,2])

# Function to compute the third derivative of f and the normalized y-velocity  
f3 = d3(eta,f)
v,nv = v(Re,u_inf,eta,f)

# Condition to start plotting results
t = 1
# Plotting results
while t == 1:
    plot = input("\nDo you want to plot the results? (Yes/No): ")
    if plot.lower() == 'y':
        try:
            input_choice = input("\nEnter: \n'1' to plot : f vs eta \n'2' to plot : f' vs eta \n'3' to plot : f'' vs eta \n'4' to plot : f''' vs eta \n\n")
            if input_choice == '1':
                plot_f()
                continue
            elif input_choice == '2':
                plot_df()
                continue
            elif input_choice == '3':
                plot_d2f()
                continue
            elif input_choice == '4':
                plot_d3f()
                continue
            else:
                print("Invalid choice. Please enter '1,2,3 or 4'.")
        except Exception as e:
            print(f"An error occurred: {e}. Please try again.")
    else:
        t = 0;
        break


# Howarth data
Eta = np.array([0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0])

# Data for f(η)
f_h = np.array(
    [0.00000, 0.16557, 0.65003, 1.39682, 2.30576, 3.28329, 4.27964, 5.27926, 6.27923, 7.27923]
)

# Data for f'(η)
f_prime_h = np.array(
    [0.00000, 0.32979, 0.62977, 0.84605, 0.95552, 0.99155, 0.99898, 0.99992, 1.00000, 1.00000]
)

# Data for f''(η)
f_dprime_h = np.array(
    [0.33206, 0.32301, 0.26675, 0.16136, 0.06424, 0.01591, 0.00240, 0.00022, 0.00001, 0.0]
)

plot_howarth = input("\nDo you want to plot the Howarth data for validation? (Yes/No): ")
if m == 0 and Bf == 0.0 and plot_howarth.lower() == 'y':
    plot_howarth_f()
    plot_howarth_df()
    plot_howarth_d2f()


