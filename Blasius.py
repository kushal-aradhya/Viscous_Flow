#   Blasius equation for flat plate with 0 pressure gradient.
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

def plot_fitu():

    plt.plot(y_fit, x_vals, label=f"Fitted Curve", color='red', linewidth=1.5)
    plt.scatter(y_vals, x_vals, color='black', s=10, label="Data Points")
    plt.xlabel(r"f'($\eta$)", fontweight='bold')
    plt.ylabel(r"($\eta$)", fontweight='bold')
    plt.title(f"Curve Fit of Velocity Profile ")
    plt.grid()
    plt.legend()
    plt.savefig('curve_fit_velocity_profile.png', dpi=300)
    plt.show()


#   Functions to solve the Blasius equation using RK4 and shooting method
def ff(u):

    f0 =   u[1]
    f1 =   u[2]
    f2 = - 0.5*u[0] * u[2]
    return np.array([f0,f1,f2])

def RK4(a,h,L):

    i_max = int(L/h)
    X = np.arange(i_max+1)*h
    U = np.zeros((i_max+1,3))
    U[0,:] = [0,0,a]

    for i in range(i_max):
        K1 = ff(U[i,:])
        K2 = ff(U[i,:] + K1*h/2)
        K3 = ff(U[i,:] + K2*h/2)
        K4 = ff(U[i,:] + K3*h)

        U[i+1,:] = U[i,:] + (h/6)*(K1+2*K2+2*K3+K4)

    return X,U

def shoot(a,h,L):

    i_max = int(L/h)
    X,U = RK4(a,h,L)
    return 1.0 - U[i_max, 1]

def blasius(h,L,tol,n_max):

    z = 0
    fz = shoot(z,h,L)
    delta = 0.01; 
    z_d = z + delta
    fz_d = shoot(z_d,h,L)
    n = 0; 
    interval = tol+1;

    while (interval >= tol and n <= n_max):
        x_old = z
        x = z - ((fz*delta)/(fz_d-fz)); 
        interval = abs(x-x_old)
        z = x 
        z_d = z + delta
        fz = shoot(z,h,L)
        fz_d = shoot(z_d,h,L)
        n = n + 1
        print(" x = %14.7e (Estimated error %13.7e at iteration %d)" % (x,abs(delta),n))        
    if (n > n_max):
        raise RuntimeError('Too many iterations.')
    return x

def v(x,rho,mu,u_inf,eta,f):
    X,U = eta,f
    fn = U[:,0]
    df = U[:,1]
    d2f = U[:,2]
    Re = (u_inf*x*rho)/mu;
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
        d3f[i] = -0.5*X[i]*d2f[i]
    return d3f

def model_function(x, a, b, c, d, e):
    return a * (1 - np.exp(-(b * x**3 +c * x**2 + d * x + e)))


#   Parameters
rho = 1.225;
mu = 1.81e-5;  
u_inf = 1.0;
n_max = int(input("Enter the maximum number of iterations: "))           # n_max = 100
h   = float(input("Enter the step size: "))                              # 0.01
L   = float(input("Enter the domain length: "))                          # 30
tol = float(input("Enter the tolerance: "))                              # 1e-8
t = 1                              

a = blasius(h,L,tol,n_max)

#   Recompute the Blasius solution with the optimised value for f''(0)                                                  
eta,f = RK4(a,h,L)        
f3 = d3(eta,f)
v,nv = v(1.0,rho,mu,u_inf,eta,f)
print("\nFinal value for f''(0) = %.10f " % f[0,2])

#   Fitting function for velocity profile
y_vals = f[:, 1] 
x_vals = eta

initial_guesses = [1.0, 0.5, 0.2, 0.1 , 0.01] 
popt, pcov = curve_fit(model_function, x_vals, y_vals, bounds=(0, np.inf), p0=initial_guesses, maxfev=500000)
y_fit = model_function(x_vals, *popt)
ss_res = np.sum((y_vals - y_fit) ** 2)
ss_tot = np.sum((y_vals - np.mean(y_vals)) ** 2)
r2 = 1 - (ss_res / ss_tot)


#   Plotting results
while t == 1:
    plot = input("\nDo you want to plot the results? (y/n): ")
    if plot.lower() == 'y':
        try:
            input_choice = input("\nEnter: \n'1' to plot : f vs eta \n'2' to plot : f' vs eta \n'3' to plot : f'' vs eta \n'4' to plot : f''' vs eta \n'5' to plot : v vs eta\n'6' to plot : Curve fit of Velocity Profile\n\n")
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
            elif input_choice == '5':
                plot_normalized_v()
                continue
            elif input_choice == '6':   
                plot_fitu()
                print(f"\nCurve Fit Parameters: a = {popt[0]:.5f}, b = {popt[1]:.5f}, c = {popt[2]:.5f}, d = {popt[3]:.5f}, e = {popt[4]:.5f}")
                print(f"\nR-squared value: {r2:.5f}")
                continue
            else:
                print("Invalid choice. Please enter '1,2,3,4,5 or 6'.")
        except Exception as e:
            print(f"An error occurred: {e}. Please try again.")
    else:
        t = 0;
        break
