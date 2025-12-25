#helper functions 
import numpy as np
from scipy.interpolate import interp1d

def create_flux_distribution(Nr, Ntheta):
    r_data = np.array([0, 3, 6, 9, 12, 15, 18, 21, 24, 27, 30,
                   33, 36, 39, 42, 45, 48, 51, 54, 57,
                   60, 61, 62, 63, 64, 65, 66, 67, 68, 69,
                   70, 71, 72, 73, 74, 75])
    q_data = np.array([87.5, 86, 84, 80, 75, 69, 64, 60, 56, 53, 51,
                   48, 46, 44, 42, 41, 40, 39, 38.5, 38,
                   38, 40, 43, 48, 54, 62, 66, 70, 74, 77,
                   76, 74, 71, 68, 64, 60])
     
    flux_interp = interp1d(r_data, q_data, kind='cubic', bounds_error=False, fill_value="extrapolate")
    r = np.linspace(.000001, r_data[-1], Nr)
    theta = np.linspace(.000001, 2*np.pi, Ntheta)
    
    flux = np.zeros((Nr, Ntheta))
    for i, j in enumerate(r):
        flux[i, :] = flux_interp(j)
    
    return r, theta, flux

def time_step_2D(Nr, Ntheta, dr, dtheta, dt, T, q, r, k, rho, cp):
    T_old = T.copy()
    alpha = k / (rho * cp)
    T_new = np.zeros((Nr,Ntheta))
    for i in range(Nr - 1):
        for j in range(Ntheta):
            if j == Ntheta - 1:
                deriv = alpha*((T_old[i+1,j] - 2*T_old[i,j] + T_old[i-1,j]) / dr**2 + (1/r[i])*(T_old[i+1,j] - T_old[i-1,j]) / (2*dr) 
                          + (1/r[i]**2)*(T_old[i,0] - 2*T_old[i,j] + T_old[i,j-1]) / dtheta**2) + (q[i,j] * 1000) / (rho*cp) #*h ????
                T_new[i,j] = T_old[i,j] + dt * deriv
            else:
                deriv = alpha*((T_old[i+1,j] - 2*T_old[i,j] + T_old[i-1,j]) / dr**2 + (1/r[i])*(T_old[i+1,j] - T_old[i-1,j]) / (2*dr) 
                            + (1/r[i]**2)*(T_old[i,j+1] - 2*T_old[i,j] + T_old[i,j-1]) / dtheta**2) + (q[i,j] * 1000) / (rho*cp) #*h ????
                T_new[i,j] = T_old[i,j] + dt * deriv
                
    
    return T_new

def transient_run_2D(time, Nr, Ntheta, dr, dtheta, dt, initial_condition, q, r, k, rho, cp):
    num_runs = int(time / dt)
    T_old = initial_condition 
    transient = []
    for i in range(num_runs):
        S_new = time_step_2D(Nr, Ntheta, dr, dtheta, dt, T_old, q, r, k, rho, cp)
        transient.append(S_new)
        T_old = S_new
        print(i)
    return transient

##stopping point; need to use ghost nodes at BC because surface flux is important for accuracy
def time_step_3D(Nr, Ntheta, Nz, dr, dtheta, dz, dt, T, q, r, k, rho, cp):
    T_old = T.copy()
    alpha = k / (rho * cp)
    T_new = np.zeros((Nr,Ntheta,Nz))
    for i in range(Nr - 1):
        for j in range(Ntheta):
            for k in range(Nz):
                # if top layer than we add flux term
                if k == 0:
                    if j == Ntheta - 1:
                        deriv = (alpha*((T_old[i+1,j,k] - 2*T_old[i,j,k] + T_old[i-1,j,k]) / dr**2 + (1/r[i])*(T_old[i+1,j,k] - T_old[i-1,j,k]) / (2*dr) 
                                + (1/r[i]**2)*(T_old[i,0,k] - 2*T_old[i,j,k] + T_old[i,j-1,k]) / dtheta**2 + (T_old[i,j,k+1] - T_old[i,j,k]) / (2*dz))
                                + (q[i,j] * 1000) / (rho*cp*dz))
                        T_new[i,j,k] = T_old[i,j,k] + dt * deriv
                    else:
                        deriv = (alpha*((T_old[i+1,j,k] - 2*T_old[i,j,k] + T_old[i-1,j,k]) / dr**2 + (1/r[i])*(T_old[i+1,j,k] - T_old[i-1,j,k]) / (2*dr) 
                                + (1/r[i]**2)*(T_old[i,j+1,k] - 2*T_old[i,j,k] + T_old[i,j-1,k]) / dtheta**2 + (T_old[i,j,k+1] - T_old[i,j,k]) / (2*dz))
                                + (q[i,j] * 1000) / (rho*cp*dz))
                        T_new[i,j,k] = T_old[i,j,k] + dt * deriv

                # if last layer than we use BC
                
                elif k == Nz - 1:
                    if j == Ntheta - 1:
                        deriv = (alpha*((T_old[i+1,j,k] - 2*T_old[i,j,k] + T_old[i-1,j,k]) / dr**2 + (1/r[i])*(T_old[i+1,j,k] - T_old[i-1,j,k]) / (2*dr) 
                                + (1/r[i]**2)*(T_old[i,0,k] - 2*T_old[i,j,k] + T_old[i,j-1,k]) / dtheta**2 + (T_old[i,j,k-1] - T_old[i,j,k]) / (2*dz)))
                        T_new[i,j,k] = T_old[i,j,k] + dt * deriv
                    else:
                        deriv = (alpha*((T_old[i+1,j,k] - 2*T_old[i,j,k] + T_old[i-1,j,k]) / dr**2 + (1/r[i])*(T_old[i+1,j,k] - T_old[i-1,j,k]) / (2*dr) 
                                + (1/r[i]**2)*(T_old[i,j+1,k] - 2*T_old[i,j,k] + T_old[i,j-1,k]) / dtheta**2 + (T_old[i,j,k-1] - T_old[i,j,k]) / (2*dz)))
                        T_new[i,j,k] = T_old[i,j,k] + dt * deriv


                # if middle layers than normal
                else:
                    if j == Ntheta - 1:
                        deriv = (alpha*((T_old[i+1,j,k] - 2*T_old[i,j,k] + T_old[i-1,j,k]) / dr**2 + (1/r[i])*(T_old[i+1,j,k] - T_old[i-1,j,k]) / (2*dr) 
                                + (1/r[i]**2)*(T_old[i,0,k] - 2*T_old[i,j,k] + T_old[i,j-1,k]) / dtheta**2 + (T_old[i,j,k+1] - 2*T_old[i,j,k] + T_old[i,j,k-1]) / (dz**2)))
                        T_new[i,j,k] = T_old[i,j,k] + dt * deriv
                    else:
                        deriv = (alpha*((T_old[i+1,j,k] - 2*T_old[i,j,k] + T_old[i-1,j,k]) / dr**2 + (1/r[i])*(T_old[i+1,j,k] - T_old[i-1,j,k]) / (2*dr) 
                                + (1/r[i]**2)*(T_old[i,j+1,k] - 2*T_old[i,j,k] + T_old[i,j-1,k]) / dtheta**2 + (T_old[i,j,k+1] - 2*T_old[i,j,k] + T_old[i,j,k-1]) / (dz**2)))
                        T_new[i,j,k] = T_old[i,j,k] + dt * deriv
        
    return T_new

def transient_run_3D(time, Nr, Ntheta, Nz, dr, dtheta, dz, dt, initial_condition, q, r, k, rho, cp):
    num_runs = int(time / dt)
    T_old = initial_condition 
    transient = []
    for i in range(num_runs):
        S_new = time_step_3D(Nr, Ntheta, Nz, dr, dtheta, dz, dt, T_old, q, r, k, rho, cp)
        transient.append(S_new)
        T_old = S_new
        print(i)
    return transient

