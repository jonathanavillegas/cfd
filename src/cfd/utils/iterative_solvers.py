"""
Iterative solvers for linear systems.
"""
import numpy as np


def jacobi_solver(A, b, x0, tol=1, max_iter=1000):
    """
    Solves Ax = b using the Jacobi iterative method.
    
    Parameters:
    -----------
    A : ndarray
        Coefficient matrix (n x n)
    b : ndarray
        Right-hand side vector (n,)
    x0 : ndarray
        Initial guess vector (n,)
    tol : float
        Tolerance for convergence (default: 1)
    max_iter : int
        Maximum number of iterations (default: 1000)
        
    Returns:
    --------
    x : ndarray
        Approximate solution vector
    """
    A = np.array(A, dtype=float)
    b = np.array(b, dtype=float)
    x = np.array(x0, dtype=float)
    n = len(b)

    for iteration in range(max_iter):
        x_new = np.zeros_like(x)
        for i in range(n):
            sum_ax = np.dot(A[i, :], x) - A[i, i] * x[i]
            x_new[i] = (b[i] - sum_ax) / A[i, i]
        
        # Check convergence
        if np.linalg.norm(x_new - x, ord=np.inf) < tol:
            print(f"Converged in {iteration + 1} iterations.")
            return x_new
        print(np.linalg.norm(x_new - x, ord=np.inf))
        x = x_new
    print("Maximum iterations reached without convergence.")
    return x


def gauss_seidel_solver(A, b, x0=None, tol=1e-4, max_iter=1000):
    """
    Solves the linear system Ax = b using the Gauss-Seidel iterative method.

    Parameters:
    -----------
    A : ndarray
        Coefficient matrix (must be square, preferably diagonally dominant or SPD)
    b : ndarray
        Right-hand side vector
    x0 : ndarray, optional
        Initial guess (default: zeros)
    tol : float
        Tolerance for convergence (default: 1e-4)
    max_iter : int
        Maximum number of iterations (default: 1000)

    Returns:
    --------
    x : ndarray
        Solution vector
    """
    n = len(b)
    x = np.zeros(n) if x0 is None else x0.copy()

    for iteration in range(max_iter):
        x_new = x.copy()

        for i in range(n):
            sum1 = np.dot(A[i, :i], x_new[:i])  # updated values
            sum2 = np.dot(A[i, i+1:], x[i+1:])  # old values
            x_new[i] = (b[i] - sum1 - sum2) / A[i, i]

        # Check for convergence
        if np.linalg.norm(x_new - x, ord=np.inf) < tol:
            print(f"Converged in {iteration+1} iterations.")
            return x_new
        print(np.linalg.norm(x_new - x, ord=np.inf))
        x = x_new

    print("Did not converge within the maximum number of iterations.")
    return x

