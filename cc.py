import numpy as np
import time

def implement_normal_equation_numpy(X_raw, y):
    """
    Implements the Normal Equation to find linear regression parameters (theta)
    using NumPy.

    Args:
        X_raw (np.ndarray): Input features (m samples, n features).
                            Does NOT include the bias (1s) column.
        y (np.ndarray): Target values (m samples).

    Returns:
        np.ndarray: The parameter vector theta (beta_0, beta_1, ..., beta_n).
    """
    # 1. Add a bias (1s) column to X_raw to create the Design Matrix X'
    # X_prime will have shape (m, n + 1)
    ones = np.ones((X_raw.shape[0], 1))
    X_prime = np.hstack((ones, X_raw))

    # Ensure y is a column vector
    y = y.reshape(-1, 1)

    # 2. Calculate (X_prime.T @ X_prime)
    XT_X = X_prime.T @ X_prime

    # 3. Calculate the inverse of (X_prime.T @ X_prime)
    try:
        XT_X_inv = np.linalg.inv(XT_X)
    except np.linalg.LinAlgError:
        print("Warning: (X'.T @ X') is singular. Cannot compute inverse. "
              "This might be due to multicollinearity or too few samples.")
        return None

    # 4. Calculate (X_prime.T @ y)
    XT_y = X_prime.T @ y

    # 5. Calculate theta = (XT_X_inv @ XT_y)
    theta = XT_X_inv @ XT_y

    return theta

# --- Generate synthetic data for demonstration ---
# m: number of samples
# n: number of features
m = 1000
n = 5

# Generate random features
np.random.seed(42) # for reproducibility
X_data = 10 * np.random.rand(m, n)

print(X_data.shape)

# Generate true parameters for y = b0 + b1*x1 + ... + bn*xn + noise
true_beta_0 = 3.0
true_betas = np.array([1.5, -0.5, 2.0, 0.8, -1.2]).reshape(n, 1) # Ensure it's a column vector

# Construct the 'true' X_prime for generating y
true_X_prime = np.hstack((np.ones((m, 1)), X_data))
true_theta = np.vstack((true_beta_0, true_betas))

# Generate y with some noise
noise = 2 * np.random.randn(m, 1) # Gaussian noise
y_data = true_X_prime @ true_theta + noise

print(f"Generated {m} samples with {n} features.\n")

# --- Run the Normal Equation implementation ---
start_time = time.time()
predicted_theta = implement_normal_equation_numpy(X_data, y_data)
end_time = time.time()

if predicted_theta is not None:
    print("Calculated Parameters (theta):")
    print(predicted_theta)
    print(f"\nTrue Parameters (theta):")
    print(true_theta)
    print(f"\nTime taken for NumPy implementation: {end_time - start_time:.6f} seconds")

    # --- Verification: Compare with scikit-learn's LinearRegression ---
    from sklearn.linear_model import LinearRegression

    print("\n--- Verification with scikit-learn ---")
    start_time_sklearn = time.time()
    model = LinearRegression(fit_intercept=True) # fit_intercept=True means it will calculate beta_0
    model.fit(X_data, y_data)
    end_time_sklearn = time.time()

    print(f"Scikit-learn Intercept (beta_0): {model.intercept_[0]:.4f}")
    print(f"Scikit-learn Coefficients (beta_1 to beta_n):")
    print(model.coef_[0])
    print(f"Time taken for scikit-learn: {end_time_sklearn - start_time_sklearn:.6f} seconds")

    # Note: Scikit-learn separates intercept and coefficients.
    # Our theta is [intercept, coef1, coef2, ...]
    # So, model.intercept_ should match predicted_theta[0]
    # And model.coef_ should match predicted_theta[1:]

    # You'll notice the coefficients are very close!