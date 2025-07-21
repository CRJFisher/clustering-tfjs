import numpy as np

n = 60  # number of samples

# Simple constant vector
const_simple = np.ones(n) / np.sqrt(n)
print(f"Simple constant 1/sqrt(n):")
print(f"  Value: {const_simple[0]:.10f}")
print(f"  L2 norm: {np.linalg.norm(const_simple):.10f}")

# Apply diffusion scaling sqrt(1 - eigenvalue) where eigenvalue = 0
scaled_simple = const_simple * np.sqrt(1 - 0)
print(f"\nAfter diffusion scaling:")
print(f"  Value: {scaled_simple[0]:.10f}")
print(f"  L2 norm: {np.linalg.norm(scaled_simple):.10f}")

# sklearn's value
sklearn_val = 0.0264185062
print(f"\nsklearn's value: {sklearn_val:.10f}")
print(f"Ratio to simple constant: {sklearn_val / const_simple[0]:.10f}")

# Maybe sklearn applies some other normalization
# Let's check what could give us that ratio
ratio = sklearn_val / const_simple[0]
print(f"\nThe ratio {ratio:.6f} is close to:")
print(f"  1/sqrt(2*pi) = {1/np.sqrt(2*np.pi):.6f}")
print(f"  sqrt(1/n) = {np.sqrt(1/n):.6f}")
print(f"  1/(2*sqrt(n)) = {1/(2*np.sqrt(n)):.6f}")
print(f"  0.2 = {0.2:.6f}")
print(f"  1/sqrt(n) * sqrt(1/n) = {1/n:.6f}")

# Actually, let's check if it's related to the sum of degrees
# In the RBF case, degrees might sum to something specific
print(f"\nActually, the exact ratio is: {ratio:.10f}")
print(f"This might be related to the eigenvector normalization in spectral_embedding")