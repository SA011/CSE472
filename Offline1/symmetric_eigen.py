import numpy as np
import warnings 
  
  
n = int(input())

# Create a matrix of size n x n
matrix = np.zeros((n, n))

# Maximum number of iterations
max_iter = 100

# Fill the matrix with random numbers
while np.linalg.det(matrix) == 0 and max_iter > 0:
    # If the matrix is singular, then fill it with random numbers again
    matrix = np.random.randint(0, 10, (n, n))
    matrix = (matrix + matrix.T)//2
    max_iter -= 1
assert np.allclose(matrix, matrix.T)

# If the matrix is still singular, then exit the program
if np.linalg.det(matrix) == 0:
    print("Can not create a matrix with non-zero determinant")
    exit()

# Print the matrix
print("Matrix:")
print(matrix)
print()


# Find the eigenvalues and eigenvectors of the matrix
eigenvalues, eigenvectors = np.linalg.eigh(matrix)

# Print the eigenvalues and eigenvectors
print("Eigenvalues: ", eigenvalues)
print("Eigenvectors: ", eigenvectors)
print()


# check eigenvalue-eigenvector equation
# for i in range(eigenvalues.shape[0]):
#     assert np.allclose(matrix.dot(eigenvectors[:,i]), eigenvalues[i] * eigenvectors[:,i])

# Reconstruct the matrix from the eigenvalues and eigenvectors
reconstructedMatrix = eigenvectors.dot(np.diag(eigenvalues)).dot(eigenvectors.T)


print("Original Matrix:")
print(matrix)
print()


print("Reconstructed matrix:")
# Settings the warnings to be ignored
warnings.filterwarnings('ignore') 
reconstructedMatrix = reconstructedMatrix.round().astype(int) 
print(reconstructedMatrix)
print()


# Check if reconstructed matrix is equal to the original matrix
assert np.allclose(matrix, reconstructedMatrix)

print ('Reconstructed Matrix is equal to the original matrix. Task 2B end.')

