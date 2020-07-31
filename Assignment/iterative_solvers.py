import numpy as np

''' Jacobi and Gauss-Siedel iterative schemes to solve Ax = b'''

### JACOBI method ###

def jacobi_solver(A,b,x0,num_itr):

    D = np.diag(A)
    R = A - np.diagflat(D)
    x = x0

    for i in range(num_itr):

        temp = (b - np.dot(R,x))/ D
        x = temp

    return x

### GAUSS-SIEDEL method ###

def seidel_solver(A,b,x0,num_itr):

    x = x0
    N = len(A)
    for j in range(0, N):
        temp = b[j]
        for i in range(0, N):
            if(j != i):
                temp -= A[j][i] * x[i]

        x[j] = temp / A[j][j]
    return x

### MAIN CODE ###

# Enter the dimension of the matrix A
N = 5

# Assign values for the matrix A
A = np.zeros((N,N))
for i in range(N):
    for j in range(N):
        if i == j:
            A[i,j] = N*(i+j+2)
        else:
            A[i,j] = (i+j+2)/N

# Entries of the b vector
b = np.random.random(N)

# Random guess for the solution
x0 = np.zeros(N)

x = jacobi_solver(A,b,x0,num_itr = 200)
y = seidel_solver(A,b,x0,num_itr = 200)

# Print statements
print("A =",np.matrix(A),"\n")
print("b =",np.matrix(b),"\n")

print("Initial Guess =", np.zeros(N),"\n")

print("Solution from Jacobi method =", x)
print("Solution from Gauss-Siedel method =",y)
print('Solution from built-in functions = %s' % np.linalg.solve(A, b),"\n")

print("Error in Jacobi method =",np.linalg.norm(x-np.linalg.solve(A,b)))
print("Error in Gauss-Siedel method =",np.linalg.norm(y-np.linalg.solve(A,b)))
