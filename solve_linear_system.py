import numpy as np

def print_linear_system(a, b):
    assert a.shape[0] == a.shape[1]
    assert a.shape[0] == b.shape[0]
    assert b.shape[1] == 1

    dim = a.shape[0]
    
    for i in range(dim):
        for j in range(dim):
            print(a[i][j], end="\t")
        print("=\t", b[i][0])
    print()

def swap_line(matrix, i, j):
    matrix[i], matrix[j] =  matrix[j].copy(), matrix[i].copy()
    return matrix

def solve_linar_system(a, b, verbose = False):
    if verbose:
        print("System to solve")
        print_linear_system(a, b)
    assert a.shape[0] == a.shape[1]
    assert a.shape[0] == b.shape[0]
    assert b.shape[1] == 1

    dim = a.shape[0]
    
    for i in range(dim):
        non_null_line = None
        for line in range(i, dim):
            if np.not_equal(a[line][i], 0):
                non_null_line = line
                break
        if i != non_null_line:
            a = swap_line(a, i, non_null_line)
            b = swap_line(b, i, non_null_line)
            if verbose:
                print(f"Swap line {i} and {non_null_line}")
                print_linear_system(a, b)
            
        for line in range(i + 1, dim):
            times_factor = a[line][i] / a[i][i]  
            a[line] = a[line] -  times_factor * a[i]  
            b[line] = b[line] -  times_factor * b[i]  
            if verbose:     
                print(f"Line {i} is added to line {line} times { - times_factor}")
                print_linear_system(a, b)

    for i in range(dim - 1, -1, -1):
        for line in range(i):
            times_factor = a[line][i] / a[i][i]  
            a[line] = a[line] -  times_factor * a[i]    
            b[line] = b[line] -  times_factor * b[i]  
            if verbose:     
                print(f"Line {i} is added to line {line} times { - times_factor}")
                print_linear_system(a, b)

    for i in range(dim):
        division_factor = a[i][i]
        a[i] = a[i] / division_factor
        b[i] = b[i] / division_factor 
        if verbose:
            print(f"Line {i} is divided by {division_factor}")
            print_linear_system(a, b)
    
    return b

a = np.array([[0,2,3],
              [2,3,45],
              [33,5,6]], dtype="float")
b = np.array([[1],
              [2],
              [3]], dtype="float")

x = solve_linar_system(a, b, verbose = True)
print(a @ x - b)