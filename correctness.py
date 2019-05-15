import numpy as np

"""
Ensures that the solution 'x' did not violate any constraints.
#1 0 \leq x \leq u
#2 E*x = b 
"""

def check(number_of_nodes, number_of_arcs, x, supplies, capacities, arcs):

    print("Checking correctness -> begin")
    # Building node arc incidence matrix E
    E = np.zeros((number_of_nodes, number_of_arcs), dtype=int)

    for i in range(number_of_arcs):
        u,v = arcs[i]
        E[u,i] = 1
        E[v,i] = -1
        
    E = np.matrix(E)
    # End E

    # b1 obtained as E*x
    b1 = (E*np.matrix(x)).round().astype(int)

    # real b reshaped as a column-vector
    b = np.transpose(np.matrix(supplies))

    # Check constraint #1
    for i in range(number_of_arcs):
        assert x[i] <= capacities[i], "x_i: " + x[i] + " > " + capacities[i]
        assert x[i] >= 0, "x_i < 0"

    # Check constraint #2
    for i in range(number_of_nodes):
        assert b1[i] == b[i]

    print("Checking correctness -> passed")
