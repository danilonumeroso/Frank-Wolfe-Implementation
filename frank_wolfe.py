import numpy as np
import time
import sys
import correctness
from ortools.graph import pywrapgraph
from mcfp import build_graph, solve, get_data

"""
Function implementing Frank Wolfe procedure.
Frank Wolfe has been used to minimize a quadratic
Min Cost Flow Problem. At each iteration it resolves
the linear approximation of the master problem, and iterates.

min_x { (1/2)*x^T*Q*x + q^t*x, subject to Ex = b, 0 <= x <= u }

x \in R^n will eventually results in a flow distribution on each arc of the graph.

Q: matrix (n x n), positive semidefinite, containing quadratic costs on its diagonal
q: vector (1 x n), containing linear costs
E: matrix (m x n), where m is the number of nodes in the graph.
b: vector (1 x m), supplies/demands of the nodes in the graph.
u: vector (1 x n), capacities of each arc (max flow that an arc can contain)

e:              scalar , represents the precision of the solutioan
stabilization:  boolean, whether stabilization is used or not 
max_iter:       scalar , maximum number of iterations permitted
"""
def fwmcfp(Q, q, arcs, u, b, t, eps=1e-6, stabilization=True, max_iter=1000):

    #- begin initialization
    best_lower_bound = -np.inf
    x  = np.transpose(np.matrix([int(i/2) for i in u]))
    y  = x
    l  = np.zeros_like(u)
    q  = np.matrix(q) # linear costs
    Qc = np.matrix(np.diag(Q)) #quadratic costs (matrix)
    
    # Objective function
    f = lambda x: 0.5 * np.transpose(x) * Qc * x + q*x
    # Gradient
    g = lambda x: Qc * x + np.transpose(q)

    #- end initializationa

    i = 1
    l_hat = l
    u_hat = u

    print("Frank-Wolfe method")
    print("iter\tf(x)\t\tlb\t\tgap")
    
    while True:
        
        value = f(x)
        gradient = g(x)
        
        if stabilization:
            mcfp = build_graph(arcs, l_hat, u_hat, gradient, b)
            y = solve(mcfp)
            l_hat = np.maximum(l, (y-t).T)
            u_hat = np.maximum(u, (y+t).T)
        else:
            mcfp = build_graph(arcs, l, u, gradient, b)
            y = solve(mcfp)
        
        lower_bound = value + np.dot(np.transpose(gradient), (y-x))
        
        if lower_bound > best_lower_bound:
            best_lower_bound = lower_bound

        gap = (value - best_lower_bound)/max(abs(value), 1)
        
        print("%s\t%7.4f\t%7.4f\t%2.7f" %(i , value , best_lower_bound , gap))

        if gap <= eps:
            status = "optimal"
            return x, status, gap, i

        if i > max_iter:
            status = "stopped"
            return x, status, gap, i   
 
        # Computing direction
        d = np.matrix(y - x)

        den = np.transpose(d) * Qc * d

        try: # Handling DivisionByZero exception
            alpha = min([((-np.transpose(gradient) * d) / den).item(0), 1])  
            # Line search
        except:
            alpha = 1

        x = x + alpha * d
    
        i = i + 1
        
        
# Loading examples

test_file = sys.argv[1] # <path/to/filename.[dmx/qfc]

graph_fn = test_file + ".dmx"
costs_fn = test_file + ".qfc"

arcs, capacities, supplies, fixed_costs, quadratic_costs = get_data(graph_fn, costs_fn, sys.argv[2])

t = int(sys.argv[3])

start = time.time()

x, status, gap, i = fwmcfp(
                        quadratic_costs,
                        fixed_costs,
                        arcs,
                        capacities,
                        supplies,
                        t,
                        stabilization=True,
                        max_iter=10000
                    )

end = time.time()

correctness.check(len(supplies), len(arcs), x, supplies, capacities, arcs)

print("Execution time: %s" %(end-start))
print("Last gap %s: %s" %(status, gap.item(0)))
print("n iter %s: %s" %(status, i))
print("Status: %s" %(status))

#uncomment the following line to get the solution
#print("x: %s" %(x))


