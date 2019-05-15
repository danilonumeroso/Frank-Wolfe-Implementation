import numpy as np

# Black magic through Pandas dataframes.
# Basically I'm building the lists out of pandas dataframes.
# It just works (I hope so).
def get_data(graph_filename, costs_filename, n_arcs):
    import pandas
    df = pandas.read_csv(graph_filename, sep=' ', skiprows=23,index_col=False, names=["1","2", "3", "4", "5", "6"])

    arcs_df     = df[df["1"] != "n"][["2","3", "5"]].astype(int)
    nodes_df    = df[df["1"] == "n"][["2","3"]].astype(int)
    costs_df    = pandas.read_csv(costs_filename, sep=' ', skiprows=1, header=-1).drop(axis=0,columns=int(n_arcs)).values
    
    start_nodes = (arcs_df["2"]-1).tolist()
    end_nodes   = (arcs_df["3"]-1).tolist()
    capacities  = arcs_df["5"].tolist()
    fixed_costs = costs_df[0]
    quadratic_costs = costs_df[1]
    supplies    = [0] * nodes_df["2"].max()
    arcs        = [None] * len(start_nodes)

    for i in range(len(start_nodes)):
        arcs[i] = [start_nodes[i], end_nodes[i]]

    for row in nodes_df.values:
        supplies[row[0]-1] = int(row[1])
    
    return arcs, capacities, supplies, fixed_costs, quadratic_costs

# Building and setting up Cplex MCF linear solver
def build_graph_cplex(arcs, l, u, costs, b):
    import cplex
    mcfp = cplex.Cplex()

    mcfp.set_results_stream(None) # Suppressing prints
    
    mcfp.objective.set_sense(mcfp.objective.sense.minimize)
    mcfp.linear_constraints.add(rhs=b)
    
    arcs = [[arcs[i], [1.0, -1.0]] for i in range(len(arcs))]
    
    try:
        l_ = l.ravel().tolist()[0]
        u_ = u.ravel().tolist()[0]
    except:
        l_ = l
        u_ = u

    mcfp.variables.add(obj=[int(i) for i in costs], lb=l_, ub=u_, columns=arcs)
    
    return mcfp

# Invokes cplex.mcfp.solve() and returns the result.
def solve_cplex(mcfp):
    mcfp.parameters.lpmethod.set(mcfp.parameters.lpmethod.values.network)
    mcfp.solve()

    y = mcfp.solution.get_values()
    return np.matrix(y).T

# Building and setting up Google's solver
def _build_graph_ortools(arcs, l, u, c, b):
    from ortools.graph import pywrapgraph

    mcfp = pywrapgraph.SimpleMinCostFlow()

    c = [int(i) for i in c] # Ortools does not like very much float costs
    
    # Adding arcs
    for i in range(len(arcs)):
        start_node, end_node = arcs[i]
        mcfp.AddArcWithCapacityAndUnitCost(start_node, end_node, u[i], c[i])

    # Adding node supplies
    for i in range(len(b)):
        mcfp.SetNodeSupply(i, b[i])

    return mcfp

# Invokes Ortools.mcfp.solve() and returns the result
def _solve_ortools(mcfp):
    mcfp.Solve()
    y = np.zeros(mcfp.NumArcs(), dtype=int)

    for i in range(mcfp.NumArcs()):
        y[i] = mcfp.Flow(i)
    
    return np.transpose(np.matrix(y))


# If you want to use ORTools' solver
# you need to change the following lines.
# Watch out! You can implement stabilization
# only using Cplex's solver, because ORTools MCFP
# doesn't let you to specify the lower bound on 
# the arc capacities. That's why we had to switch
# to CPlex although the interface is more complicated
# than ORTools one.

build_graph=build_graph_cplex
solve=solve_cplex
