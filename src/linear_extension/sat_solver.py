# Copyright (c) 2024 Maximilian Wittmann, MIT License
# Repository: https://github.com/maxitw/productdim

from sage.all import Poset

def strict_upset(P,x):
    """Calculates the upset of x in the partial order P (without x)"""
    S = set(P.order_filter([x]))
    S.discard(x)
    return S

def strict_downset(P,x):
    """Calculates the downset of x in the partial order P (without x)"""
    S = set(P.order_ideal([x]))
    S.discard(x)
    return S

def critical_pairs(P):
    """Calculates the critical pairs of a partial order P"""
    G = P.incomparability_graph()
    pairs = G.edges(sort=False, labels=False)
    criticals = []
    for (x,y) in pairs:
        both = [(x,y),(y,x)]
        for (x,y) in both:
            upx = strict_upset(P,x)
            upy = strict_upset(P,y)
            dox = strict_downset(P,x)
            doy = strict_downset(P,y)
            if dox <= doy and upy <= upx:
                criticals.append((x,y))
    return criticals

def sat_dimension(poset, certificate=False):
    """Calculates the dimension of a poset.

    Uses a SAT formulation to calculate the dimension of a partially ordered set.
    
    Arguments:
    poset: A SageMath Poset
    certificate: If certificate=True, return a realizer in addition to the dimension.
    """
    k = 1

    p = Poset(poset._hasse_diagram)
    elements = p.list()

    #identify incomparable pairs with variables
    inc_graph = p.incomparability_graph()
    inc_pairs = inc_graph.edges(sort=True, labels=False)
    n_inc = len(inc_pairs)
    to_variables = dict((pair, i) for (i, pair) in enumerate(inc_pairs, start=1))



    def get_var(a, b):
        #gets the corresponding variable for one ordered incomparable pair
        val = to_variables.get((a, b), None)
        if not val is None:
            return val

        val_flip = to_variables.get((b, a), None)
        if not val_flip is None:
            return -val_flip

        #return None if comparable
        return None

    # generate the clauses to form a linear extension
    clauses = []
    for ((a,c), ac_var) in to_variables.items():
        for b in elements:
            new_clause = []

            ab_var = get_var(a, b)
            if not ab_var is None:
                new_clause.append(-ab_var)
            elif (not p.is_less_than(a, b)):
                continue

            bc_var = get_var(b, c)
            if not bc_var is None:
                new_clause.append(-bc_var)
            elif (not p.is_less_than(b, c)):
                continue

            new_clause.append(ac_var)
            clauses.append(new_clause)


    from sage.sat.solvers.satsolver import SAT
    sign = lambda x: 1 if x > 0 else -1
    def build_sat(k):
        sat = SAT(solver="kissat")
        #add clauses to find k linear extensions
        for i in range(k):
            modifier = i * n_inc
            for clause in clauses:
                new_clause = [var + sign(var)*modifier for var in clause]
                sat.add_clause(new_clause)
        #clauses to ensure that each incomparable pair appears in both orders
        for var in range(1, n_inc+1):
            sat.add_clause([var + i*n_inc for i in range(k)])
            sat.add_clause([-var - i*n_inc for i in range(k)])

        return sat

    while True:
        #attempt to find realizer for each k

        sat = build_sat(k)
        result = sat()
        if result is not False:
            break

        k += 1

    #return the dimension
    if not certificate:
        return k

    #construct the realizer from the sat solution
    diagram = p.hasse_diagram()
    realizer = []
    for i in range(k):
        extension = diagram.copy()
        for var in range(1, n_inc+1):
            (a,b) = inc_pairs[var-1]
            if result[var + i*n_inc]:
                extension.add_edge(a, b)
            else:
                extension.add_edge(b, a)
        realizer.append(extension.topological_sort())

    return (k, [[poset._list[i] for i in l] for l in realizer])

