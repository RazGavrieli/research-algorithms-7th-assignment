import networkx as nx
import dwave_networkx as dnx
import networkx.algorithms.approximation as nx_app

import matplotlib.pyplot as plt
import math

import logging 
import time


logger = logging.getLogger()
logging.basicConfig(level=logging.INFO)

def powerset(iterable):
    from itertools import chain, combinations
    s = list(iterable)
    return chain.from_iterable(combinations(s, r) for r in range(len(s)+1))

def bruteforce_maximum_independent_set(g, cap):
    max_independent_set_size = -1
    max_independent_set = []
    for i in powerset(range(len(nx.nodes(g)))):
        if dnx.is_independent_set(g, i):
            max_independent_set = i
            max_independent_set_size = len(i)
            if max_independent_set_size >= cap:
                logger.info("brute force functio got to CAP size %f. stopping brute force..", cap)
                break

    return max_independent_set, max_independent_set_size

def create_is(g: nx.Graph(), k: int):
    """
    Gets a graph g and a number k < len(g.nodes)
    and removes edges from g so that the first k nodes of graph g will be an independent set. 
    """
    if len(g.nodes()) <= k:
        raise ValueError("k is too big")
    for u in range(k):
        for v in range(k):
            if u != k:
                g.remove_edges_from([(u, v)])

def f(n, p):
    g = nx.binomial_graph(n, p)

    #originalApproxSize = len(nx_app.maximum_independent_set(g)) # O(n/(math.log(n)**2)) approximate
    #k = int(n/(math.log(n)**2)*originalApproxSize)
    k = int(n/6)
    # k = n
    # while k > originalApproxSize:
    #     k/=2
    # k = int(k)
    #logger.info("k=%d, len(res)=%d", k, originalApproxSize)

    create_is(g, k)
    approxRes = nx_app.maximum_independent_set(g) # O(n/(math.log(n)**2)) approximate
    #k, _ = bruteforce_maximum_independent_set(g, (n/(math.log(n)**2))*len(approxRes))

    logger.info("got approximate res for n=%d, p=%f, res=%d, k=%d", n, p, len(approxRes), k)
    return k/len(approxRes)


def aSize(n)->int:
    if n < 70:
        return 5
    else: return 1

if __name__ == "__main__":
    start_time = time.perf_counter()
    nCap =180
    startn = 20
    Ps = [0.25, 0.5, 0.75]

    for p in Ps:
        n = startn
        arr = []
        expectedArr = []
        while n < nCap:
            n+=1
            avg = 0 
            avgSize = aSize(n)
            for i in range(avgSize):
                avg += f(n,p)
            avg/=avgSize
            arr.append(avg)
            expectedArr.append(n/(math.log(n)**2))
        fig, ax = plt.subplots()
        end_time = time.perf_counter()

        print(arr, end_time-start_time)
        ax.plot(range(startn, n), arr)
        ax.plot(range(startn, n), expectedArr)
        ax.set_title('p='+str(p))

    plt.show()
