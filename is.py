import networkx as nx
import dwave_networkx as dnx
import networkx.algorithms.approximation as nx_app

import matplotlib.pyplot as plt
import math

import numpy as np

import logging 
import time

import threading

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

def f(n, p):
    g = nx.binomial_graph(n, p)
    approxRes = nx_app.maximum_independent_set(g) # O(n/(math.log(n)**2)) approximate
    logger.info("got approximate res for n=%d, p=%f, res=%d", n, p, len(approxRes))
    res, _ = bruteforce_maximum_independent_set(g, (n/(math.log(n)**2))*len(approxRes))
    logger.info("got res for n=%d, p=%f, res=%d", n, p, len(res))
    return len(res)/len(approxRes)


def aSize(n)->int:
    if n < 11:
        return 30
    if n < 14:
        return 10
    if n < 17:
        return 3
    if n < 23:
        return 2
    return 1

flag = True
def thread_function():
    global flag
    s = input()
    flag = False

if __name__ == "__main__":
    start_time = time.perf_counter()
    nCap = 35
    Ps = [0.5]

    listener = threading.Thread(target=thread_function)
    listener.start()  # start to listen on a separate thread
    for p in Ps:
        n = 3
        arr = []
        expectedArr = []
        while flag:
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
        ax.plot(range(n-3), arr)
        ax.plot(range(n-3), expectedArr)
        ax.set_title('p='+str(p))

    plt.show()
