

import matplotlib.pyplot as plt

import time
import logging

import random
import numpy as np
import cvxpy as cp

logger = logging.getLogger()
logging.basicConfig(level=logging.INFO)

numOfVarCAP = 10
rangeCap = 999999999
def create_random_problem(n = None):
    num_of_variables = random.randint(1, numOfVarCAP) if n == None else n # this is also the amount of equations needed to solve
    
    equations = []
    solutions = []
    for i in range(0, num_of_variables):
        currEqu = []
        for j in range(0, num_of_variables):
            factor = random.randint(-rangeCap, rangeCap)
            currEqu.append(factor)
        equations.append(currEqu)

    for i in range(0, num_of_variables):
        solutions.append(random.randint(-rangeCap, rangeCap))
    
    return equations, solutions

def convert_to_cp(equationslist, solutionsList):
    a = np.array(equationsList)
    b = np.array(solutionsList)
    return a, b

def solve_with_np(a,b):
    """ Gets a list of equations and list of solutions (parameters) for this equations and solves using numpy
    >>> solve_with_np([[1, 2], [3, 5]], [1, 2])
    array([-1.,  1.])
    >>> solve_with_np([[-968, 3595, -3626, 827], [86, -2863, -2246, 1005], [4204, -3842, 4891, -3975], [-1133, 3360, -1635, 687]],[3342, 504, -3428, 416])
    array([-22.03858214,  -5.03154322,  -5.61177412, -24.48759685])
    """
    results = np.linalg.solve(a, b)
    return results

def convert_to_cp(equationsList, solutionsList):
    ExpList = [cp.Expression]*len(equationsList)
    VarList = cp.Variable(len(equationsList))
    for i in range(len(equationsList)):
        ExpList[i] = equationsList[i][0]*VarList[0]
        for j in range(1, len(equationsList)):
            ExpList[i] += equationsList[i][j]*VarList[j]
    constraints=[item == solutionsList[i] for i, item in enumerate(ExpList)] # for each constraint, compare it to the parameter
    return constraints, VarList

def solve_with_cp(constraints, VarList):
    results = []
    x = 0
    for i in VarList:
        x += i
    prob = cp.Problem(objective=cp.Maximize(x), constraints=constraints)
    res = prob.solve()
    results.append(res)
    # for i in range(len(constraints)):
    #     prob = cp.Problem(objective=cp.Minimize(VarList[i]), constraints=constraints)
    #     res = prob.solve()
    #     results.append(res)
    return results

if __name__ == "__main__":
    import doctest
    doctest.testmod()

    npTimes = []
    cpTimes = []
    n = 1
    for n in range(2,125):
        equationsList, solutionsList = create_random_problem(n)
        start_time = time.perf_counter()
        npsolve = solve_with_np(equationsList, solutionsList)
        end_time = time.perf_counter()
        npTimes.append(end_time-start_time)

        constraints, VarList = convert_to_cp(equationsList, solutionsList)
        start_time = time.perf_counter()
        cpsolve = solve_with_cp(constraints, VarList)
        end_time = time.perf_counter()
        cpTimes.append(end_time-start_time)
        logger.info("finished calculating for %d", n)

    fig, ax = plt.subplots()
    end_time = time.perf_counter()

    ax.plot(range(n-1), cpTimes)
    ax.plot(range(n-1), npTimes)
    ax.set_title('p=')

    plt.show()






