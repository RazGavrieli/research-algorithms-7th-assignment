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
    """
    Gets an n value and returns a linear equations represented in a matrix of factors and an array of solutions.
    >>> len(create_random_problem(n = 3)[0])
    3
    """
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
    ExpList = [cp.Expression()]*len(equationsList)
    VarList = cp.Variable(len(equationsList))
    for i in range(len(equationsList)):
        ExpList[i] = equationsList[i][0]*VarList[0]
        for j in range(1, len(equationsList)):
            ExpList[i] += equationsList[i][j]*VarList[j]
    constraints=[item == solutionsList[i] for i, item in enumerate(ExpList)] # for each constraint, compare it to the parameter
    # constraints = [3*x + y == 1, x + y <= 2]
    return constraints, VarList

def solve_with_cp(constraints, VarList, obj):
    """
    Gets a list of constraints (cp.Expression) and a list of variables (cp.Variable) and returns a list of results to the linear equations
    >>> solve_with_cp(*convert_to_cp([[1, 2], [3, 5]], [1, 2]), obj=cp.Maximize(0))
    [-0.9999999999989323, 0.999999999999404]
    >>> solve_with_cp(*convert_to_cp([[-968, 3595, -3626, 827], [86, -2863, -2246, 1005], [4204, -3842, 4891, -3975], [-1133, 3360, -1635, 687]],[3342, 504, -3428, 416]), obj=cp.Maximize(0))
    [-22.038582135559295, -5.031543216314029, -5.6117741168788084, -24.48759684691261]
    """
    results = []
    prob = cp.Problem(objective=obj, constraints=constraints)
    prob.solve()
    results = [item.value for item in VarList]
    return results

if __name__ == "__main__":
    import doctest
    doctest.testmod()

    npTimes = []
    cpTimes = []
    n = 1
    for n in range(2,250):
        currtime = 0
        avg = 0
        for _ in range(0,5):
            equationsList, solutionsList = create_random_problem(n)
            start_time = time.perf_counter()
            npsolve = solve_with_np(equationsList, solutionsList)
            end_time = time.perf_counter()
            avg += end_time-start_time
        avg /= 5
        npTimes.append(avg)

        # constraints, VarList = convert_to_cp(equationsList, solutionsList)
        # start_time = time.perf_counter()
        # cpsolve = solve_with_cp(constraints, VarList, cp.Minimize(0))
        # end_time = time.perf_counter()
        # cpTimes.append(end_time-start_time)
        logger.info("finished calculating for n=%d", n)

        #logger.info("\nnp solutions:\n" + str(list(npsolve)) + "\ncp solutions:\n" + str(cpsolve) + "\n" + str(np.allclose(np.dot(equationsList, npsolve), solutionsList)) + "\n---------------------")

    fig, ax = plt.subplots()
    end_time = time.perf_counter()

    # ax.plot(range(n-1), cpTimes, 'green')
    ax.plot(range(n-1), npTimes, 'orange')
    ax.set_title('Only NP (for high n)')

    plt.show()






