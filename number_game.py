'''

In the Letters and Numbers (L&N) game,
One contestant chooses how many "small" and "large" numbers they would like 
to make up six randomly chosen numbers. Small numbers are between 
1 and 10 inclusive, and large numbers are 25, 50, 75, or 100. 
All large numbers will be different, 
so at most four large numbers may be chosen. 


How to represent a computation?

Let Q = [q0, q1, q2, q3, q4, q5] be the list of drawn numbers

The building blocks of the expression trees are
 the arithmetic operators  +,-,*
 the numbers  q0, q1, q2, q3, q4, q5

We can encode arithmetic expressions with Polish notation
    op arg1 arg2
where op is one of the operators  +,-,*

or with expression trees:
    (op, left_tree, right_tree)
    
Recursive definition of an Expression Tree:
 an expression tree is either a 
 - a scalar   or
 - a binary tree (op, left_tree, right_tree)
   where op is in  {+,-,*}  and  
   the two subtrees left_tree, right_tree are expressions trees.

When an expression tree is reduced to a scalar, we call it trivial.


Author: f.maire@qut.edu.au

Created on April 1 , 2021
    

This module contains functions to manipulate expression trees occuring in the
L&N game.

'''



import numpy as np
import random

import copy # for deepcopy

import collections

import multiprocessing
# from genetic_algorithm import evolve_pop


SMALL_NUMBERS = tuple(range(1,11))
LARGE_NUMBERS = (25, 50, 75, 100)


# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -


def my_team():
    '''
    Return the list of the team members of this assignment submission as a list
    of triplet of the form (student_number, first_name, last_name)
    
    '''
    return [ (9960392, 'Liam', 'Hulsman-Benson'), (10077413, 'Alexander', 'Farrall')]


# ----------------------------------------------------------------------------

def pick_numbers():
    '''    
    Create a random list of numbers according to the L&N game rules.
    
    Returns
    -------
    Q : int list
        list of numbers drawn randomly for one round of the game
    '''
    LN = set(LARGE_NUMBERS)
    Q = []
    for i in range(6):
        x = random.choice(list(SMALL_NUMBERS)+list(LN))
        Q.append(x)
        if x in LN:
            LN.remove(x)
    return Q


# ----------------------------------------------------------------------------

def bottom_up_creator(Q):
    '''
    Create a random algebraic expression tree
    that respects the L&N rules.
    
    Warning: Q is shuffled during the process

    Parameters
    ----------
    Q : non empty list of available numbers
        

    Returns  T, U
    -------
    T : expression tree 
    U : values used in the tree

    '''
    n = random.randint(1,6) # number of values we are going to use
    
    random.shuffle(Q)
    # Q[:n]  # list of the numbers we should use
    U = Q[:n].copy()
    
    if n==1:
        # return [U[0], None, None], [U[0]] # T, U
        return U[0], [U[0]] # T, U
        
    F = [u for u in U]  # F is initially a forest of values
    # we start with at least two trees in the forest
    while len(F)>1:
        # pick two trees and connect then with an arithmetic operator
        random.shuffle(F)
        op = random.choice(['-','+','*'])
        T = [op,F[-2],F[-1]]  # combine the last two trees
        F[-2:] = [] # remove the last two trees from the forest
        # insert the new tree in the forest
        F.append(T)
    # assert len(F)==1
    return F[0], U
  
# ---------------------------------------------------------------------------- 

def display_tree(T, indent=0):
    '''
    
    Eval the algebraic expression represented by T
    
    Parameters
    ----------
    T : Expression Tree
    indent: indentation for the recursive call

    Returns None

    '''
    # if T is a scalar, then we return it directly
    if isinstance(T, int):
        print('|'*indent,T, sep='')
        return
    # T is non trivial
    root_item = T[0]
    print('|'*indent, root_item, sep='')
    display_tree(T[1], indent+1)
    print('|'*indent)
    display_tree(T[2], indent+1)
   
# ---------------------------------------------------------------------------- 

def eval_tree(T):
    '''
    
    Eval the algebraic expression represented by T
    
    Parameters
    ----------
    T : Expression Tree

    Returns
    -------
    value of the algebraic expression represented by the T

    '''
    # if T is a scalar, then we return it directly
    if isinstance(T, int):
        return T
    # T is non trivial
    root_item = T[0]
    # assert root_item in ('-','+','*')
    left_value = eval_tree(T[1])
    right_value = eval_tree(T[2])
    return eval( str(left_value) +root_item + str(right_value) )
    # return eval(root_item.join([str(left_value), str(right_value)]))
   
     
# ---------------------------------------------------------------------------- 

def expr_tree_2_polish_str(T):
    '''
    Convert the Expression Tree into Polish notation

    Parameters
    ----------
    T : expression tree

    Returns
    -------
    string in Polish notation represention the expression tree T

    '''
    if isinstance(T, int):
        return str(T)
    root_item = T[0]
    # assert root_item in ('-','+','*')
    left_str = expr_tree_2_polish_str(T[1])
    right_str = expr_tree_2_polish_str(T[2])
    return '[' + ','.join([root_item,left_str,right_str]) + ']'
    

# ----------------------------------------------------------------------------

class ExpNode:
    """Class for a node in Expression tree."""
    def __init__(self, val):
        self.val = val
        self.left = None
        self.right = None

    def toFormat(self):
        return [self.val, self.left, self.right]


def expTreeTraverse(root):
    tempQ = []
    tempQ.append(root)

    for node in tempQ:
        if node.left != None:
            tempQ.append(node.left)
        if node.right != None:
            tempQ.append(node.right)


    return tempQ



def polish_str_2_expr_tree(pn_str):
    '''
    
    Convert a polish notation string of an expression tree
    into an expression tree T.

    Parameters
    ----------
    pn_str : string representing an L&N algebraic expression

    Returns
    -------
    T

    '''
    # raise NotImplementedError()
    pn_str = pn_str.split(" ")
    opStack = []
    outputTree = []

    for val in pn_str:
        if val in "*+-":
            opStack.append(val)
        else:
            outputTree.append(val)

    while len(opStack) > 0:
        outputTree.insert(0, [opStack.pop(), outputTree.pop(-2), outputTree.pop(-1)])


    return outputTree
    def find_match(i):
        '''
        Starting at position i where pn_str[i] == '['
        Return the index j of the matching ']'
        That is, pn_str[j] == ']' and the substring pn_str[i:j+1]
        is balanced
        '''
        count = 0

        for j, char in enumerate(pn_str[i+1:]):
            if char == "]" and count == 0:
                return j
            elif char == "]":
                count -= 1
            elif char == "[":
                count += 1

        return None
    # .................................................................

    left_p = pn_str.find('[')

    raise NotImplementedError()
 
   
# ----------------------------------------------------------------------------

def op_address_list(T, prefix = None):
    '''
    Return the address list L of the internal nodes of the expresssion tree T
    
    If T is a scalar, then L = []

    Note that the function 'decompose' is more general.

    Parameters
    ----------
    T : expression tree
    prefix: prefix to prepend to the addresses returned in L

    Returns
    -------
    L
    '''
    if isinstance(T, int):
        return []
    
    if prefix is None:
        prefix = []
        
    L = [prefix.copy()+[0]] # first adddress is the op of the root of T
    left_al = op_address_list(T[1], prefix.copy()+[1])
    L.extend(left_al)
    right_al = op_address_list(T[2], prefix.copy()+[2])
    L.extend(right_al)
    
    return L


# ----------------------------------------------------------------------------

def decompose(T, prefix = None):
    '''
    Compute
        Aop : address list of the operators
        Lop : list of the operators
        Anum : address of the numbers
        Lnum : list of the numbers
    
    For example, if 
    
    T =  ['-', ['+', ['-', 75, ['-', 10, 3]], ['-', 100, 50]], 3]
    
    then, 
    
     Aop is  [[0], [1, 0], [1, 1, 0], [1, 1, 2, 0], [1, 2, 0]] 
    
     Lop is ['-', '+', '-', '-', '-'] 
    
     Anum is [[1, 1, 1], [1, 1, 2, 1], [1, 1, 2, 2], [1, 2, 1], [1, 2, 2], [2]] 
    
     Lnum is [75, 10, 3, 100, 50, 3]    
        
    
    Parameters
    ----------
    T : expression tree 
    
    prefix : address to preprend 

    Returns
    -------
    Aop, Lop, Anum, Lnum

    '''
    if prefix is None:
        prefix = []

    if isinstance(T, int):
        Aop = []
        Lop = [] 
        Anum = [prefix]
        Lnum = [T]
        return Aop, Lop, Anum, Lnum
    
    assert isinstance(T, list)

    tempAop = []
    tempLop = []
    tempAnum = []
    tempLnum = []

    for i,val in enumerate(T):
        x = type(val)
        if type(val) == str or type(val) == int:
            if val in ['*', '+', '-']:
                tempAop.append(prefix+[i])
                tempLop.append(val)
            else:
                tempAnum.append(prefix+[i])
                tempLnum.append(val)
        else:
            tempReturn = decompose(val, prefix + [i])
            tempAop += tempReturn[0]
            tempLop += tempReturn[1]
            tempAnum += tempReturn[2]
            tempLnum += tempReturn[3]

    return tempAop, tempLop, tempAnum, tempLnum


# ----------------------------------------------------------------------------

def get_item(T, a):
    '''
    Get the item at address a in the expression tree T

    Parameters
    ----------
    T : expression tree
    a : valid address of an item in the tree

    Returns
    -------
    the item at address a

    '''
    if len(a)==0:
        return T
    # else
    return get_item(T[a[0]], a[1:])
        
# ----------------------------------------------------------------------------

def replace_subtree(T, a, S):
    '''
    Replace the subtree at address a
    with the subtree S in the expression tree T
    
    The address a is a sequence of integers in {0,1,2}.
    
    If a == [] , then we return S
    If a == [1], we replace the left subtree of T with S
    If a == [2], we replace the right subtree of T with S

    Returns
    ------- 
    The modified tree

    Warning: the original tree T is modified. 
             Use copy.deepcopy()  if you want to preserve the original tree.
    '''    
    
    # base case, address empty
    if len(a)==0:
        return S
    
    # recursive case
    T[a[0]] = replace_subtree(T[a[0]], a[1:], S)
    return T


# ----------------------------------------------------------------------------

def mutateIndex(arr, loc, newVal):
    if type(arr) == int or type(arr) == str:
        return newVal
    if len(arr) == 0:
        return newVal
    else:
        arr[loc[0]] = mutateIndex(arr[loc[0]],loc[1:], newVal)
    return arr

def mutate_num(T, Q):
    '''
    Mutate one of the numbers of the expression tree T
    
    Parameters
    ----------
    T : expression tree
    Q : list of numbers initially available in the game

    Returns
    -------
    A mutated copy of T

    '''
    
    Aop, Lop, Anum, Lnum = decompose(T)    
    mutant_T = copy.deepcopy(T)
        
    counter_Q = collections.Counter(Q) # some small numbers can be repeated

    mutationIndex = random.randint(0,len(Anum)-1)

    mutationLoc = Anum[mutationIndex]
    prevVal = Lnum[mutationIndex]

    #Get a different number than was there previously
    mutatedVal = random.choice(Q)
    while mutatedVal == prevVal:
        mutatedVal = random.choice(Q)

    #Adjust selected location to random value
    mutant_T = mutateIndex(mutant_T, mutationLoc, mutatedVal)

    return mutant_T
    

# ----------------------------------------------------------------------------

def mutate_op(T):
    '''
    Mutate an operator of the expression tree T
    If T is a scalar, return T

    Parameters
    ----------
    T : non trivial expression tree

    Returns
    -------
    A mutated copy of T

    '''
    Q = ['*', '+', '-']
    Aop, Lop, Anum, Lnum = decompose(T)
    mutant_T = copy.deepcopy(T)

    counter_Q = collections.Counter(Q)  # some small numbers can be repeated

    mutationIndex = random.randint(0, len(Aop) - 1)

    mutationLoc = Aop[mutationIndex]
    prevVal = Lop[mutationIndex]

    # Get a different number than was there previously
    mutatedVal = random.choice(Q)
    while mutatedVal == prevVal:
        mutatedVal = random.choice(Q)

    # Adjust selected location to random value
    mutant_T = mutateIndex(mutant_T, mutationLoc, mutatedVal)

    return mutant_T
    

# ----------------------------------------------------------------------------

def cross_over(P1, P2, Q):    
    '''
    Perform crossover on two non trivial parents
    
    Parameters
    ----------
    P1 : parent 1, non trivial expression tree  (root is an op)
    P2 : parent 2, non trivial expression tree  (root is an op)
        DESCRIPTION
        
    Q : list of the available numbers
        Q may contain repeated small numbers    
        

    Returns
    -------
    C1, C2 : two children obtained by crossover
    '''
    
    def get_num_ind(aop, Anum):
        '''
        Return the indices [a,b) of the range of numbers
        in Anum and Lum that are in the sub-tree 
        rooted at address aop

        Parameters
        ----------
        aop : address of an operator (considered as the root of a subtree).
              The address aop is an element of Aop
        Anum : the list of addresses of the numbers

        Returns
        -------
        a, b : endpoints of the semi-open interval
        
        '''
        d = len(aop)-1  # depth of the operator. 
                        # Root of the expression tree is a depth 0
        # K: list of the indices of the numbers in the subtrees
        # These numbers must have the same address prefix as aop
        p = aop[:d] # prefix common to the elements of the subtrees
        K = [k for k in range(len(Anum)) if Anum[k][:d]==p ]
        return K[0], K[-1]+1
        # .........................................................
        
    Aop_1, Lop_1, Anum_1, Lnum_1 = decompose(P1)
    Aop_2, Lop_2, Anum_2, Lnum_2 = decompose(P2)

    C1 = copy.deepcopy(P1)
    C2 = copy.deepcopy(P2)
    
    i1 = np.random.randint(0,len(Lop_1)) # pick a subtree in C1 by selecting the index
                                         # of an op
    i2 = np.random.randint(0,len(Lop_2)) # Select a subtree in C2 in a similar way
 
    # i1, i2 = 4, 0 # DEBUG    
 
    # Try to swap in C1 and C2 the sub-trees S1 and S2 
    # at addresses Lop_1[i1] and Lop_2[i2].
    # That's our crossover operation!
    
    # Compute some auxiliary number lists
    
    # Endpoints of the intervals of the subtrees
    a1, b1 = get_num_ind(Aop_1[i1], Anum_1)     # indices of the numbers in S1 
                                                # wrt C1 number list Lnum_1
    a2, b2 = get_num_ind(Aop_2[i2], Anum_2)   # same for S2 wrt C2
    
    # Lnum_1[a1:b1] is the list of numbers in S1
    # Lnum_2[a2:b2] is the list of numbers in S2
    
    # numbers is C1 not used in S1
    nums_C1mS1 = Lnum_1[:a1]+Lnum_1[b1:]
    # numbers is C2-S2
    nums_C2mS2 = Lnum_2[:a2]+Lnum_2[b2:]
    
    # S2 is a fine replacement of S1 in C1
    # if nums_S2 + nums_C1mS1 is contained in Q
    # if not we can bottom up a subtree with  Q-nums_C1mS1

    counter_Q = collections.Counter(Q) # some small numbers can be repeated
    
    d1 = len(Aop_1[i1])-1
    aS1 = Aop_1[i1][:d1] # address of the subtree S1 
    S1 = get_item(C1, aS1)

    # ABOUT 3 LINES DELETED
    d2 = len(Aop_2[i2])-1
    aS2 = Aop_2[i2][:d2] # address of the subtree S1
    S2 = get_item(C2, aS2)

    # print(' DEBUG -------- S1 and S2 ----------') # DEBUG
    # print(S1)
    # print(S2)


    # count the numbers (their occurences) in the candidate child C1
    counter_1 = collections.Counter(Lnum_2[a2:b2]+nums_C1mS1)
    
    # Test whether child C1 is ok
    if all(counter_Q[v]>=counter_1[v]  for v in counter_Q):
        # candidate is fine!  :-)
        C1 = replace_subtree(C1, aS1, S2)
    else:
        available_nums = counter_Q.copy()
        available_nums.subtract(
            collections.Counter(nums_C1mS1)
            )
        R1, _ = bottom_up_creator(list(available_nums.elements()))
        C1 = replace_subtree(C1, aS1, R1)
        
    # count the numbers (their occurences) in the candidate child C2
    counter_2 = collections.Counter(Lnum_1[a1:b1]+nums_C2mS2)
    
    # Test whether child C2 is ok
    if all(counter_Q[v]>=counter_2[v] for v in counter_Q):
        # candidate is fine!  :-)
        C2 = replace_subtree(C2, aS2, S1)
    else:
        available_nums = counter_Q.copy()
        available_nums.subtract(
            collections.Counter(nums_C2mS2)
            )
        R2, _ = bottom_up_creator(list(available_nums.elements()))
        C2 = replace_subtree(C2, aS2, R2)
    
    return C1, C2

def task2(Q, target, popSize, returnQueue, maxGens=-1):
    #Perform the genetic algorithim
    mutation_probability = 0.1
    pop_size = popSize

    params = {
        'max_num_iteration': 50,
        'population_size': 100,
        'mutation_probability': 0.1,
        'elit_ratio': 0.05,
        'parents_portion': 0.3}

    # ------------- Initialize Population ------------------------

    pop = []  # list of pairs (cost, individuals)

    for _ in range(pop_size):
        T, _ = bottom_up_creator(Q)
        cost = abs(target - eval_tree(T))
        pop.append((cost, T))

    # Sort the initial population
    # print(pop) # debug
    pop.sort(key=lambda x: x[0])

    # ------------- Loop on generations ------------------------

    # Rank of last individual in the current population
    # allowed to breed.
    rank_parent = int(params['parents_portion'] *
                      params['population_size'])

    # Rank of the last elite individual. The elite is copied unchanged
    # into the next generation.
    rank_elite = max(1, int(params['elit_ratio'] *
                            params['population_size']))



    #Looping
    outputDict = {}
    outputDict["genCount"] = 0
    outputDict["bestTree"] = [np.inf]
    outputDict["Q"] = Q
    outputDict["target"] = target

    while 1:
        # print(maxGens)
        # print(outputDict["genCount"])
        #Will only be true after 1st iteration set a maxgens value, first run through will run till out of time.
        if outputDict["genCount"] == maxGens:
            break

        outputDict["genCount"] += 1

        # Generate children
        children = []
        while len(children) < pop_size:
            # pick two parents
            (_, P1), (_, P2) = random.sample(pop[:rank_parent], 2)
            # skip cases where one of the parents is trivial (a number)
            if isinstance(P1, list) and isinstance(P2, list):
                C1, C2 = cross_over(P1, P2, Q)
            else:
                # if one of the parents is trivial, just compute mutants
                C1 = mutate_num(P1, Q)
                C2 = mutate_num(P2, Q)
            # Compute the costs of the children
            cost_1 = abs(target - eval_tree(C1))
            cost_2 = abs(target - eval_tree(C2))
            children.extend([(cost_1, C1), (cost_2, C2)])

        new_pop = pop[rank_elite:] + children

        # Mutate some individuals (keep aside the elite for now)
        # Pick randomly the indices of the mutants
        mutant_indices = random.sample(range(len(new_pop)),
                                       int(mutation_probability * pop_size))
        # i: index of a mutant in new_pop
        for i in mutant_indices:
            # Choose a mutation by flipping a coin
            Ti = new_pop[i][1]  # new_pop[i][0]  is the cost of Ti
            # Flip a coin to decide whether to mutate an op or a number
            # If Ti is trivial, we can only use mutate_num
            if isinstance(Ti, int) or random.choice((False, True)):
                Mi = mutate_num(Ti, Q)
            else:
                Mi = mutate_op(Ti)
            # update the mutated entry
            new_pop[i] = (abs(target - eval_tree(Mi)), Mi)

        # add without any chance of mutation the elite
        new_pop.extend(pop[:rank_elite])

        # sort
        new_pop.sort(key=lambda x: x[0])

        # keep only pop_size individuals
        pop = new_pop[:pop_size]

        if pop[0][0] < outputDict['bestTree'][0]:
            outputDict['bestTree'] = pop[0]

        #If value is already in queue, remove it and then add new value
        if not returnQueue.empty():
            returnQueue.get()
        returnQueue.put(outputDict)
    return

def task2Run():
    timoutTime = 2
    gameCount = 30
    numPairs = 20
    stepSize = 25
    startPopSize = 200
    endPopSize = 200 + numPairs*stepSize

    popSize = range(startPopSize, endPopSize, stepSize)

    #Game play details
    inputSize = 6
    smallNumbersLower = 1
    smallNumbersHigher = 11
    largeNumbers = [25, 50, 75, 100]
    numLargerNumbersToDraw = 4
    minTarget = 100
    maxTarget = 999

    #Variables to store gameplay data and results
    returnQueue = multiprocessing.SimpleQueue()
    gameDict = {}
    for i in range(gameCount):
        print(f'\nGame {i}')
        for ps in popSize:
            print(f'Executing for popsize : {ps}')

            #Empty return Queue
            returnQueue.empty()

            #How many large numbers
            largeNumbers = random.sample(largeNumbers ,random.randint(0,numLargerNumbersToDraw))
            #Generate random number for Q
            Q = random.sample(range(smallNumbersLower,smallNumbersHigher), inputSize-len(largeNumbers)) + largeNumbers
            #Generate random target
            target = random.randint(minTarget, maxTarget)

            #Set a max gens value for subsuquent runs and remove time limit for eqecution time
            maxGens = -1
            if ps in gameDict:
                maxGens = gameDict[ps]['maxGenCount']
                timoutTime = 100

            # Create a Process, which will run for timeoutTime seconds
            task2Process = multiprocessing.Process(target=task2,args=(Q, target, ps, returnQueue, maxGens,))
            task2Process.start()
            task2Process.join(timeout=timoutTime)
            task2Process.terminate()


            #Get details from fucntion and store.
            lastNode = returnQueue.get()

            #Initilise if ps not already there
            if ps not in gameDict:
                gameDict[ps] = {}
                gameDict[ps]["maxGenCount"] = -1
                gameDict[ps]["successCount"] = 0

            #For first iteraiton store teh max number of generations for subsequent runs
            if gameDict[ps]["maxGenCount"] == -1:
                gameDict[ps]["maxGenCount"] = lastNode['genCount']

            #Increment success counter if success
            if lastNode['bestTree'][0] == 0:
                gameDict[ps]["successCount"] += 1



    print("\n\n\n")
    print("Completed all games")
    print(gameDict)

    #Print success rate for each population size
    outputStr = map(lambda k: f'popSize of {k} has success rate of {gameDict[k]["successCount"]/gameCount} for {gameCount} games.', gameDict.keys())
    for l in outputStr:
        print(l)
        print("")


if __name__ == "__main__":

    task2Run()
