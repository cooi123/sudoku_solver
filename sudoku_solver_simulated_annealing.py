import numpy as np

def metropolis(f_x, f_x_star, t, minimisation=False):
    delta = (-1 if minimisation else 1)*(f_x_star - f_x)
    if delta > 0:
        return True
    return np.random.random() < np.exp(delta/t)

def SA(f, peturb, sol, per_temp, t0, alpha, minimisation=False, tol=1e-3, verbose=False):
    T = t0
    if verbose: print("initial guess: {0}, quality: {1}".format(sol, f(sol)))
    while T > tol:
        for _ in range(per_temp):
            sol_new = peturb(sol)
            if metropolis(f(sol), f(sol_new), T, minimisation=minimisation):
                sol = sol_new.copy()
        if verbose: print("Best Guess at T={0}: {1}, quality: {2}".format(T, sol, f(sol)))
        T = alpha*T
    if verbose: print("Best Guess: {0}, quality: {1}".format(sol, f(sol)))
    return sol, f(sol)


def sudoku_cost_gen(n):
    def sudoku_evaluation(sol):
        #assume that solution is filled with numbers from 1 to n**2
        #check the number of duplicate for row
        row_duplicate_sum = 0
        for i in range(n**2):
            count = [0] * n**2
            for j in range(n**2):
                count[sol[i,j]-1] += 1
            row_duplicate_sum += sum([x-1 for x in count if x > 1])
        #check the number of duplicate for column
        col_duplicate_sum = 0
        for j in range(n**2):
            count = [0 ]* n**2
            for i in range(n**2):
                count[sol[i,j]-1] += 1
            col_duplicate_sum += sum([x-1 for x in count if x > 1])
        return row_duplicate_sum + col_duplicate_sum
    return sudoku_evaluation

def sudoku_perb_gen(n, fix):
    def sudoku_perb( sol):
        #cant move fix number
        #make sure the block doesnt have duplicate
        #only modify block 
        sol_new = sol.copy()
        
        i, j = np.random.randint(0,n,2)
        r1, r2 = np.random.randint(i*n,(i+1)*n,2)
        c1,c2 = np.random.randint(j*n,(j+1)*n,2)
        while (fix[r1,c1] == 1) or (fix[r2,c2] == 1):
            i, j = np.random.randint(0,n,2)
            r1, r2 = np.random.randint(i*n,(i+1)*n,2)
            c1,c2 = np.random.randint(j*n,(j+1)*n,2)
        sol_new[r1,c1], sol_new[r2,c2] = sol_new[r2,c2], sol_new[r1,c1]

        return sol_new
    return sudoku_perb


def fillInitial(t0, n):
    #fill the block first
    t0_new = t0.copy()
    for i in range(n):
        for j in range(n):
            block = t0_new[i*n:(i+1)*n, j*n:(j+1)*n]
            missing = [x for x in range(1,n**2+1) if x not in block]
            for row in range(i*n,(i+1)*n):
                for col in range(j*n,(j+1)*n):
                    if t0_new[row,col] == None:
                        t0_new[row,col] = missing.pop()
    return t0_new
def create_fix(puzzle):
    fix = puzzle.copy()

    return fix!=None

def sudoku_solver(puzzle):
    #assume puzzle is given in a 2d array
    #assume puzzle is square
    n = int(np.sqrt(len(puzzle)))
    fix = create_fix(puzzle)
    t0 = fillInitial(puzzle, n)
    sudoku_cost = sudoku_cost_gen(n)
    sudoku_perb = sudoku_perb_gen(n, fix)
    sol, cost = SA(sudoku_cost, sudoku_perb, t0, 100, 100, 0.9, minimisation=True, tol=1e-3, verbose=False)
    print(sol, cost)
    return sol


if __name__ == "__main__":
    t_p = np.array([[None,2,4,None,None,7,None,None,None],
               [6, None, None, None, None, None, None, None, None],
               [None,None, 3, 6,8,None, 4,1,5],
               [4,3,1, None, None,5, None, None, None],
               [5, None, None, None, None, None, None, 3,2],
               [7,9, None, None, None, None, None, 6, None],
               [2,None, 9 ,7,1, None,8, None, None],
               [None, 4, None,None,9,3, None, None, None],
               [3,1,None,None,None, 4,7,5, None]
               ])
    sudoku_solver(t_p)


