import numpy as np
import sympy as sp
def data_architect_analysis(matrix_input=None):
    if matrix_input is not None:
        A = matrix_input
        print("--- Step 1: Analyzing Provided CSV Dataset ---")
    else:
        print("--- Step 1: Generating Synthetic FinTech Dataset ---")
        np.random.seed(42)
        n_borrowers = 100
        annual_income = np.random.normal(70000, 15000, n_borrowers)
        credit_score = np.random.randint(300, 850, n_borrowers)
        debt_ratio = np.random.uniform(0.1, 0.6, n_borrowers)
        monthly_income = annual_income / 12 
        A = np.column_stack((annual_income, credit_score, debt_ratio, monthly_income))
        A = np.round(A, 2)
    
    # RREF Analysis
    matrix_sp = sp.Matrix(A[:10, :]) 
    rref_matrix, pivots = matrix_sp.rref(iszerofunc=lambda x: abs(x) < 1e-2)
   
    rank = len(pivots)
    nullity = A.shape[1] - rank


    print(f"\n" + "="*40)
    print(f"RANK-NULLITY THEOREM RESULTS")
    print(f"="*40)
    print(f"Rank (Unique Features): {rank}")
    print(f"Nullity (Redundancy):    {nullity}")
    print(f"Pivot Columns:          {pivots}")
    print(f"="*40 + "\n")

    return A, pivots