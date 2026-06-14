import numpy as np
import matplotlib.pyplot as plt

def run_part2(A, b, pivots):
    #print("\n--- Step 3: Orthogonalization & Prediction ---")
    
    # 1. Select Basis (Removing the redundant column identified by Member 1)
    # We take columns 0, 1, 2 (the pivots)
    basis = A[:, pivots]        #all rows and pivot no of cols
    
    # 2. Gram-Schmidt Logic
    n, m = basis.shape          
    Q = np.zeros((n, m))     #a matrix Q which will hold the orthogonal vectors
    for j in range(m):       #loop through all col of basis
        v = basis[:, j]      #vector v with all rows and jth col
        for i in range(j):   #another loop to compare current vector with previously computed orthogonal vectors
            proj = np.dot(Q[:, i], basis[:, j]) / np.dot(Q[:, i], Q[:, i])   #calc proj coeff of current vector on Q[:,i] 
            #calculating how much the current vector points in the direction of prev
            v = v - proj * Q[:, i]  #subtract proj from v to make it orthogonal
        Q[:, j] = v          
    
    # 3. Normalize to Orthonormal
    for i in range(Q.shape[1]):           #loop through each col in Q
        norm = np.linalg.norm(Q[:, i])    #find the lenght of each col
        if norm != 0:
            Q[:, i] = Q[:, i] / norm      #normalise the vector to make it unit lenght

    # 4. Least Squares Prediction
    # Solving Ax = b to predict risk scores
    x_hat = np.linalg.inv(basis.T @ basis) @ basis.T @ b   #least square formula to get the best weights for prediction of scores
    prediction = basis @ x_hat     #multiply borrower data with the weights to get pred risk scores

    # WOW FACTOR: Visualization of Prediction
    plt.figure(figsize=(10, 4))
    plt.scatter(range(len(b)), b, color='gray', alpha=0.5, label='Actual Historical Risk')  #plots actual risk scores
    plt.plot(range(len(b)), prediction, color='red', linewidth=2, label='Model Prediction')  #draws prediction curve
    plt.title("Least Squares: Predicting Borrower Risk Scores")
    plt.xlabel("Borrower ID")
    plt.ylabel("Risk Score")
    plt.legend()
    #plt.show()

    return {"Q": Q, "least_squares_solution": x_hat}