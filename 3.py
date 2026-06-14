import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def member_3_system_optimization(processed_matrix):
    print("\n--- Step 4: System Optimization & Risk Clustering ---")
    
    # 1. Correlation Heatmap
    cov_matrix = np.cov(processed_matrix, rowvar=False)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cov_matrix, annot=True, cmap='YlGnBu', fmt='.2f')
    plt.title("Risk Factor Correlation Heatmap")
    plt.show()

    # 2. Eigen-Analysis
    eigenvalues, eigenvectors = np.linalg.eig(cov_matrix)
    
    # 3. Diagonalization Verification
    D = np.diag(eigenvalues)
    P = eigenvectors
    is_consistent = np.allclose(cov_matrix, P @ D @ np.linalg.inv(P))

    # WOW FACTOR: Principal Component Clustering
    # Projecting 100 borrowers onto the 2 most important risk axes
    projected = processed_matrix @ eigenvectors[:, :2]
    
    plt.figure(figsize=(8, 6))
    plt.scatter(projected[:, 0], projected[:, 1], c=projected[:, 0], cmap='plasma', edgecolor='k')
    plt.title("PCA: 100 Borrowers Clustered by Risk Profile")
    plt.xlabel("Principal Risk Component 1")
    plt.ylabel("Principal Risk Component 2")
    plt.grid(True, linestyle='--', alpha=0.6)
    #plt.show()

    return {
        "eigenvalues": sorted(eigenvalues, reverse=True),
        "is_consistent": is_consistent
    }