import pandas as pd
import numpy as np

file_path = r'C:\Users\Cvinu\Downloads\Lab Session Data.xlsx'
xls = pd.ExcelFile(file_path)
purchase_data = pd.read_excel(xls, sheet_name="Purchase data")
if purchase_data.shape[1] < 5: 
    raise ValueError("The data does not contain enough columns.")

A = purchase_data.iloc[:, 1:4].apply(pd.to_numeric, errors='coerce').to_numpy()
C = purchase_data.iloc[:, 4].apply(pd.to_numeric, errors='coerce').to_numpy().reshape(-1, 1)

if np.isnan(A).any() or np.isnan(C).any():
    raise ValueError("Some values in A or C are non-numeric or missing.")
dimensionality = A.shape[1]
num_vectors = A.shape[0]
rank_A = np.linalg.matrix_rank(A)
A_pinv = np.linalg.pinv(A)
if A.shape[1] != A_pinv.shape[0]:
    raise ValueError("Matrix dimensions are not aligned for pseudo-inverse multiplication.")
X = np.dot(A_pinv, C)
print("Dimensionality of the vector space:", dimensionality)
print("Number of vectors in the space:", num_vectors)
print("Rank of matrix A:", rank_A)
print("\nCost of each product (Candies, Mangoes, Milk Packets):")
print(X)
