import pandas as pd
import numpy as np
import os

file_path = os.getenv('EXCEL_FILE_PATH', r"C:\Users\Cvinu\OneDrive - Amrita vishwa vidyapeetham (1)\SEM-4\ML\lab-2\Lab Session Data.xlsx")

try:
    with pd.ExcelFile(file_path) as xls:
  
        purchase_data = pd.read_excel(xls, sheet_name="Purchase data", usecols=[1, 2, 3, 4])
        
    A = purchase_data.iloc[:, :3].apply(pd.to_numeric, errors='coerce').to_numpy()
    C = purchase_data.iloc[:, 3].apply(pd.to_numeric, errors='coerce').to_numpy()

    if np.isnan(A).any() or np.isnan(C).any():
        print("Warning: There are NaN values in the data after conversion. Please check the input data.")

    A_pinv = np.linalg.pinv(A)

    X = np.dot(A_pinv, C)

    print("\nPredicted Cost per Unit for each Product:")
    print(f"Candies: Rs. {X[0]:.2f} per unit")
    print(f"Mangoes: Rs. {X[1]:.2f} per Kg")
    print(f"Milk Packets: Rs. {X[2]:.2f} per unit")

except FileNotFoundError:
    print(f"Error: The file was not found at {file_path}. Please check the path.")
except Exception as e:
    print(f"An error occurred: {e}")
