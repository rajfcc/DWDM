import numpy as np

def minmax_scalar(arr):
    minvalue = min(arr)
    maxvalue = max(arr)
    arr1 = [ round((((x-minvalue)/(maxvalue-minvalue))),2) for x in arr]
    return arr1

def standard_scalar(arr):
    mean = np.mean(arr)
    sample_sd = np.std(arr)
    arr1 = [round((x-mean)/sample_sd,2) for x in arr]
    # float_list = [f"{x:.2f}" for x in arr1]
    return [f"{x:.2f}" for x in arr1]
    #return arr1

arr = input("Enter the feature values sepereated by ',': \n")
try:
    split_arr = arr.split(',')
    input_array = list(map(float,split_arr))
except ValueError as e:
     print("Error: Invalid input. Please enter valid integers.")
else:
    print(f"The minmax scalar for the provided data is : {minmax_scalar(input_array)}")
    print(f"The standard scalar for the provided data is : {standard_scalar(input_array)}")


