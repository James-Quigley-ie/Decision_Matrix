# -*- coding: utf-8 -*-
"""
Created on Wed Jan 24 13:54:35 2024

@author: James
"""
import numpy as np
#The expert weightings below taken from Sataay's 9-point scale.
expert_1 = (1, 1, 1/3, 1/3, 1/5, 1/5)   
expert_2 = (1/3, 1/3, 9, 3, 7, 7)       
expert_3 = (5,3, 1/3, 1, 1/5, 1/4)      
expert_4 = (2, 4, 8, 3, 2, 4)          
expert_5 = (1/6,1/3,7,3,5,5)           
expert_6 = (3,7,3,7,1/3,1/7)            

criteria = ("Average wind speed","Large distance from residential areas","Large distance from nature areas","Proximity to 50kV/150kV transformer stations")
experts = (expert_1, expert_2, expert_3, expert_4, expert_5, expert_6)
weights = np.zeros((1, 4))
n=len(criteria)

with open('results.txt', 'w') as file:

    for expert in experts:
    
        #create the decision matrix for each expert and load their values.
        matrix = np.identity(n)
    
        comparison_value = iter(expert)
        for i in range(0,n):
            for j in range(i+1,n):
                matrix[i, j]=next(comparison_value)
                matrix[j, i]=1/matrix[i, j]
    
        # Check for consistancy
        eigenvalues, eigenvectors = np.linalg.eig(matrix)
        eigenvalues = eigenvalues.real
        eigenvectors = eigenvectors.real
        max_index = np.argmax(eigenvalues)
        CI = (eigenvalues[max_index]-n)/(n-1)     #Consistency index
        CR = CI/0.89                              #Consistency ratio when n=4
        print("\n\nConsistency Index\t", CI, file=file)
        print("Consistency Ratio\t", CR, file=file)
        
        principal_eig = eigenvectors[:,max_index]/eigenvectors[:,max_index].sum()
        print("Normalized Principal eigenvector", principal_eig, file=file)
        weights = np.vstack([weights, principal_eig])
        
        #the following section is an alternative but slightly less accurate
        #method to estimate the expert weights using the "priority vector"
        '''
        #print("_", _)        
        # Normalize the matrix
        normalized_matrix = matrix / matrix.sum(axis=0)
    
        # Calculate the priority vector
        priority_vector = normalized_matrix.mean(axis=1)
        
        # Normalize the priority vector
        normalized_priority_vector =  priority_vector/priority_vector.sum()
        #print("Expert Weights: " + str((normalized_priority_vector)), file=file)
        print("Expert Weights: " + str((normalized_priority_vector)))
        weights = np.vstack([weights, normalized_priority_vector])
        '''
     
    # Remove the first filler row.
    weights = weights[1:]
    
    # Find the average weighting
    average_weights= np.around(weights.mean(axis=0), 3)
    std_weights= np.around(weights.std(axis=0), 3)

    print("\n\nThe average weights are: ", file=file)
    
    for i, j , k in zip(average_weights, std_weights, criteria):
        print (i, "\t+/-", j, "\t", k, file=file)
