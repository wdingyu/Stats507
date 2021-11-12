#!/usr/bin/env python
# coding: utf-8
# %%
"""
@author: Dingyu Wang
@date: September 12, 2021
"""

# modules: -------------------------------------------------------------------
import math
import time
import numpy as np
import pandas as pd
import warnings
from scipy import stats


# %%
# # Question 1

# for loop and test function: ------------------------------------------------
def fib_for(n):
    '''

    Computes F_n by summation using a for loop.

    Parameters
    ----------
    n : Integer
        The nth Fibonacci Sequence we want

    Returns
    -------
    Integer
        The nth Fibonacci Sequence(F_n)

    '''

    a = 0; b = 1;
    for i in range(n):
        a, b = b, a + b
    return a

def test_for():
    '''

    Test function(n = 7, 11, 13)

    Parameters
    ----------

    Returns
    -------

    '''
    n = [7, 11, 13]
    for i in range(len(n)):
        print("F_{0:.0f} = {1:.0f}".format(n[i], fib_for(n[i])))
    return

# while loop and test function: ----------------------------------------------
def fib_whl(n):
    '''

    Computes F_n by summation using a while loop.

    Parameters
    ----------
    n : Integer
        The nth Fibonacci Sequence we want

    Returns
    -------
    Integer
        The nth Fibonacci Sequence(F_n)

    '''

    a = 0; b = 1; i = 0;
    while(i < n):
        a, b = b, a + b;
        i = i + 1;
    return a

def test_whl():
    '''

    Test function(n = 7, 11, 13)

    Parameters
    ----------

    Returns
    -------

    '''
    n = [7, 11, 13]
    for i in range(len(n)):
        print("F_{0:.0f} = {1:.0f}".format(n[i], fib_whl(n[i])))
    return

# recursive and test function: -----------------------------------------------
def fib_rec(n, f0, f1):
    '''

    Computes F_n by using a recursive function.

    Parameters
    ----------
    n : Integer
        The nth Fibonacci Sequence we want
        
    f0 : Integer
        Define the first Fibonacci Sequence 
        
    f1 : Integer
        Define the second Fibonacci Sequence 

    Returns
    -------
    Integer
            The nth Fibonacci Sequence(F_n)

    '''

    if n == 0:
        return f0
    if n == 1:
        return f1
    return (fib_rec(n-2, f0, f1) + fib_rec(n-1, f0, f1))

def test_rec(f0, f1):
    '''

    Test function(n = 7, 11, 13)

    Parameters
    ----------
    f0 : Integer
        Define the first Fibonacci Sequence 
        
    f1 : Integer
        Define the second Fibonacci Sequence 

    Returns
    -------

    '''
    n = [7, 11, 13]
    for i in range(len(n)):
        print("F_{0:.0f} = {1:.0f}".format(n[i], fib_rec(n[i], f0, f1)))
    return

# rounding function and test function: ---------------------------------------
def fib_rnd(n):
    '''
    Computes F_n by using the rounding method.

    Parameters
    ----------
    n : Integer
        The nth Fibonacci Sequence we want

    Returns
    -------
    Integer
        The nth Fibonacci Sequence(F_n)

    '''
    phi = 1 / 2 + math.sqrt(5) / 2;
    a = round(math.pow(phi, n)) / math.sqrt(5);
    return a

def test_rnd():
    '''

    Test function(n = 7, 11, 13)

    Parameters
    ----------

    Returns
    -------

    '''
    n = [7, 11, 13]
    for i in range(len(n)):
        print("F_{0:.0f} = {1:.0f}".format(n[i], fib_rnd(n[i])))
    return 
# truncation function and test function: -------------------------------------
def fib_flr(n):
    '''
    Computes F_n by using the truncation method.

    Parameters
    ----------
    n : Integer
        The nth Fibonacci Sequence we want

    Returns
    -------
    Integer
        The nth Fibonacci Sequence(F_n)

    '''

    phi = 1 / 2 + math.sqrt(5) / 2;
    a = int(math.pow(phi, n) / math.sqrt(5) + 1 / 2);
    return a

def test_flr():
    '''

    Test function(n = 7, 11, 13)

    Parameters
    ----------

    Returns
    -------

    '''
    n = [7, 11, 13]
    for i in range(len(n)):
        print("F_{0:.0f} = {1:.0f}".format(n[i], fib_flr(n[i])))
    return

# test running time: ---------------------------------------------------------
def show_time():
    '''
    Computes the operating time of each method.

    Parameters
    ----------
    
    Returns
    -------
    Data with Dataframe format

    '''
    f0 = 0
    f1 = 1
    k = 0
    result = np.zeros(shape=(7,5))
    for n in range(5, 40, 5):
        times = np.zeros(shape=(3,5))
        for i in range(3):
            start_1 = time.perf_counter()
            fib_rec(n, f0, f1)
            end_1 = time.perf_counter()
            times[i][0] = (end_1 - start_1)

            start_2 = time.perf_counter()
            fib_for(n)
            end_2 = time.perf_counter()
            times[i][1] = (end_2 - start_2)

            start_3 = time.perf_counter()
            fib_whl(n)
            end_3 = time.perf_counter()
            times[i][2] = (end_3 - start_3)

            start_4 = time.perf_counter()
            fib_rnd(n)
            end_4 = time.perf_counter()
            times[i][3] = (end_4 - start_4)

            start_5 = time.perf_counter()
            fib_flr(n)
            end_5 = time.perf_counter()
            times[i][4] = (end_5 - start_5)

        for i in range(5):
            result[k][i] = round(np.median(times[:,i]), 8)
        k = k + 1
    

    result_pd = pd.DataFrame(result)
    result_pd.columns = ['recursive function', 'for loop',
                         'while loop', 'rounding method', 'truncation method']
    result_pd.index = ['n = 05', 'n = 10', 'n = 15', 
                       'n = 20', 'n = 25', 'n = 30', 'n = 35']
    return result_pd


# %%
# # Question 2(a)

# Calculate Pascal's triangle: -----------------------------------------------
def compute_row(n,k):
    '''
    Calculate the value of combinations.

    Parameters
    ----------
    n : Integer
        
    k : Integer

    Returns
    -------
    Integer
        C_{n}^{k}

    '''
    if k == 0:
        return 1
    return int(compute_row(n,k-1) * (n + 1 - k) / k)
    
def Pascal(n):
    '''
    Calculate the nth row of Pascal's triangle.

    Parameters
    ----------
    n : Integer
        nth row 

    Returns
    -------
    Str 
        The nth row of Pascal's triangle.
        
    '''
    a = []
    for i in range(n+1):
        a.append(str(compute_row(n,i)))
    a = ' '.join(a)
    return a


# %%
# # Question 2(b)

# print Pascal's triangle: ---------------------------------------------------
def Pascal_centered(n):
    '''
    Output the nth row of Pascal's triangle in center format.

    Parameters
    ----------
    n : Integer
        nth row 

    Returns
    -------
    Centered printing the nth row Pascal's triangle
        
    '''
    a = []
    for i in range(n+1):
        a.append(str(compute_row(n,i)))
    a = '         '.join(a)   
    a = a.center(130)
    return a

def print_Pascal(n):
    '''
    Print Pascal's triangle.

    Parameters
    ----------
    n : Integer
        nth row 

    Returns
    -------
    print n rows of Pascal's triangle
        
    '''
    for i in range(0,n+1):
        print(Pascal_centered(i))
    return



# %%
# # Question 3(a)

# Compute point and interval estimates: --------------------------------------
def estiamte(data, conf_level, string_set = "str"):
    '''
    Compute point and interval estimates of a data set.

    Parameters
    ----------
    data : 1d Numpy array
           The data need to be estimated 
           
    conf_level : Integer
                 Confidence level
                 
    string_set : Default Parameters
                 Control the type of return value 
    
    Returns
    -------
    Str
        Point and interval estiamation with type of str
        
    Dictoniary 
        Dictionary with keys est, lwr, upr, and level

    '''

    if type(data) != np.ndarray:
        raise Exception("Invalid data format")

    mean_hat = np.mean(data)
    std_hat = np.std(data)
    alpha = 1 - conf_level/100
    z = stats.norm(0, 1)
    n = len(data)
    est = std_hat / np.sqrt(n) * z.ppf(1 - alpha / 2)
    upr = mean_hat + est
    lwr = mean_hat - est
    
    if string_set == "str":
        return (
            "{0:.4f}[{1:.0f}%CI:({2:.4f}, {3:.4f})]"
                .format(mean_hat,conf_level, lwr, upr)
        )
    if string_set == None:
        dic = {"est": mean_hat, "lwr": lwr, "upr": upr, "level": conf_level}
        return dic



# %%
# # Question 3(b)

# Compute point and interval estimates: --------------------------------------
def Binomial_est(data, conf_level, method, string_set = "str"):
    '''
    Compute point and interval estimates of a data set with different methods.

    Parameters
    ----------
    data : 1d Numpy array
           The data need to be estimated 
           
    conf_level : Integer
                 Confidence level
                 
    method : Str
             The method used to compute point and interval estiamtes
                 
    string_set : Default Parameters
                 Control the type of return value 
    
    Returns
    -------
    Str
        Point and interval estiamation with type of str
        
    Dictoniary 
        Dictionary with keys est, lwr, upr, and level

    '''
    if type(data) != np.ndarray:
        raise Exception("Invalid data format")
        
    n = len(data)
    x = np.sum(data)
    p_hat = x / n
    alpha = 1 - conf_level/100
    
    if method == 'Normal':
        mean_hat = np.mean(data)
        std_hat = np.std(data)
        z = stats.norm(0, 1)
        est = std_hat / np.sqrt(n) * z.ppf(1 - alpha / 2)
        upr = mean_hat + est
        lwr = mean_hat - est
        
               
    if method == "Binomial":
        if max(n * p_hat, n * (1 - p_hat))<12:
            warnings.warn("Condition is not satisfied", UserWarning)
        z = stats.norm(0, 1)
        num = np.sqrt(p_hat * (1 - p_hat) / n) * z.ppf(1 - alpha / 2)         
        upr = p_hat + num
        lwr = p_hat - num

            
            
    if method == "Clopper-Pearson":       
        z_1 = stats.beta(x, n - x + 1)
        lwr = z_1.ppf(alpha / 2)
        z_2 = stats.beta(x + 1, n - x)
        upr = z_2.ppf(1 - alpha / 2)
        
        
    if method == "Jeffrey":       
        z = stats.beta(x + 0.5, n - x + 0.5)
        lwr = max(z.ppf(alpha / 2), 0)
        upr = min(z.ppf(1 - alpha / 2), 1)    
        
        
    
    if method == "Agresti-Coull":       
        z = stats.norm(0, 1).ppf(1 - alpha / 2)
        n_bar = n + z * z
        p_bar = (x + z * z / 2) / n_bar
        num = np.sqrt(p_bar * (1 - p_bar) / n_bar) * z
        upr = p_hat + num
        lwr = p_hat - num
    
    
    
    
    if string_set == "str":
        return (
            "{0:.4f}[{1:.0f}%CI:({2:.4f},{3:.4f})]"
                .format(p_hat, conf_level, lwr, upr)
        )
    if string_set == None:
        dic = {"est": p_hat, "lwr": lwr, "upr": upr, "level": conf_level}
        return dic

# %%
# # Question 3(c)

# Compute confidence interval: -----------------------------------------------
def show_interval(data, confidence_interval):
    '''
    Output the confidence interval of different method with different
    confidence level.

    Parameters
    ----------
    data : 1d Numpy array
           The data need to be estimated 
           
    confidence_interval : 1d Numpy array
                          Confidence level
                 
    Returns
    -------
    Data with table format 

    '''

    method = ["Binomial", "Clopper-Pearson", "Jeffrey", "Agresti-Coull"]
    result = [[], [], [], [], []]
    for i in range(len(confidence_interval)):
        for j in range(len(method)):
            dic = Binomial_est(data, confidence_interval[i], method[j], None)
            result[j].append(('%.4f'%dic["lwr"], '%.4f'%dic["upr"]))
        dic = estiamte(data, confidence_interval[i], None)
        result[j + 1].append(('%.4f'%dic["lwr"], '%.4f'%dic["upr"]))

    result_pd = pd.DataFrame(result)
    result_pd.columns = [
        "{0:.0f}% confidence intervals".format(confidence_interval[0]), 
        "{0:.0f}% confidence intervals".format(confidence_interval[1]), 
        "{0:.0f}% confidence intervals".format(confidence_interval[2])
    ]
    result_pd.index = [
        'Tradition Binomial Method', 'Clopper-Pearson Method', 
        'Jeffrey Method', 'Agresti-Coull Method', 'Normal theory'
    ]
    return result_pd


def show_width(data, confidence_interval):
    '''
    Output the width of confidence interval of different method with different
    confidence level.

    Parameters
    ----------
    data : 1d Numpy array
           The data need to be estimated 
           
    confidence_interval : 1d Numpy array
                          Confidence level
                 
    Returns
    -------
    Data with table format 

    '''
    method = ["Binomial", "Clopper-Pearson", "Jeffrey", "Agresti-Coull"]
    result_interval = [[], [], [], [], []]
    for i in range(len(confidence_interval)):
        for j in range(len(method)):
            dic = Binomial_est(data, confidence_interval[i], method[j], None)
            result_interval[j].append(dic["upr"] - dic["lwr"])
        dic = estiamte(data, confidence_interval[i], None)
        result_interval[j + 1].append(dic["upr"] - dic["lwr"])

    result_interval_pd = pd.DataFrame(result_interval)
    result_interval_pd.columns = [
        "{0:.0f}% confidence intervals".format(confidence_interval[0]), 
        "{0:.0f}% confidence intervals".format(confidence_interval[1]), 
        "{0:.0f}% confidence intervals".format(confidence_interval[2])
    ]
    result_interval_pd.index = [
        'Tradition Binomial Method', 'Clopper-Pearson Method', 
        'Jeffrey Method', 'Agresti-Coull Method', 'Normal theory'
    ]
    return result_interval_pd


