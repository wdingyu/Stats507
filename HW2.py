# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:light
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.11.5
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# ## Problem Set 2, Solution
# **Stats 507, Fall 2021**
# *Dingyu Wang*
# *September 28, 2021*

# modules: --------------------------------------------------------------------
import numpy as np
import pandas as pd
import collections 
import re
from timeit import Timer
from IPython.core.display import display, HTML
# 79: -------------------------------------------------------------------------

# ## Question 0


sample_list = [(1, 3, 5), (0, 1, 2), (1, 9, 8)]
op = []
for m in range(len(sample_list)):
    li = [sample_list[m]]
    for n in range(len(sample_list)):
        if (sample_list[m][0] == sample_list[n][0] and
                sample_list[m][2] != sample_list[n][2]):
            li.append(sample_list[n])
    op.append(sorted(li, key=lambda dd: dd[2], reverse=True)[0])
res = list(set(op))


# ### a)What task the code above accomplishes
# The code will scratch the tuple from sample_list with two types: the first
# type is the tuple whose first element is unique among the whole sample_list,
# the second type is the tuple with the first elements is equal to another
# tuple and the third element is the biggest (or share the same maximum with
# other tuple) among all the tuple with the same first element in the sample_
# list. Finally, the function will eliminate the same tuple it scratched and
# output a list with tuples (the output order is unordered, which may varies
# between different running environment).


# ### b)Code review
# * Indice out of range.
# * For loop indentation error.
# * Iterate over indices only when necessary, else iterate over values.
# * There is no need to creat a new list res.
# * Try to apply list comprehensions


# ## Question 1
# In this question, we write a function uses list comprehension to generate a
# random list of n k-tuples.


def generate_list(n, k = 10, low = 0, high = 10):
    '''
    generate a random list of n k-tuples.

    Parameters
    ----------
    n : int
        The length of the list.
    k : int, optional
        The length of the tuple.
    low : int, optional
        The lower bond of generated number.
    high : int, optional
        The upper bond of generated number.

    Returns
    -------
    A list contains n k-tuples.
    '''
    return ([tuple([np.random.randint(low, high+1)
                    for i in range(k)]) for j in range(n)])


# In this part we use `assert` to test if `generate_list()` returns a list of
# tuples.


test = generate_list(20)
assert(isinstance(test,list)),"The function doesn't return a list of tuple"
for m in test:
    assert(isinstance(m,tuple)), "The function doesn't return a list of tuple"
print("The function does return a list of tuple")

# ## Question 2
# In this question we will write several functions to accomplish the goal that
# code in Question 0 does. And a Monte carlo simulation is applied to compare
# the execution times of each functions.

# ### a) tup_for()
# This function is totally a copy of the code in Question 0, we only
# encapsulate it into a function, with input `a` represent the first indices,
# `b` represent the second indices and `sample_list` represent the list we cope
# with.


def tup_for(a,b,sample_list):
    '''
    Apply for loop to select tuples in a list.

    Parameters
    ----------
    a : int
        The first indices
    b : int
        The second indices
    sample_list : list
        The list with n k-tuples.

    Returns
    -------
    A list contains selected tuple.
    '''
    op = []
    for m in range(len(sample_list)):
        li = [sample_list[m]]
        for n in range(len(sample_list)):
            if (sample_list[m][a] == sample_list[n][a] and
                    sample_list[m][b] != sample_list[n][b]):
                li.append(sample_list[n])
        op.append(sorted(li, key=lambda dd: dd[b], reverse=True)[0])
    res = list(set(op))
    return res 


# ### b) tup_for_adv()
# In this function I apply the advice in code review to make the code more
# efficient and literate. Specifically, I use list comprehensions instead of
# simple for loop, iterate the list over values instead of indices and avoid
# redundant list generation.


def tup_for_adv(a,b,sample_list):
    '''
    An advanced function using for loop to select tuples in a list.

    Parameters
    ----------
    a : int
        The first indices
    b : int
        The second indices
    sample_list : list
        The list with n k-tuples.

    Returns
    -------
    A list contains selected tuple.
    '''
    op = []
    for m in sample_list:
        li = [m]
        li.extend([n for n in sample_list if (m[a] == n[a] and m[b] != n[b])])
        op.append(sorted(li, key=lambda dd: dd[b], reverse=True)[0])
    op = list(set(op))
    return op


# ### c) tup_dict()
# I use nested dictionaries to achieve the goal of the code in Question 0.
# Specifically, I use defaultdict to generate nested dictionary. The keys in
# outer dictionary will store the first indices of each tuple, the keys in
# inner dictionary will store the second indices of each tuple, and the values
# in inner dictionary will calculate the number of tuples with the first and
# second indices corresponding to the values of keys in the dictionary. In this
# way, we can calculate the number of tuples we want and do the following step.
#
# In this function I apply two independent for loop with sample_list. The first
# for loop is to transmit the information from sample_list to dictionary, the
# second for loop is to indices the sample_list according to our dictionary.
# List comprehension is also used to make the code more efficient.


def tup_dict(a,b,sample_list):
    '''
    Apply dictionary to select tuples in a list.

    Parameters
    ----------
    a : int
        The first indices
    b : int
        The second indices
    sample_list : list
        The list with n k-tuples.

    Returns
    -------
    A list contains selected tuple.
    '''
    d = collections.defaultdict(dict)
    for m in sample_list:                                   # First for loop
        d[m[a]] = (d.get(m[a]) if m[a] in d else collections.defaultdict(int))
        d[m[a]][m[b]] = (d.get(m[a]).get(m[b]) 
                         if m[b] in d.get(m[a]) else 0) + 1 
    li = []
    for k,v in d.items():
        if len(v) == 1:
            for e in v.keys():
                li.extend([(k, e)])
        else:
            li.extend([(k,sorted(v, reverse = True)[0])])
    op = []
    for m in sample_list:                                   # First for loop
        op.extend([m for j in li if (m[a]==j[0] and m[b]==j[1])])
    op = list(set(op))
    return op


# ### d) Comparisons
# In this part I will generate different sample_list to compare the execution
# time of each funtion. In turns of each $n=10, 100, 1000, 10000$, every
# functions will execute 10 times and calculate the mean.


# timing comparisons: ---------------------------------------------------------
res = collections.defaultdict(list)
n = [10,100,1000,10000]
res['n'] = n
for k in n:
    for f in (tup_for, tup_for_adv, tup_dict):
        t = Timer("f(a,b,n)", 
                  globals={"f": f, "a": 0, "b": 3, "n": generate_list(k)})
        m = np.mean([t.timeit(1) for i in range(10)])
        res[f.__name__].append(round(m,6))

# construct a table, include a caption: ---------------------------------------
cap = """
<b> Table 1.</b> <em> Timing comparisons for tuple selection functions.</em>
Mean computation times.
"""
res = pd.DataFrame(res)
t1 = res.to_html(index=False)
t1 = t1.rsplit('\n')
t1.insert(1, cap)
tab1 = ''
for i, line in enumerate(t1):
    tab1 += line
    if i < (len(t1) - 1):
        tab1 += '\n'
display(HTML(tab1))


# Apparently, we nested two for loops in the original code snippet, and the
# code running time is extremely long when  𝑛  is large. In the advanced
# format, even though we keep nesting two for loops, but as we apply the list
# comprehensions and reduce the redundancy, the running time is a little
# improved. As for the third function, we only apply two for loop independently
# and introduce dictionary to achieve our goal and the running time is improved
# a lot according to Table 1.


# ## Question 3
# In this question we will use Pandas to read, clean, and append several data
# files from the National Health and Nutrition Examination Survey NHANES.

# ### a) Read and append the demographic datasets
# The target of the function is to
# * Choose specific columns and rename the columns with literate variable names
# * Add an additional column identifying to which cohort each case belongs
# ("years" + "datasets name").
# * Cope with missing data and convert each column to an appropriate type.


def pd_demographic(name, year):
    '''
    Read and append the '.XPT' file of demographic datasets.

    This function will read the '.XPT' file and convert it to a DataFrame.
    Several columns are selected and renamed according to the meaning of the
    columns. Additional one column is added to the DataFrame and each column is
    convert into a appropriate type. Finally the function will return the
    processed DataFrame.

    Parameters
    ----------
    name: str
        The file's name.
    year:  str
        The conducted year of the file.

    Returns
    -------
    Processed DataFrame.
    '''
    df = pd.read_sas(name)
    columns = ['SEQN', 'RIDAGEYR', 'RIDRETH3', 'DMDEDUC2', 'DMDMARTL', 
               'RIDSTATR', 'SDMVPSU', 'SDMVSTRA', 'WTMEC2YR', 'WTINT2YR']
    columns_add = ['unique id', 'age', 'race and ethnicity', 'education',
            'marital status', 'interview/examination status', 
            'masked variance pseudo-psu', 'masked variance pseudo-stratum', 
            'sample exam weight', 'sample interview weight']
    df = df[columns]
    df = df.convert_dtypes()
    for i in range(3):
        df.iloc[:,i+3] = pd.Categorical(df.iloc[:,i+3])
    df.columns = columns_add
    cohort = 'NHANES' + ' ' + year
    df1 = pd.DataFrame({'cohort':[cohort for i in range(len(df.index))]}, 
                       index=df.index)
    df = pd.concat([df,df1], axis=1)             
    return df


# Read the files directly from the website and apply the function abrove to 
# each dataset and finally save the processed DataFrame into a pickle format.


url = 'https://wwwn.cdc.gov/Nchs/Nhanes/'
years = ['2011-2012', '2013-2014', '2015-2016', '2017-2018']
names_1 = ['DEMO_G.XPT', 'DEMO_H.XPT', 'DEMO_I.XPT', 'DEMO_J.XPT']
for i in range(4):
    name = url + years[i] +'/' + names_1[i]
    df = pd_demographic(name, years[i])
    file_name = years[i] + ' ' + 'demographic.plk'
    df.to_pickle(file_name)  


# ### b) Read and append the oral health and dentition datasets


def pd_health(name, year):
    '''
    Read and append the '.XPT' file of oral health and dentition datasets.

    This function will read the '.XPT' file and convert it to a DataFrame.
    Several columns are selected and renamed according to the meaning of the
    columns. Additional one column is added to the DataFrame and each column is
    convert into a appropriate type. Finally the function will return the
    processed DataFrame.

    Parameters
    ----------
    name: str
        The file's name.
    year:  str
        The conducted year of the file.

    Returns
    -------
    Processed DataFrame.
    '''
    df = pd.read_sas(name)
    columns_li = ['SEQN', 'OHDDESTS']
    column_1 = r'OHX\d\dCTC'
    column_2 = r'OHX\d\dTC'
    columns_li.extend(
        [m for m in df.columns if re.search(column_1, m) != None])
    columns_li.extend(
        [m for m in df.columns if re.search(column_2, m) != None])
    df = df[columns_li]
    columns_lower = [m.lower() for m in columns_li]
    columns_lower[0] = 'unique id'
    columns_lower[1] = 'dentition code'
    df.columns = columns_lower
    df1 = df.convert_dtypes()
    for i in range(61):
        df1.iloc[:, i + 1] = pd.Categorical(df1.iloc[:, i + 1])
    cohort = 'NHANES' + ' ' + year
    df2 = pd.DataFrame({'cohort': [cohort for i in range(len(df.index))]},
                       index=df.index)
    df1 = pd.concat([df1, df2], axis=1)
    return df1


# Read the files directly from the website and apply the function abrove to 
# each dataset and finally save the processed DataFrame into a pickle format.


url = 'https://wwwn.cdc.gov/Nchs/Nhanes/'
years = ['2011-2012', '2013-2014', '2015-2016', '2017-2018']
names_2 = ['OHXDEN_G.XPT', 'OHXDEN_H.XPT', 'OHXDEN_I.XPT', 'OHXDEN_J.XPT']
for i in range(4):
    name = url + years[i] +'/' + names_2[i]
    df = pd_health(name, years[i])
    file_name = years[i] + ' ' + 'oral health - dentition.plk'
    df.to_pickle(file_name)


# ### c) Number of cases there are in the two datasets
# In this step I will combine the dataset from different year together and
# reindex the DataFrame to calculate the number of cases in each datasets.

# calculate cases in each dataset: --------------------------------------------
df_demo = pd.concat([pd.read_pickle(years[i] + ' ' + 'demographic.plk')
                     for i in range(4)])
df_demo = df_demo.reset_index()
df_demo = df_demo.drop(columns = ['index'])
df_ohxden = pd.concat(
    [pd.read_pickle(years[i] + ' ' + 'oral health - dentition.plk')
     for i in range(4)
     ]
)
df_ohxden = df_ohxden.reset_index()
df_ohxden = df_ohxden.drop(columns = ['index'])
print(df_demo.shape)
print(df_ohxden.shape)


# There are 39156 cases in the demographic datasets and 35909 cases in the
# ohxden datasets.


# Calculate common cases: -----------------------------------------------------
demo_id = np.array(df_demo['unique id'], dtype='int64')
ohxden_id = np.array(df_ohxden['unique id'], dtype='int64')
count_demo = np.bincount(demo_id)
count_ohxden = np.bincount(ohxden_id)
c = count_demo + count_ohxden
count = [idx for idx, val in enumerate(c) if val == 2]
len(count)


# There are 35909 cases are both in the demographic datasets and ohxden
# datasets.
