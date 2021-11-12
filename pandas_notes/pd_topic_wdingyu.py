# -*- coding: utf-8 -*-
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

# *Dingyu Wang*
#
# <wdingyu@umich.edu>

# ## Concatenate 
# * Concatenate pandas objects along a particular axis with optional set logic
# along the other axes.
# * Combine two Series.

import pandas as pd

s1 = pd.Series(['a', 'b', 'c'])
s2 = pd.Series(['d', 'e', 'f'])
pd.concat([s1, s2])

# ## Concatenate 
# * Concatenate pandas objects along a particular axis with optional set logic
# along the other axes.
# * Combine two Series.
# * Combine two DataFrame objects with identical columns.

df1 = pd.DataFrame([['a', 1], ['b', 2], ['c', 3]],
                   columns=['letter', 'number'])
df2 = pd.DataFrame([['d', 4], ['e', 5], ['f', 6]],
                   columns=['letter', 'number'])
pd.concat([df1, df2])

# ## Concatenate 
# * Concatenate pandas objects along a particular axis with optional set logic
# along the other axes.
# * Combine two Series.
# * Combine two DataFrame objects with identical columns.
# * Combine DataFrame objects with overlapping columns and return only those
# that are shared by passing inner to the join keyword argument
# (default outer).

df3 = pd.DataFrame([['a', 1, 'Mary'], ['b', 2, 'John'], ['c', 3, 'James']],
                   columns=['letter', 'number', 'name'])
pd.concat([df1, df3])
pd.concat([df1, df3], join='inner')

# ## Concatenate 
# * Concatenate pandas objects along a particular axis with optional set logic
# along the other axes.
# * Combine two Series.
# * Combine two DataFrame objects with identical columns.
# * Combine DataFrame objects with overlapping columns and return only those
# that are shared by passing in `join=inner`(default outer).
# * Combine DataFrame objects horizontally along the x axis by passing in
# `axis=1`(default 0).

df4 = pd.DataFrame([['Tom', 24], ['Jerry', 18], ['James', 22]],
                   columns=['name', 'age'])
pd.concat([df1, df4])
pd.concat([df1, df4], axis=1)
