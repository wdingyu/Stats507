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

import pandas as pd
import numpy as np
import statsmodels.api as sm
import statsmodels.formula.api as smf
from os.path import exists
import matplotlib.cm as cm
import matplotlib.pyplot as plt
import patsy
import re
from collections import defaultdict
from base64 import b64encode
from IPython.core.pylabtools import print_figure 
from matplotlib._pylab_helpers import Gcf
from IPython.core.display import display, HTML
import warnings

# ## Question 0 - R-Squared Warmup
# In this question you will fit a model to the ToothGrowth data used in the
# notes on Resampling and Statsmodels-OLS. Read the data, log transform tooth
# length, and then fit a model with indpendent variables for supplement type,
# dose (as categorical), and their interaction. Demonstrate how to compute the
# R-Squared and Adjusted R-Squared values and compare your compuations to the
# attributes (or properties) already present in the result object.

file = 'tooth_growth.feather'
if exists(file):
    tg_data = pd.read_feather(file)
else: 
    tooth_growth = sm.datasets.get_rdataset('ToothGrowth')
    tg_data = tooth_growth.data
    tg_data.to_feather(file)

# Log transform the len and transform supplement to one-hot variable-----------
tg_data['log_len']= tg_data[['len']].transform(np.log)
tg_data['OJ'] = pd.get_dummies(tg_data['supp'])['OJ']

# Transform the type of dose to categorical------------------------------------
tg_data['dose_cat'] = pd.Categorical(tg_data['dose'])

# Linear regression with supp, dose and their interaction----------------------
mod = smf.ols('log_len ~ OJ*dose_cat', data=tg_data)
res = mod.fit()
res.summary2()

# $$R^2 = 1 - \frac{(\hat{y}_i-y_i)^2}{(y_i-\bar{y})^2}$$
# where $\bar{y}=\frac{1}{n}\sum_{i=1}^{n}y_i$, $\hat{y}_i=\sum_{j=1}^{n}x_{ij}
# \beta_{j}+\epsilon_i$

# Calculate R-square and compare R-square in the model-------------------------
rss = ((np.dot(mod.exog, res.params)-mod.endog)**2).sum()
tss = ((mod.endog.mean() - mod.endog)**2).sum()
R_square = 1 - rss/tss
assert(np.round(R_square, 10)== np.round(res.rsquared, 10))

# $$\mbox{Adj} \ R^2= 1 - \frac{(1 - R^2)(n-1)}{n-(p+1)}$$

# Where $n$ is the number of observations and $p$ is the number of predictors.

# Calculate  Adj R-square and compare with Adj R-square in the model-----------
Adj_r = 1 - np.divide(mod.nobs-1, mod.df_resid)*(1-res.rsquared)
assert(Adj_r == res.rsquared_adj)

# ## Question 1 - NHANES Dentition
# ### Part a)
# Pick a single tooth (OHXxxTC) and model the probability that a permanent
# tooth is present (look up the corresponding statuses) as a function of age
# using logistic regression. For simplicity, assume the data are iid and ignore
# the survey weights and design. Use a B-Spline basis to allow the probability
# to vary smoothly with age. Perform model selection using AIC or another
# method to choose the location of knots and the order of the basis (or just
# use order=3 and focus on knots).

demo = pd.read_pickle('demographic.pickle') 
ohx = pd.read_pickle('oral health - dentition.pickle')

ohx0 = pd.merge(ohx, demo.loc[:,['id', 'gender', 'age', 'race']],
                how='left', on='id')
ohx0['age'] = ohx0['age'].astype(int)
dep_vars =  {'ohx01tc': {'Permanent tooth present': 1, 
                         'Tooth not present': 0, 
                         'Permanent dental root fragment present': 0,
                         'Primary tooth present': 0,
                         'Dental Implant': 0,
                         'Could not assess': np.nan}}
gender_col = {'Female':0, 'Male':1}
race_col = {'Mexican American':1,
            'Other Hispanic':2,
            'Non-Hispanic White':3,
            'Non-Hispanic Black':4,
            'Non-Hispanic Asian':5,
            'Other/Multiracial':7
            }
ohx0['race'] = pd.Categorical(ohx0['race'].replace(race_col))
ohx0['gender'] = pd.Categorical(ohx0['gender'].replace(gender_col))
for c in dep_vars.keys():
    ohx0[c] = ohx0[c].replace(dep_vars[c])

# Limit the analyses to those age 12 and older so that singular matrix will not
# happen during the logistic regression.

ohx0 = ohx0.query('age>=12')

ohx0 = ohx0.dropna()

# Apply logistic regression to independent variable age and dependent variable
# "ohx01tc".

mod0 = smf.logit('ohx01tc ~ age', data=ohx0)
res0 = mod0.fit(disp=True)
res0.summary()

# This plot is mainly to display the correlation between age and the
# probability the specific tooth is permanent tooth of a person. It will give
# us a straightword insight on how the probability varies with age, which will
# help us to select knots in the following step.

plt.plot(ohx0.groupby(['age'])['ohx01tc'].mean())

# In this step, I am going to use B-spline basis to allow the probability vary
# smoothly with age. My strategy fot knots selection is as following:
# * Assume there is only one knot in the model and traverse all the points from
# 12-79 to select the point with the minimal AIC, denote as $P_1$.
# * Assume there are two knots in the model and the first knot is $P_1$, select
# another knot from the rest of points $\in [12,P_1)\cup(P_1,79]$ with the
# minal AIC and denote as $P_2$. The knots of this step is $[P_1, P_2]$.
# * Approximately repeat the above step with $k = 3,4,5,...,n$ knots and find
# the best point $P_n$ adding to the former model.
# * Compare the AIC of the best model we select from  knots with
# $k = 1,2,...,n$ and finalize the model with the best AIC
#
# I set $n=10$ in my model selection code, which I believe is suifficient to
# find the best model with knots. As for the order of the B-spline basis, I set
# the order to be 3 because it is the minimal order that can allow the first
# and second derivate continues. In the following code, I calculate the minimal
# AIC and its corresponding knots.

age = [i for i in range(13,80)]
val = []
for i in range(10):
    aic = 30000
    if i != 0:
        age.remove(val[-1])
    for m in age:
        try:     
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore")
                knot = []
                x = ohx0['age']
                vall = val.copy()
                vall.append(m)
                knot.extend(vall)
                knot = [str(x) for x in knot]
                knots = ','.join(knot)
                y = patsy.dmatrix("bs(x, knots = ["
                                  + knots +
                                  "], degree=3, include_intercept=False)-1",
                                  {"x": x}, return_type='dataframe')
                mod0 = smf.logit('ohx01tc ~ y', data=ohx0)
                res0 = mod0.fit(disp=False)
                if res0.aic<aic:
                    aic = res0.aic
                    num = m
        except np.linalg.LinAlgError as err:
            if 'Singular matrix' in str(err):
                print(
                    'knots = [{0:d}], Singular matrix in regression'.format(m)
                )
            else:
                raise
    val.append(num)
    print(val)
    print(aic)

# According to the result above, the best model with knots is
# $[25, 79, 78, 77]$. But as for this kind of knots, I find most of the knots
# located at the tail of the data, and I still want to add some knot at the
# middle range of the data. Therefore I find when knots =$[25, 79, 78, 77, 66]$
# , the AIC is almost the same as the best model. But with knot=$66$ disperse
# the distribution of knots. As a result, I do believe
# knots =$[25, 79, 78, 77, 66]$ is my best result of knots selection. Here is
# the final model and the summary of the regression.

x = ohx0['age']
y = patsy.dmatrix(
    "bs(x, knots = [25,79,78,77,66], degree=3, include_intercept=False)-1",
    {"x": x}, return_type='dataframe'
)
mod_best = smf.logit('ohx01tc ~ y', data=ohx0)
res_best = mod_best.fit(disp=True)
res_best.summary()

# According to the result, we can find that the p-value of all the predictors
# is signifiacnt under the 95% confidence level, which indicate that all the
# variables we include in the regression do have a significant impact on the
# reponse.

res_best.aic

# In the next step, we try to include the information in the demographics table
# to make the prediction more reliable. In order to select the variables from
# demographics data, let us first consider the realistic meaning of the
# logistic regression we do. Generally speaking, we predict the probability of
# a tooth is a permanent tooth. This probability is apperantly influenced by
# age, but it is also resonable that the probability is effected by gender and
# race because such probability is a kind of physiological phenomenon and will
# relate with the gender and race. Therefore, I include gender and race
# independently in the model and also include both of them in the model. Here
# is the results in terms of these three models.

mod_gender = smf.logit('ohx01tc ~ y+gender', data=ohx0)
res_gender = mod_gender.fit(disp=False)
mod_race = smf.logit('ohx01tc ~ y+race', data=ohx0)
res_race = mod_race.fit(disp=False)
mod_comb = smf.logit('ohx01tc ~ y+gender+race', data=ohx0)
res_comb = mod_comb.fit(disp=False)
pd.DataFrame([['age', res_best.aic], ['age, gender', res_gender.aic],
              ['age, race', res_race.aic],
              ['age, gender, race', res_comb.aic]],
             columns=['Variables', 'AIC'])

# According to the result, we find that the AIC of the model is the smallest
# when include variables gender and race. Therefore, if I take AIC as the
# evaluation standard, my final model include three variables which are
# **age(smooth), gender and race**.

# ### part b)

# According to the discussion above, I pick the knots = $[25,79,78,77,66]$ and
# include variables "gender" and "race" in demographic data.

ohx_new = pd.merge(ohx, demo.loc[:,['id', 'age', 'race', 'gender']],
                   how='left', on='id')
ohx_new['age'] = ohx_new['age'].astype(int)
tc = {'Permanent tooth present': 1, 
      'Tooth not present': 0, 
      'Permanent dental root fragment present': 0,
      'Primary tooth present': 0,
      'Dental Implant': 0,
      'Could not assess': np.nan}
ohx_new['race'] = pd.Categorical(ohx_new['race'].replace(race_col))
ohx_new['gender'] = pd.Categorical(ohx_new['gender'].replace(gender_col))

column = r'ohx\d\dtc'
columns_tc = [m for m in ohx0.columns if re.search(column,m) != None]

for col in columns_tc:
    ohx_new[col] = ohx_new[col].replace(tc)

ohx_new = ohx_new.query('age>=12')

result = ohx_new.copy()
for col in columns_tc:
    ohx_test = result.loc[:,['id', 'age','gender', 'race', col]]
    ohx_test = ohx_test.dropna()
    x = ohx_test['age']
    y = patsy.dmatrix(
      "bs(x, knots=[25, 79, 78, 77, 66], degree=3, include_intercept=False)-1",
      {"x": x}, return_type='dataframe')
    mod = smf.logit('{0:s}~y + gender + race'.format(col), data=ohx_test)
    res = mod.fit(disp=False)
    ohx_test[col] = mod.predict(params=res.params, exog=mod.exog)
    ohx_test = ohx_test.loc[:,['id', col]]
    result = result.drop(columns=[col])
    result = pd.merge(result, ohx_test, how='left', on='id')

columns_new = ['id', 'age', 'race'] + columns_tc
pred = result.loc[:,columns_new]
pred = pred.dropna()

table = pred.head(100)

cap = """
<b> Table 1. </b> <em>Predicted probability of a tooth is a permanent tooth in 
terms of age.</em> The predicted values are designed to retain 6 decimal 
places. This table only shows the first 100 predicted data because printing all
 the datas in the HTML form is too time-consuming.
"""
t1 = table.to_html(index=True)
t1 = t1.rsplit('\n')
t1.insert(1, cap)
tab1 = ''
for i, line in enumerate(t1):
    tab1 += line
    if i < (len(t1) - 1):
        tab1 += '\n'

display(HTML(tab1))

# ### part c)

# Use the result above, I can calculate the mean probability that a specific
# tooth is permanent of the perople with the same age.

result = pred.groupby(['age']).mean().drop(columns=['id'])

# Make the plots.

fig0= plt.figure()
fig0.set_size_inches(16, 12)
x = [i for i in range(12,81,1)]
_ = plt.plot(x, result)
_ = fig0.legend([j for j in result.columns], loc='center left')
_ = plt.xlabel('Age') 
_ = plt.ylabel('Probability')
fig = Gcf.get_all_fig_managers()[-1].canvas.figure
image_data = ("data:image/png;base64,%s" % b64encode(print_figure(fig))
    .decode("utf-8"))
Gcf.destroy_fig(fig)
HTML(
"""
<b><center> Figure 1.</b> 
<em>The plot of predicted probability of a permanent tooth with age(12-80).
</em><br> <img src='%s'> <br>
"""
% image_data
)

fig0, ax0 = plt.subplots(nrows=4, ncols=2, sharex=False, sharey=False)
fig0.set_size_inches(16, 12)
fig0.tight_layout() 
x = [i for i in range(12,81,1)]
for i in range(8):
    lst = [result.columns[i], result.columns[15-i], 
           result.columns[16+i], result.columns[-i-1]]
    ax0[i//2,i%2].plot(x, result.loc[:,lst])
    ax0[i//2,i%2].legend(lst, loc='upper right')
fig = Gcf.get_all_fig_managers()[-1].canvas.figure
image_data = ("data:image/png;base64,%s" % b64encode(print_figure(fig))
    .decode("utf-8"))
Gcf.destroy_fig(fig)
HTML(
"""
<b><center> Figure 2.</b> 
<em>The plot of predicted probability of a permanent tooth with age(12-80).
</em>The subplot display the information of every four teeth. The select of 
teeth in one specific subplot is according to the universal numbering system.
<br> <img src='%s'> <br>
"""
% image_data
)

# +
fig0, ax0 = plt.subplots(nrows=8, ncols=4, sharex=False, sharey=False)
fig0.set_size_inches(16, 12)
fig0.tight_layout() 
x = [i for i in range(12,81,1)]
lst = []
ylab = ['Third Molar', 'Second Molar', 'First Molar', 'Second Bicuspid',
        'First Bicuspid', 'Cuspid', 'Lateral incisor', 'Central incisor']
for i in range(8):
    lst.extend([result.columns[i], result.columns[15-i], 
           result.columns[16+i], result.columns[-i-1]])
    
for i in range(32):
    ax0[i//4,i%4].plot(x, result.loc[:,lst[i]], color = 'blue')
    #ax0[i//4,i%4].legend(lst[i], loc='upper right', fontsize = 'medium')
    if i % 4 == 0:
        ax0[i//4,0].set_ylabel(ylab[i//4])
ax0[0,0].set_title('upper left')
ax0[0,1].set_title('upper right')
ax0[0,2].set_title('lower left')
ax0[0,3].set_title('lower right')

fig = Gcf.get_all_fig_managers()[-1].canvas.figure
image_data = ("data:image/png;base64,%s" % b64encode(print_figure(fig))
    .decode("utf-8"))
Gcf.destroy_fig(fig)
HTML(
"""
<b><center> Figure 3.</b> 
<em>The plot of predicted probability of a permanent tooth with age(12-80).
</em> The subplot display the information of each tooth. The select of 
teeth in specific row and column is based on the universal numbering
system.
<br> <img src='%s'> <br>
"""
% image_data
)
# -

# ## Question 2 - Hosmer-Lemeshow Calibration Plot

# Make the prediction for my best model.

ohx0['est'] = mod_comb.predict(params=res_comb.params, exog=mod_comb.exog)
new = ohx0.loc[:,['ohx01tc','age','est']]

# Sort the data according to the fitted value.

new.sort_values(['est'],inplace=True)

# Split the data into deciles based on the fitted probabilities.

sub_df = {}
for i in range(10):
    if i != 9:
        sub_df[i] = new.iloc[2531*i : 2531*i + 2531]
    else:
        sub_df[i] = new.iloc[2531*i :]

# Compute the observed proportion of cases with a permanent tooth present and
# the expected proportion found by averaging the probabilities.

obs = []
est = []
for i in range(10):
    obs_sub = sub_df[i]['ohx01tc'].sum()/sub_df[i]['ohx01tc'].count()
    est_sub = sub_df[i]['est'].mean()
    obs.append(obs_sub)
    est.append(est_sub)

# Make the Hosmer-Lemeshow Calibration Plot.

fig0 = plt.figure()
fig0.tight_layout() 
_ = plt.scatter(obs, est, color='blue')
_ = plt.plot([0,0.51],[0,0.51], color='red', linestyle='dashed')
_ = plt.xlabel('Expected probabilities') 
_ = plt.ylabel('Observed probabilities')
fig = Gcf.get_all_fig_managers()[-1].canvas.figure
image_data = ("data:image/png;base64,%s" % b64encode(print_figure(fig))
    .decode("utf-8"))
Gcf.destroy_fig(fig)
HTML(
"""
<b><center> Figure 4.</b> 
<em>Scatter plot of observed probabilities versus expected probabilities.</em>
The x axis represent the expected probability calculated from the origin data,
the y axis represnt the observed probability according to the predicted value 
of logistic regression. The plot also include a line through the origin with 
slope 1 as a guide.<br> <img src='%s'> <br>
"""
% image_data
)

# **Comment:** Most of the plots approximately fall on the line with slope
# equal to 1, but there are still some plots fall above or below the line. To
# conclude, I believe that the model I selected is considered well-calibrated.
