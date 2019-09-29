'''
@Description: 
@Version: 
@School: Tsinghua Univ
@Date: 2019-09-19 09:59:30
@LastEditors: Xie Yufeng
@LastEditTime: 2019-09-22 23:52:45
'''
#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File    :   sta.py
@Time    :   2019/09/19 09:59:37
@Author  :   Xie Yufeng 
@Version :   1.0
@Contact :   xyfzkd@outlook.com
@Desc    :   None
'''

# -*- coding: utf-8 -*-
#%%
import pandas as pd
raw = pd.read_excel('raw.xlsx')

#%%
import seaborn as sns
sns.set_style("darkgrid")
#sns.set(style="ticks")
g = sns.pairplot(raw,hue="sex",diag_kind='hist')

#%%
g.savefig('cov.pdf',facecolor='white')

#%%
from pywaffle import Waffle
import pandas as pd
import matplotlib.pyplot as plt
#%%
df_age = raw.groupby('age').size().reset_index(name='counts_age')
n_categories = df_agef.shape[0]
colors_age = [plt.cm.Set3(i/float(n_categories)) for i in range(n_categories)]
fig = plt.figure(
    FigureClass=Waffle,
    plots={
        '111': {
            'values': df_age['counts_age'],
            'labels': ["{1}".format(n[0], n[1]) for n in df_age[['age', 'counts_age']].itertuples()],
            'legend': {'loc': 'upper left', 'bbox_to_anchor': (1.05, 1), 'fontsize': 12, 'title':'Age'},
            'title': {'label': '# Vehicles by Age', 'loc': 'center', 'fontsize':18},
            'colors': colors_age
        }
    },
    rows=12,
    figsize=(16, 10)
)
fig.savefig("waffe.pdf",transparent=True)
#%%
df_age = raw.groupby('sex').size().reset_index(name='counts_age')
n_categories = df_age.shape[0]
colors_age = [plt.cm.Set3(i/float(n_categories)) for i in range(n_categories)]
fig = plt.figure(
    FigureClass=Waffle,
    plots={
        '111': {
            'values': df_age['counts_age'],
            'labels': ["{1}".format(n[0], n[1]) for n in df_age[['sex', 'counts_age']].itertuples()],
            'legend': {'loc': 'upper left', 'bbox_to_anchor': (1.05, 1), 'fontsize': 12, 'title':'Gender'},
            'title': {'label': '# Vehicles by Age', 'loc': 'center', 'fontsize':18},
            'colors': colors_age
        }
    },
    rows=12,
    figsize=(16, 10)
)
fig.savefig("waffle_sex.pdf",transparent=True)
#%%
df_agef = raw[raw['sex']=='F'].groupby('age').size().reset_index(name='counts_age')
df_agem = raw[raw['sex']=='M'].groupby('age').size().reset_index(name='counts_age')
n_categoriesf = df_agef.shape[0]
n_categoriesm = df_agem.shape[0]
colors_agef = [plt.cm.Set3(i/float(n_categoriesf)) for i in range(n_categoriesf)]
colors_agem = [plt.cm.Set3(i/float(n_categoriesm)) for i in range(n_categoriesm)]
fig = plt.figure(
    FigureClass=Waffle,
    plots={
        '211': {
            'values': df_agef['counts_age'],
            'labels': ["{1}".format(n[0], n[1]) for n in df_agef[['age', 'counts_age']].itertuples()],
            'legend': {'loc': 'upper left', 'bbox_to_anchor': (1.05, 1), 'fontsize': 12, 'title':'Age'},
            'title': {'label': '# Vehicles by Age', 'loc': 'center', 'fontsize':18},
            'colors': colors_agef
        },
        '212': {
            'values': df_agem['counts_age'],
            'labels': ["{1}".format(n[0], n[1]) for n in df_agem[['age', 'counts_age']].itertuples()],
            'legend': {'loc': 'upper left', 'bbox_to_anchor': (1.05, 1), 'fontsize': 12, 'title':'Age'},
            'title': {'label': '# Vehicles by Age', 'loc': 'center', 'fontsize':18},
            'colors': colors_agem
        }
    },
    columns=6,
    figsize=(16, 10)
)
#g.savefig('1.pdf',facecolor='white')
#%%
raw
#%%
from scipy import stats
fig,ax = plt.subplots()
scipy.stats.probplot(raw['pred_age'],dist='norm',plot=ax,fit=True)
#%%
import probscale
def equality_line(ax, label=None):
    limits = [
        np.min([ax.get_xlim(), ax.get_ylim()]),
        np.max([ax.get_xlim(), ax.get_ylim()]),
    ]
    ax.set_xlim(limits)
    ax.set_ylim(limits)
    ax.plot(limits, limits, 'k-', alpha=0.75, zorder=0, label=label)

norm = stats.norm(loc=21, scale=8)
fig, ax = plt.subplots(figsize=(5, 5))
ax.set_aspect('equal')

common_opts = dict(
    plottype='qq',
    probax='x',
    problabel='Theoretical Quantiles',
    datalabel='Emperical Quantiles',
    scatter_kws=dict(label='Bill amounts')
)

fig = probscale.probplot(raw['pred_age'], ax=ax, dist=norm, **common_opts)

equality_line(ax, label='Guessed Normal Distribution')
ax.legend(loc='lower right')
sns.despine()
#%%
fig.savefig('norm.pdf',edgecolor='black',transparent=False)

#%%
import seaborn as sns
import numpy as np
x = np.linspace(min(raw['pred_age']), max(raw['pred_age']), 50)
y = 239*1/(3.82 * np.sqrt(2 * np.pi))*np.exp( - (x - 45.12)**2 / (2 * 3.82**2))
plt.plot(x,y)
plt.hist(raw['pred_age'],bins=int(max(raw['pred_age'])-min(raw['pred_age'])))
plt.savefig('normbin.pdf')
#%%
max(raw['pred_age'])-min(raw['pred_age'])

#%%
import seaborn as sns
import numpy as np 
import matplotlib.mlab as mlab 
import matplotlib.pyplot as plt
x = raw['pred_age']
mu =np.mean(x)
sigma =np.std(x,ddof=1)

num_bins = int(max(raw['pred_age'])-min(raw['pred_age']))
n, bins, patches = plt.hist(x, num_bins,normed=1, facecolor='blue', alpha=0.5) 
y = mlab.normpdf(bins, mu, sigma)
plt.plot(bins, y, 'r--')
plt.xlabel('Age Estimation') #绘制x轴 
plt.ylabel('Probability') #绘制y轴 
plt.title(r'Normal Distribution Fit: $\mu=%.1f$,$\sigma=%.1f$'%(mu,sigma))
plt.savefig('norm_fit.pdf')
#%%
def qqplot(sample=raw['pred_age']):
    import numpy as np
    x = sample
    mu =np.mean(x)
    sigma =np.std(x,ddof=1)
    from scipy.stats import norm,percentileofscore
    samp_pct = [percentileofscore(x, i) for i in x]
    fit_pct = [norm.cdf(i,mu,sigma)*100 for i in x]
    import matplotlib.pyplot as plt
    plt.scatter(x=samp_pct, y=fit_pct)
    linex = np.arange(0, 100, 1)
    liney = np.arange(0, 100, 1)
    plt.plot(linex, liney, 'r--')
    plt.xlabel('Sample Percentage%') #绘制x轴 
    plt.ylabel('Fit Percentage%') #绘制y轴 
    plt.title(r'Q-Q plot')
    plt.savefig('qqplot.pdf')
qqplot()


#%%
import scipy.stats as stats
x = raw['pred_age']
mu =np.mean(x)
sigma =np.std(x,ddof=1)
normed_data=(x-mu)/sigma
print(stats.kstest(normed_data,'norm'))
#%%
import scipy.stats as stats
x = raw['pred_age']
sp_x=x.tolist()
sp_x.sort()
sp_x = sp_x[2:]
mu =np.mean(sp_x)
sigma =np.std(sp_x,ddof=1)
normed_data=(sp_x-mu)/sigma
print(stats.kstest(normed_data,'norm'))

#%%

import scipy.stats as stats
x = raw[raw['sex']=='M']['pred_age']
mu =np.mean(x)
sigma =np.std(x,ddof=1)
normed_data=(x-mu)/sigma
print(stats.kstest(normed_data,'norm'))
#%%
import scipy.stats as stats
x = raw[raw['sex']=='F']['pred_age']
sp_x=x.tolist()
sp_x.sort()
sp_x = sp_x[2:]
mu =np.mean(sp_x)
sigma =np.std(sp_x,ddof=1)
normed_data=(sp_x-mu)/sigma
print(stats.kstest(normed_data,'norm'))

#%%
import scipy.stats as stats
import pandas as pd
pd.DataFrame(raw.groupby('sex').describe()).to_csv('des_sex.csv',sep=',')
pd.DataFrame(raw.groupby('age').describe()).to_csv('des.csv',sep=',')

#%%
from scipy import stats

F, p = stats.f_oneway(d_data['ctrl'], d_data['trt1'], d_data['trt2'])
#%%
for i,j in raw.groupby('sex'):
    print(i)

#%%
[j for i,j in raw.groupby('sex')].values()

#%%
archive = {'group1': np.array([ 1, 2, 3 ]),
        'group2': np.array([ 9, 8, 7])}

#%%


#%%
import scipy
archive = {i:j['pred_age'].tolist() for i,j in raw.groupby('sex')}
scipy.stats.f_oneway(*archive.values())

#%%
import seaborn as sns
import matplotlib.pyplot as plt
fig,ax = plt.subplots()
ax = sns.stripplot(y='sex',x='pred_age',data=raw)
fig.savefig('sex.pdf')
#%%
import scipy
archive = {i:j['pred_age'].tolist() for i,j in raw.groupby('age')}
scipy.stats.f_oneway(*archive.values())

#%%
import seaborn as sns
import matplotlib.pyplot as plt
fig,ax = plt.subplots()
ax = sns.stripplot(x='age',y='pred_age',data=raw)
fig.savefig('age.pdf')


#%%
