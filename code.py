"""Code for the article"""

# Libraries used
print('Importing libraries')
import pathlib
import typing as t
import pandas as pd
import numpy as np
import sklearn.datasets as skd
import sklearn.linear_model as lm
import matplotlib.pyplot as plt
import seaborn as sns

# Basic plot configuration
sns.set(style='white', rc={'figure.figsize': (11.7, 8.27)})

# A path to save artifacts
p = pathlib.Path(__file__).parent

# Some functions to help
def tstat(*, x0: pd.Series, x1: pd.Series) -> float:
    """ Calculates Welch's t statistic. This is a modified version of Student's t for the case 
        where samples are independent (control and treatment groups) and each has a different 
        variance. For more information, the Wikipedia articule on Welch's t test is a good source
    """
    return (x0.mean() - x1.mean()) / np.sqrt(x0.var() / len(x0) + x1.var() / len(x1))

def center(*, x0: pd.Series, x1: pd.Series) -> t.Tuple[pd.Series, pd.Series]:
    """ Subtracts the mean of x0 and x1. This function is used to force the validity of the null 
        hypothesis before applying the bootstrap algorithm
    """
    return (
        x0 - x0.mean(),
        x1 - x1.mean()
    )

def split_by_class(*, df: pd.DataFrame, y_name: str, score: str) -> t.Tuple[pd.Series, pd.Series]:
    """ Splits the dataset by the class labels. This results in two separate arrays that make up 
        the two populations being tested
    """
    return (
        df[df[y_name] == 0][score].copy(),
        df[df[y_name] == 1][score].copy(),
    )

def bootstrap(*,
        x0_h0: pd.Series,
        x1_h0: pd.Series,
        n: int = 20,
        samples: int = 10_000
    ) -> t.List[float]:
    """ The bootstrap algorithm. The suffix `h0` is to remember you that the bootstrapping is 
        applied to arrays such that the null hypothesis (commonly denoted H0) is true
    """
    t_dist = []
    for _ in range(samples):
        s0 = x0_h0.sample(n=n, replace=True)
        s1 = x1_h0.sample(n=n, replace=True)
        t_dist.append(tstat(x0=s0, x1=s1))
    return t_dist

# Simulating data for the good score
print('Simulating data for good score')
X, y = skd.make_classification(
    n_samples=1000,
    n_features=5,
    n_informative=5,
    n_redundant=0,
    random_state=0,
)

# Calculating scores using logistic regression
print('Calculating good score')
scores_good = pd.Series(
    data=lm.LogisticRegression().fit(X, y).predict_proba(X)[:, 0],
    name='score',
).mul(1000).round(0).astype(int) # typical format of a score: between 0-1000 and an integer

# Creating dataset for the good score
print('Creating dataset for case study of good score')
df_good = pd.concat(objs=[pd.Series(data=y, name='y'), scores_good], axis='columns')

# Figure 1: distribution for a good score
print('Creating Figure 1')
sns.kdeplot(
    data=df_good,
    x='score',
    hue='y',
    fill=True,
    legend=False,
).set(xlabel='Score', ylabel=None, yticklabels=[])
plt.savefig(p.joinpath('fig1_good_score_dist.png'), dpi=800)
plt.close()

# Creating a bad score
print('Simulating data for bad score')
np.random.seed(0)
scores_bad = pd.Series(data=np.random.normal(500, 100, size=1000), name='score')

# Data for bad score
print('Creating dataset for case study of bad score')
df_bad = pd.concat(objs=[pd.Series(data=y, name='y'), scores_bad], axis='columns')

# Figure 2: distribution for a bad score
print('Creating Figure 2')
sns.kdeplot(
    data=df_bad,
    x='score',
    hue='y',
    fill=True,
    legend=False,
).set(xlabel='Score', ylabel=None, yticklabels=[])
plt.savefig(p.joinpath('fig2_bad_score_dist.png'), dpi=800)
plt.close()

### Testing the good score ###
# Separating by classes
print('Separating good score by classes')
x0_good, x1_good = split_by_class(df=df_good, y_name='y', score='score')

# Centering data to ensure the null hypothesis is true
print('Centering data for good score')
x0_good_h0, x1_good_h0 = center(x0=x0_good, x1=x1_good)

# Bootstrapping
print('Bootstrapping for good score')
t_dist_good = bootstrap(x0_h0=x0_good_h0, x1_h0=x1_good_h0)

# Figure 3: distribution of the test statistic under the null hypothesis for a good score
print('Creating Figure 3')
sns.kdeplot(
    x=pd.Series(t_dist_good),
    fill=True,
).set(xlabel='Test statistic', ylabel=None, yticklabels=[])
plt.savefig(p.joinpath('fig3_t_dist_good_score.png'), dpi=800)
plt.close()

### Testing the bad score ###
# Separating by classes
print('Separating bad score by classes')
x0_bad, x1_bad = split_by_class(df=df_bad, y_name='y', score='score')

# Centering data to ensure the null hypothesis is true
print('Centering data for bad score')
x0_bad_h0, x1_bad_h0 = center(x0=x0_bad, x1=x1_bad)

# Bootstrapping
print('Bootstrapping for bad score')
t_dist_bad = bootstrap(x0_h0=x0_bad_h0, x1_h0=x1_bad_h0)

# Figure 4: distribution of the test statistic under the null hypothesis for a bad score
print('Creating Figure 4')
sns.kdeplot(
    x=pd.Series(t_dist_bad),
    fill=True,
).set(xlabel='Test statistic', ylabel=None, yticklabels=[])
plt.savefig(p.joinpath('fig4_t_dist_bad_score.png'), dpi=800)
plt.close()

### Test report ###
print('Writing test report')
content = [
    'TEST REPORT',

    '\nGOOD SCORE\n'
    '\tAverage of good clients under h0:  {:>8.2f}',
    '\tAverage of bad clients under h0:   {:>8.2f}',
    '\tVariance of good clients:          {:>8.2f}',
    '\tVariance of bad clients:           {:>8.2f}',
    '\tCritical value for test statistic: {:>8.2f}',
    '\tObserved value for test statistic: {:>8.2f}',

    '\nBAD SCORE\n'
    '\tAverage of good clients under h0:  {:>8.2f}',
    '\tAverage of bad clients under h0:   {:>8.2f}',
    '\tVariance of good clients:          {:>8.2f}',
    '\tVariance of bad clients:           {:>8.2f}',
    '\tCritical value for test statistic: {:>8.2f}',
    '\tObserved value for test statistic: {:>8.2f}',
]
with open(p.joinpath('report.txt'), 'w') as report:
    report.write('\n'.join(content).format(
        x0_good_h0.mean(),
        x1_good_h0.mean(),
        x0_good_h0.var(),
        x1_good_h0.var(),
        np.percentile(t_dist_good, 95),
        tstat(x0=x0_good, x1=x1_good),
        x0_bad_h0.mean(),
        x1_bad_h0.mean(),
        x0_bad_h0.var(),
        x1_bad_h0.var(),
        np.percentile(t_dist_bad, 95),
        tstat(x0=x0_bad, x1=x1_bad)
    ))