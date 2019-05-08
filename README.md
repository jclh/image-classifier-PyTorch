# Finding Donors for *CharityML*
## Exercise evaluating classifiers in `scikit-learn`
### [Udacity Program: Data Science, Project 1](https://github.com/udacity/DSND_Term1)
---


<img src="screen-example.png" width="768" alt="Example" />

In this project I use several classification algorithms included in the **scikit-learn** package to model person-level income — using data collected from the 1994 U.S. Census — and predict whether an individual makes more than \$50,000 per year. 

First, I obtain preliminary results from a small set of algorithms. Second, I choose the best candidate algorithm and further optimize it to improve performance.

Main files in the repository:

- `census.csv`: 1994 U.S. Census data; 14 features for 45,222 individuals. Extraction was done by Barry Becker from the Census database. A set of reasonably clean records was extracted using the following conditions: ((AAGE>16) && (AGI>100) && (AFNLWGT>1)&& (HRSWK>0)). More details [here](https://archive.ics.uci.edu/ml/datasets/Census+Income).

- `find_donors-sklearn.ipynb`: Jupyter notebook including main Python code.

- `visuals.py`: Plotting functions.


## Economic or Business question

The goal of the project is to build a tool that predicts whether an individual makes more than \$50,000 per year. This sort of task can arise in a non-profit setting, where organizations survive on donations and have limited resources for fund-raising. Estimating people's income can help the non-profit make their fund-raising more cost-effective; for example, deciding whether to reach out at all to a potential donor.


## Data Science motivation

In this project I use several supervised algorithms included in the scikit-learn package to model person-level income using data collected from the 1994 U.S. Census. First, I obtain preliminary results from a set of algorithms. Second, I choose the best candidate algorithm and further optimize it to best model the data.


## Usage Example

The Jupyter Project highly recommends new users to install [Anaconda](https://www.anaconda.com/distribution/); since it conveniently installs Python, the Jupyter Notebook, and other commonly used packages for scientific computing and data science.

Use the following installation steps:

1. Download Anaconda.

2. Install the version of Anaconda which you downloaded, following the instructions on the download page.

3. To run the notebook:

```
jupyter notebook find_donors-sklearn.ipynb
```

## Python version

3.7.1 (default, Oct 23 2018, 14:07:42) 


## Python libraries

The Jupyter Notebook file, `find_donors-sklearn.ipynb`, requires the following Python libraries:

- IPython
- matplotlib
- numpy
- pandas
- sklearn
- sys
- time
- warnings


## 1994 Census data

> - `census.csv`: 1994 U.S. Census data; 14 features for 45,222 individuals.

### Outcome

- `'income'`: >50K, <=50K. 

### Attribute information

- `'age'`: continuous.
- `'workclass'`: Private, Self-emp-not-inc, Self-emp-inc, Federal-gov, Local-gov, State-gov, Without-pay, Never-worked. 
- `'education'`: Bachelors, Some-college, 11th, HS-grad, Prof-school, Assoc-acdm, Assoc-voc, 9th, 7th-8th, 12th, Masters, 1st-4th, 10th, Doctorate, 5th-6th, Preschool. 
- `'education-num'`: continuous. 
- `'marital-status'`: Married-civ-spouse, Divorced, Never-married, Separated, Widowed, Married-spouse-absent, Married-AF-spouse. 
- `'occupation'`: Tech-support, Craft-repair, Other-service, Sales, Exec-managerial, Prof-specialty, Handlers-cleaners, Machine-op-inspct, Adm-clerical, Farming-fishing, Transport-moving, Priv-house-serv, Protective-serv, Armed-Forces. 
- `'relationship'`: Wife, Own-child, Husband, Not-in-family, Other-relative, Unmarried. 
- `'race'`: White, Asian-Pac-Islander, Amer-Indian-Eskimo, Other, Black. 
- `'sex'`: Female, Male. 
- `'capital-gain'`: continuous. 
- `'capital-loss'`: continuous. 
- `'hours-per-week'`: continuous. 
- `'native-country'`: United-States, Cambodia, England, Puerto-Rico, Canada, Germany, Outlying-US(Guam-USVI-etc), India, Japan, Greece, South, China, Cuba, Iran, Honduras, Philippines, Italy, Poland, Jamaica, Vietnam, Mexico, Portugal, Ireland, France, Dominican-Republic, Laos, Ecuador, Taiwan, Haiti, Columbia, Hungary, Guatemala, Nicaragua, Scotland, Thailand, Yugoslavia, El-Salvador, Trinadad&Tobago, Peru, Hong, Holand-Netherlands.


## Acknowledgments

- [Udacity: Data Scientist Nanodegree program](https://www.udacity.com/course/data-scientist-nanodegree--nd025).
- Jupyter Documentation: [Installing Jupyter Notebook](https://jupyter.readthedocs.io/en/latest/install.html)


## Author

Juan Carlos Lopez
[GitHub](https://github.com/jclh/)
[LinkedIn](https://www.linkedin.com/in/jclopezh/)
jc.lopezh@gmail.com


## Contributing

1. Fork it (https://github.com/jclh/finding-donors-classifier/fork)
2. Create your feature branch (git checkout -b feature/fooBar)
3. Commit your changes (git commit -am 'Add some fooBar')
4. Push to the branch (git push origin feature/fooBar)
5. Create a new Pull Request




























