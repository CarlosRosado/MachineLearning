import matplotlib.pyplot as plt
import pandas
import numpy as np
import seaborn as sns
from collections import Counter
from sklearn.linear_model import LinearRegression
import statsmodels.formula.api as smf


def plotValues_num(type):

	plt.hist(salary[type],20)
	plt.title("%s Histogram" % type)
	plt.xlabel("Value")
	plt.ylim([0,7])
	plt.ylabel("Frequency")
	plt.show()

def plotValues_str(type):

	letter_counts = Counter(salary[type])
	df = pandas.DataFrame.from_dict(letter_counts, orient='index')
	df.plot(kind='bar')
	plt.title("%s Histogram" % type)
	plt.xlabel("Value")
	plt.ylabel("Frequency")
	plt.show()

def stadisticValue(salary):

	print "\nMax:"
	print salary.max()
	print "\nMin:"
	print salary.min()
	print "\nMean:"
	print salary.mean()
	print "\nSTD:"
	print salary.std()
	print "\nMedian:"
	print salary.median()

def linearRegression(salary,op1,op2):

	typ = "%s ~ %s" % (op1,op2)
	lm1 = smf.ols(formula=typ, data=salary).fit()
	print lm1.params

	feature_cols = [op2]
	X = salary[feature_cols]
	y = salary.Salary

	# instantiate and fit
	lm2 = LinearRegression()
	lm2.fit(X, y)

	# print the coefficients
	print lm2.intercept_
	print lm2.coef_

	if op1 == 'Salary':
		sns.pairplot(salary, x_vars=['Experience','Education_Coded','Management_Coded'], y_vars='Salary', size=7, aspect=0.7, kind='reg')
	else:
		sns.pairplot(salary, x_vars=['Experience','Education_Coded','Salary'], y_vars='Management_Coded', size=7, aspect=0.7, kind='reg')

	sns.plt.show()

	#Example
	X_new = pandas.DataFrame({'Education_Coded': [20]})

	# predict for a new observation
	print "Predict:"
	print lm2.predict(X_new)


#Define a generic function using Pandas replace function
def nominalToNum(col, codeDict):
  colCoded = pandas.Series(col, copy=True,dtype=int)
  for key, value in codeDict.items():
    colCoded.replace(key, value, inplace=True)
  return colCoded


if __name__ == '__main__':
	
	#Read the dataset
	salary = pandas.read_csv("salary.csv")

	print "Plot Colums:"
	#Plot columns
	plotValues_num("Salary")
	plotValues_num("Experience")
	plotValues_str("Education")
	plotValues_str("Management")

	#Nominal to numeric
	salary["Education_Coded"] = nominalToNum(salary["Education"], {'Bachelor':0,'Master':1,'PhD':2})
	salary["Management_Coded"] = nominalToNum(salary["Management"], {'False':0,'True':1})

	#Calculate stadistical values
	print "Stadistical Values: \n"
	stadisticValue(salary)

	print "\nLinear Regression Salary vs Experience:\n"
	linearRegression(salary,"Salary","Experience")
	print "\nLinear Regression Salary vs Education:\n"
	linearRegression(salary,"Salary","Education_Coded")
	print "\nLinear Regression Salary vs Management:\n"
	linearRegression(salary,"Salary","Management_Coded")


	print "\nLinear Regression Management vs salary:\n"
	linearRegression(salary,"Management_Coded","Salary")
	print "\nLinear Regression Management vs Experience:\n"
	linearRegression(salary,"Management_Coded","Experience")
	print "\nLinear Regression Salary vs Education:\n"
	linearRegression(salary,"Management_Coded","Education_Coded")




