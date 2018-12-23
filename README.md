# Data Preprocessing Project

# Dealing with missing numerical values

In this project, I explore various Data Preprocessing techniques to deal with missing numerical values. as listed below:-


## Introduction

Over the last few decades, Machine Learning (ML) has gained immense popularity in solving real world business problems. It has emerged as a technology tool for companies looking to boost productivity and profit. ML practitioners source real world data and write algorithms to solve business problems. The success of the ML algorithms depends on the quality of the data. The data must be free from errors and discrepancies. It must adhere to a specific standard so that ML algorithms can accept them. But, this does not happen in reality.

In reality, the data has its own limitations. The data is dirty. It is incomplete, noisy and inconsistent.  Incomplete data means it has missing values and lacks certain attributes. The data may be noisy as it contains errors and outliers and hence does not produce desired results. Lastly, the data may be inconsistent as it contains discrepancies in data or duplicate data.

So, ML practitioners must take steps to transform raw data into standardized data that is suitable for ML algorithms.  It involves cleaning, transforming and standardizing data to remove all the inadequacies and irregularities in the data. These steps are collectively known as Data Preprocessing.   








I will discuss these data preprocessing tasks in detail and write code snippets to deal with each task.  







## 1. Dealing with missing values

It is a very common scenario that when looking at a real world data, a data scientist may come across missing values. These missing values could be due to error prone data entry process, wrong data collection methods, certain values not applicable, particular fields left blank in a survey or the respondent decline to answer. Whatever may be the reason for the missing value, the data scientist must find ways to handle these missing values. He knows that missing values need to be handled carefully, because they give wrong results if we simply ignore them. He must answer whether he should delete these missing values or replace them with a suitable statistic. The first step in dealing with missing values properly is to identify them. 

### Identify missing values
Missing values are encoded in different ways. They can appear as ‘NaN’, ‘NA’, ‘?’, zero ‘0’, ‘xx’, minus one ‘-1’ or a blank space “ ”. We need to use various pandas methods to deal with missing values. But, pandas always recognize missing values as ‘NaN’.  So, it is essential that we should first convert all the ‘?’, zeros ‘0’, ‘xx’, minus ones ‘-1’ or blank spaces “ ” to ‘NaN’. If the missing values isn’t identified as ‘NaN’, then we have to first convert or replace such non ‘NaN’ entry with a ‘NaN’.



### Convert '?' to ‘NaN’

`df[df == '?'] = np.nan`

The initial inspection of the data help us to detect whether there are missing values in the data set. It can be done by Exploratory Data Analysis. So, it is always important that a data scientist always perform Exploratory Data Analysis (EDA) to identify missing values correctly.





### Exploratory Data Analysis (EDA)

Below is the list of commands to identity missing values with EDA.

1.	`df.head()`

This will output the first five rows of the dataset. It will give us quick view on the presence of ‘NaN’ or ‘?’ ‘-1’ or ’0’ or blank spaces “” in the dataset. If required, we can view more number of rows by specifying the number of rows inside the parenthesis. 



2.	`df.info()`

This command is quite useful in detecting the missing values in the dataset. It will tell us the total number of non - null observations present including the total number of entries. Once number of entries isn’t equal to number of non - null observations, we know there are missing values in the dataset.



3.	`df.describe()`

This will display summary statistics of all observed features and labels. The most important statistic is the minimum value. If we see -1 or 0 in our observations, then we can suspect missing value.




4.	`df.isnull()`

The above command checks whether each cell in a dataframe contains missing values or not. If the cell contains missing value, it returns True otherwise it returns False. 

5.	`df.isnull.sum()`

The above command returns the total number of missing values in each column in the dataset.


6.	isna() and notna() functions to detect ‘NA’ values

Pandas provides isna() and notna() functions to detect ‘NA’ values. These are also methods on Series and DataFrame objects.

Examples of isna() and notna() commands


### detect ‘NA’ values in the dataframe	

`df.isna()`

### detect ‘NA’ values in a particular column in the dataframe

`pd.isna(df[‘col_name’])`

`df[‘col_name’].notna()`

	
### Handle missing values

There are several methods to handle missing values. Each method has its own advantages and disadvantages. The choice of the method is subjective and depends on the nature of data and the missing values. The summary of the options available for handling missing values is given below:-

1.	**Drop missing values with dropna()**

2.	**Fill missing values with a test statistic**

3.	**Fill missing values with Imputer**

4.	**Build a Prediction Model**

5.	**KNN Imputation**


I have discussed each method in below sections:-

### 1. Drop missing values

This is the easiest method to handle missing values. In this method, we drop labels or columns from a data set which refer to missing values. 

### drop labels or rows from a data set containing missing values

`df.dropna (axis = 0)`


 ### drop columns from a data set containing missing values

`df.dropna(axis = 1)`


**A note about axis parameter** 

Axis value may contain (0 or ‘index’) or (1 or ‘columns’). Its default value is 0.

We set axis = 0 or ‘index’ to drop rows which contain missing values.

We set axis = 1 or ‘columns’ to drop columns which contain missing values.


This is the Pandas dataframe dropna() method. An equivalent dropna() method is available for Series with same functionality.

After dropping the missing values, we can again check for missing values and the dimensions of the dataframe.


### again check the missing values in each column

`df.isnull.sum()`  

### again check the dimensions of the dataset

`df.shape`

But, this method has one disadvantage. It involves the risk of losing useful information. Suppose there are lots of missing values in our dataset. If drop them, we may end up throwing away valuable information along with the missing data. It is a very grave mistake as it involves losing key information. So, it is only advised when there are only few missing values in our dataset.
So, it's better to develop an imputation strategy so that we can impute missing values with the mean or the median of the row or column containing the missing values.




## 2.	Fill missing values with a test statistic

In this method, we fill the missing values with a test statistic like mean, median or mode of the particular feature the missing value belongs to. One can also specify a forward-fill or back-fill to propagate the next values backward or previous value forward.


### Filling missing values with a test statistic like mean

`median = df['col_name'].median()`

`df['col_name'].fillna(value = median, inplace = True )`


### We can also use replace() in place of fillna()

`df[‘col_name’].replace(to_replace = NaN, value = median, inplace = True)`


If we choose this method, then we should compute the median value on the training set and use it to fill the missing values in the training set. Then we should save the median value that we have computed.  Later, we will replace missing values in the test set with the median value to evaluate the system.




## 3.	Fill missing values with Imputer

Scikit-Learn provides Imputer class to deal with the missing values. In this method, we replace the missing value with the mean value of the entire feature column. This can be done as shown in the following code:

`from sklearn.preprocessing import Imputer

imp = Imputer(missing_values='NaN',  strategy='mean', axis=0)

imp = imp.fit(df.values)

imputed_data = imp.transform(df.values)

imputed_data`


Here, I have replaced each ‘NaN’ value with the corresponding mean value. The mean value is separately calculated for each feature column. If instead of axis = 0, we set axis = 1, then mean values are calculated for each row. 
Other options for strategy parameter are ‘median’ or ‘most_frequent’. The ‘most_frequent’ parameter replaces the missing values with the most frequent value. It is useful for imputing categorical feature values.



## 4.	Build a Prediction Model

We can build a Prediction Model to handle missing values. In this method, we divide our data set into two sets – training set and test set. Training set does not contain any missing values and test set contains missing values. The variable containing missing values can be treated as a target variable. Next, we create a model to predict target variable and use it to populate missing values of test data set. 



## 5.	KNN Imputation

In this method, the missing values of an attribute are imputed using the given number of attributes that are mostly similar to the attribute whose values are missing. The similarity of attributes is determined using a distance function.



## Check with ASSERT statement

Finally, we can check for missing values programmatically. If we drop or fill missing values, we expect no missing values. We can write an assert statement to verify this. So, we can use an assert statement to programmatically check that no missing or unexpected ‘0’ value is present. This gives confidence that our code is running properly.

Assert statement will return nothing if the value being tested is true and will throw an AssertionError if the value is false.

Asserts

•	assert 1 == 1   (return Nothing if the value is True)

•	assert 1 == 2   (return AssertionError if the value is False)


### assert that there are no missing values in the dataframe

`assert pd.notnull(df).all().all()`


### assert that there are no missing values for a particular column in dataframe

`assert df.column_name.notnull().all()`


### assert all values are greater than 0

`assert (df >=0).all().all()`


### assert no entry in a column is equal to 0

`assert (df['column_name']!=0).all().all()`



This concludes our discussion on missing numerical values.







