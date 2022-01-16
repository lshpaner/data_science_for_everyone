
##########################################################################
#### R/RStudio Introductory Workshop by University of San Diego (USD)#####
####                      Date: January 22, 2022                     #####  
##########################################################################


# The following link shows what each quadrant in the RStudio IDE represents:
# https://www.leonshpaner.com/teaching/post/rstudio/

# Let us start with the basics.

####################################
#### General/Console Operations #### 
###################################

# In R, we can simply type commands in the console window and hit "enter." 
# This is how they are executed. For example, R can be used as a calculator.
2 + 2
3 * 3
sqrt(9)
log10(100)


## What is a string? ##

# A string is simply any open alpha/alphanumeric text characters surrounded by either single
# quotation marks or double quotation marks, with no preference assigned for either single or double
# quotation marks, with a print statement being called prior to the strings. For example,

print('This is a string.')
print( "This is also a string123.")

## Determining and setting the current working directory ## 

# The importance of determining and setting the working directory cannot be stressed enough. 
# Obtain the path to the working directory by running the `getwd()` function. Set the working 
# directory by running the `setwd("...")` function, filling the parentheses inside with the correct path.
getwd() 
setwd()


############################
#### Installing Packages ###
############################

# To install a package or library, simply type in `install.packages('package_name).` 
# For the following exercises, let us ensure that we have installed the following packages:
install.packages('psych') # psychological library as extension for statistical tools
install.packages("readr") # for reading rectangular data (i.e., 'csv', 'tsv', and 'fwf')
install.packages('summarytools')  # additional library for summarizing data
install.packages('caret') # classification and regression training (modeling)
install.packages('e1071') # other miscellaneous function
install.packages('rpart') # for classification and decision trees
install.packages('rpart.plot') # for plotting classification and decision trees
install.packages("cluster") # methods for cluster analysis
install.packages('factoextra') # clustering algorithms & visualization
# To read the documentation for any given library, simply put a "?" before the library name
# and press "enter." For example, 

library(summarytools) # load the library first
?summarytools # then view the documentation

# This will open the documentation in the 'Files, Plots, Packages. Help, Viewer' pane.


###########################################
#### Source Pane (Workspace) Scripting ####
###########################################


# Creating objects: 
# <- : assignment in R. Shortcut: "Alt + -" on Windows or "Option + -" on Mac.
var1 <- c(0, 1, 2, 3)

## Differences between "=" and "<-":
# Whereas "=" sign can also be used for assignment in R, it is best suited for 
# specifying field equivalents (i.e., number of rows, columns, etc.). 
# For example, let us take the following dataframe with 4 rows and 3 columns. 
# Here, we specify A=4 and B=3 as follows:

dataframe_1 <- c(A = 4, B = 3)
dataframe_1

# If we specify `A <- 4`, and `B <- 3` instead, `A` and `B` will effectively evaluate to those 
# respective values as opposed to listing them in those respective columns. Let us take a look:
dataframe_2 <- c(A <- 4, B <- 3)
dataframe_2

## Import data from flat .csv file
# Assuming that your file is located in the same working directory that you have specified at the onset of 
# this tutorial/workshop, make an assignment to a new variable (i.e., `ex_csv`) 
# and call read.csv() in the following generalized format

# for Windows users: 
ex_csv_1 <- read.csv(choose.files(), header = T, quote = "'")

# for macOS users:
# example1.6 = read.csv(file.choose(), header = T, quote = "'")

# The `choose.files()`, `file.choose()` function calls, respectively. allow the user to locate the file on their 
# local drive, regardless of the working directory. That being said, there is more than one way to read in a 
# `.csv` file. It does not have to exist on the local machine. If the file exists on the cloud, the path (link)
# can be parsed into the `read.csv()` function call as a string. 

# Now, let us print what is contained in `var 1`
print(var1)

# or we can simply call var1
var1

# Let us assign a variable to our earlier calculation and call the variable.
two <- 2+2
two

# Any object (variable, vector, matrix, etc.) that is created is stored in the R workspace. 
# From time to time, it is best practice to clear certain unused elements from the workspace 
# so that it does not congest the memory.
rm(two)

# when we proceed to type in two, we will see that there is an error, confirming 
# that the variable has been removed from the workspace.
two


# There are 4 main data types in R: numeric, character, factor, and logical. 
# Data classes
# Numeric (includes integer, decimal)
num <- 12.6
num

# Character (includes characters and strings):
char <- "Male"
char

# Factor (ordinal/categorical data)
gender <- as.factor(char)
gender

# Logical()
TRUE 
FALSE
T # abbreviation also works for Boolean object
F
TRUE * 7
FALSE * 7

#########################
#### Data Structures ####
#########################

##  What is a variable? 
# A variable is a container for storing a data value, exhibited as a reference to "to an 
# object in memory which means that whenever a variable is assigned
# to an instance, it gets mapped to that instance. A variable in R can store a vector, 
# a group of vectors or a combination of many R objects" (GeeksforGeeks, 2020). 


# There are 3 most important data structures in R: vector, matrix, and dataframe. 

## * Vector: the most basic type of data structure within R; contains a series of values of 
#            the same data class. It is a "sequence of data elements" (Thakur, 2018). 
#  * Matrix: a 2-dimensional version of a vector. Instead of only having a single row/list of data,
#            we have rows and columns of data of the same data class.
#  * Dataframe: the most important data structure for data science. Think of dataframe as loads of 
#               vectors pasted together as columns. Columns in a dataframe can be of different data 
#               class, but values within the same column must be the same data class.

#    The `c()` function is to R what `concatenate()` is to excel. For example,
vector_1 <- c(2,3,5)
vector_1

# Similarly, a vector of logical values will contain the following.
vector_2 <- c(TRUE, FALSE, TRUE, FALSE, FALSE)
vector_2

# To determine the number of members inside any given vector, we apply the `length()` 
# function call on the vector as follows:
length(c(TRUE, FALSE, TRUE, FALSE, FALSE))

# or, since we already assigned this to a data frame named `vector_2`, we can simply 
# call length of `vector_2` as follows:
length(vector_2)

# Let's say for example, that we want to access the third element of `vector_1.` We can do so as follows:
vector_1[3]

# What if we want to access all elements within the dataframe except for the first one? To this end, 
# we use the "-" symbol as follows:
vector_1[-1]

# Let us create a longer arbitrary vector so we can illustrate some further examples.
vector_3 <- c(1,3,5,7,9,20,2,8,10,35,76,89,207)

# Let us now say we want to access the first, fifth, and ninth elements of this dataframe. 
# To this end, we can do the following:
vector_3[c(1,5,9)]

# Now if we want to access all elements within a specified range, we can specify the exact range using 
# the ":" separator as follows:
vector_3[3:11]

## Let us create a mock dataframe for five fictitious individuals representing
# different ages, and departments at a research facility.

Name <- c('Jack', 'Kathy', 'Latesha', 'Brandon', 'Alexa', 'Jonathan', 'Joshua', 
          'Emily', 'Matthew', 'Anthony', 'Margaret', 'Natalie')
Age <- c(47, 41, 23, 55, 36, 54, 48, 23, 22, 27, 37, 43)
Experience <- c(7,5,9,3,11,6,8,9,5,2,1,4)
Position <- c('Economist', 'Director of Operations', 'Human Resources', 
              'Admin. Assistant', 'Data Scientist', 'Admin. Assistant', 
              'Account Manager', 'Account Manager', 'Attorney', 'Paralegal',
              'Data Analyst', 'Research Assistant')

df <- data.frame(Name, Age, Experience, Position)
df

# Let us examine the structure of the dataframe.
str(df)

# Let us examine the dimensions of the dataframe (number of rows and columns, respectively).
dim(df)

## Sorting Data ##
# Let us say that now we want to sort this dataframe in order of age (youngest to oldest).
df_age <- df[order(Age),]
df_age

# Now, if we want to sort experience by descending order while keeping age sorted 
# according to previous specifications, we can do the following:
df_age_exp <- df[order(Age, Experience),]
df_age_exp


## Handling #NA values ##
## #NA (not available) refers to missing values. What if our dataset has missing values? 
## How should we handle this scenario? For example, age has some missing values.
Name_2 <- c('Jack', 'Kathy', 'Latesha', 'Brandon', 'Alexa', 'Jonathan', 'Joshua', 
            'Emily', 'Matthew', 'Anthony', 'Margaret', 'Natalie')
Age_2 <- c(47, NA, 23, 55, 36, 54, 48, NA, 22, 27, 37, 43)
Experience_2 <- c(7,5,9,3,11,6,8,9,5,2,1,4)
Position_2 <- c('Economist', 'Director of Operations', 'Human Resources', 
                'Admin. Assistant', 'Data Scientist', 'Admin. Assistant', 
                'Account Manager', 'Account Manager', 'Attorney', 'Paralegal',
                'Data Analyst', 'Research Assistant')

df_2 <- data.frame(Name_2, Age_2, Experience_2, Position_2)
df_2

## Inspecting #NA values ##
is.na(df_2) # returns a Boolean matrix (True or False)
sum(is.na(df_2)) # sums up all of the NA values in the dataframe 
df_2[!complete.cases(df_2),] # we can provide a list of rows with missing data

# We can delete the rows with missing values by making an `na.omit()` function call 
# in the following manner:
df_2_na_omit <- na.omit(df_2)
df_2_na_omit

# Or we can use `complete.cases()` to subset only those rows that do not have missing values:
df_2[complete.cases(df_2), ]


## What if we receive a dataframe that, at a cursory glance, warehouses numerical values where 
# we see numbers, but when running additional operations on the dataframe, we discover that we 
# cannot conduct numerical exercises with columns that appear to have numbers. This is exactly 
# why it is of utmost importance for us to always inspect the structure of the dataframe using 
# the `str()` function call. Here is an example of the same dataframe with altered data types.

Name_3 <- c('Jack', 'Kathy', 'Latesha', 'Brandon', 'Alexa', 'Jonathan', 'Joshua', 
            'Emily', 'Matthew', 'Anthony', 'Margaret', 'Natalie')
Age_3 <- c('47', '41', '23', '55', '36', '54', '48', '23', '22', '27', '37', '43')
Experience_3 <- c(7,5,9,3,11,6,8,9,5,2,1,4)
Position_3 <- c('Economist', 'Director of Operations', 'Human Resources', 
                'Admin. Assistant', 'Data Scientist', 'Admin. Assistant', 
                'Account Manager', 'Account Manager', 'Attorney', 'Paralegal',
                'Data Analyst', 'Research Assistant')

df_3 <- data.frame(Name_3, Age_3, Experience_3, Position_3)

# Notice how Age is now expressed as a character data type, whereas Experience still shows as 
# a numeric datatype. 
str(df_3)

# Let us convert Age back to numeric using the `as.numeric()` function, and re-examine the dataframe.
df_3$Age_3 <- as.numeric(df_3$Age_3)
str(df_3)

# We can also convert experience from numeric to character/categorical data as follows:
df_3$Experience_3 <- as.character(df_3$Experience_3)
str(df_3)


##########################
#### Basic Statistics ####
##########################


## Setting the seed
# First, let us discuss the importance of setting a seed. Setting a seed to a specific yet arbitrary 
# value in R ensures the reproducibility of results.
# It is always best practice to use the same assigned seed throughout the entire experiment.
# Setting the seed to this arbitrary number (of any length) will guarantee exactly the same 
# output across all R sessions and users, respectively. 

# Let us create a new data frame of numbers 1 - 100.
mystats <- c(1:100)

# and go over the basic statistical functions
mean(mystats) # mean of the vector
median(mystats) # median of the vector
min(mystats) # minimum of the vector
max(mystats) # maximum of the vector
range(mystats)  # range of the vector
sum(mystats) # sum of the vector
sd(mystats) # standard deviation of the vector
class(mystats) # return data class of the vector
length(mystats) # the length of the vector
summary(mystats) # summary of the dataset

# To use an installed library, we must open that library with the following `library()` function call
library(psych)
# For example, the psych library uses the `describe()` function call to give us an alternate perspective int the 
# summary statistics of the data.
describe(mystats)

library(summarytools)
dfSummary(mystats)

# Simulating a normal distribution
# Now, We will use the `rnorm()` function to simulate a vector of 100 random normally 
# distributed data with a mean of 50, and a standard deviation of 10.
set.seed(222)
norm_vals <- rnorm(n = 100, mean = 50, sd = 10)
norm_vals


###############
#### Plots ####
###############

# Let's make a Simple stem-and-leaf plot. Here, we call the `stem()` function as follows:
stem(norm_vals)

# We can plot a histogram of these `norm_vals` in order to inspect their distribution 
# from a purely graphical standpoint.
# R uses the built-in `hist()` function to accomplish this task. Let us now plot the histogram.
hist(norm_vals)

# Our title, x-axis, and y-axis labels are given to us by default. 
# However, let us say that we want to change all of them to our desired specifications.
# To this end, we can parse in and  control the following parameters:
hist(norm_vals,
     col = 'lightblue', # specify the color
     xlab = 'Values', # specify the x-axis label 
     ylab = 'Frequency', # specify the y-axis label
     main = 'Histogram of Simulated Data', # specify the new title
     )

## Boxplots ##
# Similarly, we can make a boxplot in base R using the `boxplot()` function call as follows:
boxplot(norm_vals,
        col = 'lightblue', # specify the color
        xlab = '', # specify the x-axis label 
        ylab = 'Values', # specify the y-axis label
        main = 'Boxplot of Simulated Data' # specify the new title
        )


# Now, let us pivot the boxplot by parsing in the `horizontal = TRUE` parameter:
boxplot(norm_vals, horizontal = TRUE,
        col = 'lightblue', # specify the color
        xlab = 'Values', # specify the x-axis label 
        ylab = '', # specify the y-axis label
        main = 'Boxplot of Simulated Data' # specify the new title
        )

## Scatter Plots ##
# To make a simple scatter plot, we will simply call the `plot()` function on the same dataframe as follows:

plot(norm_vals,
     main = 'Scatter Plot of Simulated Data',
     pch = 20, # plot character - in this case default (circle)
     xlab = 'Index', 
     ylab = 'Value')

## Quantile-Quantile Plot ##
# Let us create a vector for the next example data and generate a normal quantile plot
quant_ex <- c(48.30, 49.03, 50.24, 51.39, 48.74, 51.55, 51.17, 49.87, 50.16, 
              49.80, 46.83, 48.48, 52.24,
    0.01, 49.50, 49.97, 48.56, 50.87)
qqnorm(quant_ex)
qqline(quant_ex) # adding a theoretical Q-Q line


#############################################
#### Skewness and Box-Cox Transformation ####
#############################################

# From statistics, let us recall that if the mean is greater than the median, the distribution will be 
# positively skewed. Conversely, if the median is greater than the mean, or the mean is less than the 
# median, the distribution will be negatively skewed.

# Let us examine the exact skewness of our distribution.

# Test skewness by looking at mean and median relationship
mean_norm_vals <- round(mean(norm_vals),0)
median_norm_vals <- round(median(norm_vals),0)
distribution <- data.frame(mean_norm_vals, median_norm_vals)
distribution

library(e1071)
skewness(norm_vals) # apply `skewness()` function from the e1071 library

# Applying Box-Cox Transformation on skewed variable
library(caret)
trans <- preProcess(data.frame(norm_vals), method=c("BoxCox"))
trans

########################
#### Basic Modeling #### 
########################


###########################
#### Linear Regression ####
###########################

# Let us set up an example dataset for the following modeling endeavors.

# independent variables (x's):
# X1
Hydrogen <- c(.18,.20,.21,.21,.21,.22,.23,.23,.24,.24,.25,.28,.30,.37,.31,.90,
              .81,.41,.74,.42,.37,.49,.07,.94,.47,.35,.83,.61,.30,.61,.54)
# X2
Oxygen <- c(.55,.77,.40,.45,.62,.78,.24,.47,.15,.70,.99,.62,.55,.88,.49,.36,.55,
            .42,.39,.74,.50,.17,.18,.94,.97,.29,.85,.17,.33,.29,.85)
# X3
Nitrogen <- c(.35,.48,.31,.75,.32,.56,.06,.46,.79,.88,.66,.04,.44,.61,.15,.48,
              .23,.90,.26,.41,.76,.30,.56,.73,.10,.01,.05,.34,.27,.42,.83) 

# there is always one dependent variable (target, response, y)
Gas_Porosity <- c(.46,.70,.41,.45,.55,.44,.24,.47,.22,.80,.88,.70,.72,.75,.16,
                  .15,.08,.47,.59,.21,.37,.96,.06,.17,.10,.92,.80,.06,.52,.01,.3) 

## Simple Linear Regression ##

# Prior to partaking in linear regression, it is best practice to examine correlation
# from a strictly visual perspective visa vie scatterplot as follows:
plot(Hydrogen, Gas_Porosity, 
     main = "Scatter Plot - Gas Porosity vs. Hydrogen Content", 
     xlab = "Hydrogen Content", ylab = "Gas Porosity",
     pch=16, col="blue")

# Find correlation coefficient, r
r1 <- cor(Hydrogen, Gas_Porosity)
r1

# Now we can set-up the linear model between one independent variable and one 
# dependent variable.
simple_linear_mod <- data.frame(Hydrogen, Gas_Porosity)

# By the correlation coefficient r you will see that there exists a relatively 
# moderate (positive) relationship. Let us now build a simple linear model from this dataframe.
lm_model1 <- lm(Gas_Porosity ~ Hydrogen, data = simple_linear_mod)
summary(lm_model1)

# Notice how the p-value for hydrogen content is 0.189, which lacks statistical significance
# when compared to the alpha value of 0.05 (at the 95% confidence level). Moreover, the R-Squared value
# of .05877 suggests that roughly 6% of the variance for gas propensity is explained by hydrogen content.

# we can make the same scatter plot, but this time with a best fit line
plot(Hydrogen, 
     Gas_Porosity, main = "Scatter Plot - Gas Porosity vs. Hydrogen Content", 
     xlab = "Hydogen Content", ylab = "Gas Porosity",
     pch=16, col="blue", abline(lm_model1, col="red")) 

## Multiple Linear Regression ##

# To account for all independent (x) variables in the model, let us set up the model in a dataframe:
multiple_linear_mod <- data.frame(Hydrogen, Oxygen, Nitrogen, Gas_Porosity)

# we can make additional scatter plots:

# gas porosity vs. oxygen
plot(Oxygen, 
     Gas_Porosity,
     main = "Scatter Plot - Gas Porosity vs. Oxygen Content", 
     xlab = "Oxygen Content", 
     ylab = "Gas Porosity",
     pch=16, 
     col="blue")


# gas porosity vs. nitrogen content
plot(Nitrogen, 
     Gas_Porosity, 
     main = "Scatter Plot - Gas Porosity vs. Oxygen Content", 
     xlab = "Oxygen Content", 
     ylab = "Gas Porosity",
     pch=16, 
     col="blue")

# And lastly we can build a multiple linear model from these 3 independent variables:
lm_model2 <- lm(Gas_Porosity ~ Hydrogen + Oxygen + Nitrogen,
                data = multiple_linear_mod)
summary(lm_model2)


#############################
#### Logistic Regression ####
#############################

# Whereas in linear regression, it is necessary to have a quantitative and continuous target variable,
# logistic regression is part of the generalized linear model series that has a categorical (often binary)
# target (outcome) variable. For example,

# let us say we want to predict grades for mathematics courses taught at a university. 
# So, we have the following example dataset:

# grades for calculus 1
calculus1 <- c(56,80,10,8,20,90,38,42,57,58,90,2,34,84,19,74,13,67,84,31,82,67,
               99,76,96,59,37,24,3,57,62)

# grades for calculus 2
calculus2 <- c(83,98,50,16,70,31,90,48,67,78,55,75,20,80,74,86,12,100,63,36,91,
               19,69,58,85,77,5,31,57,72,89)

# grades for linear algebra
linear_alg <- c(87,90,85,57,30,78,75,69,83,85,90,85,99,97, 38,95,10,99,62,47,17,
                31,77,92,13,44,3,83,21,38,70) 

# students passing/fail
pass_fail <- c('P','F','P','F','P','P','P','P','F','P','P','P','P','P','P','F',
               'P','P','P','F','F','F','P','P','P','P','P','P','P','P','P')

# At this juncture, we cannot build a model with categorical values until and unless they are binarized
# using the `ifelse()` function call as follows. A passing score will be designated by a 1, and failing 
# score with a 0, respectively.

math_outcome <- ifelse(pass_fail=='P', 1, 0)
math_outcome

logistic_model <- data.frame(calculus1, calculus2, linear_alg, 
                             pass_fail, math_outcome)
str(logistic_model) # examine data structures of the model

# We can also specify `glm` instead of just `lm` as in linear regression example:
lm_model3 <- glm(math_outcome ~ calculus1 + calculus2 + linear_alg, 
                 family = binomial(), data = logistic_model)
summary(lm_model3)


########################
#### Decision Trees ####
########################

# Similarly, we can plot the trajectory of the outcome using  the `rpart()` function
# of the `library(rpart)` and `library(rpart.plot)`, respectively.
library(rpart)
library(rpart.plot)

# In favor of a larger dataset to illustrate the structure, function, and overall efficacy
# of decision trees in R, we will rely on the built-in `mtcars` dataset.
?mtcars# more information about this dataset

str(mtcars)
mtcars

# So we introduce the model as follows:
tree_model <- rpart(mpg ~., data=mtcars) # the decision tree model
rpart.plot(tree_model, main = 'Cars: Classification Tree') # plot the decision tree

# Passing in a `type=1,2,3,4, or 5` value changes the appearance of the tree
rpart.plot(tree_model, main = 'Cars: Classification Tree', type=2)


##################################################
#####           Basic Modeling with           #### 
#####          Cross-Validation in R          ####
##################################################

# We use cross-validation as a "a statistical approach for determining how well 
# the results of a statistical investigation generalize to a different data set" (finnstats, 2021).
# The `library(caret)` will help us in this endeavor.

# partition into 75:25 train_test_split

set.seed(222) # for reproducibility
dt <- sort(sample(nrow(mtcars), nrow(mtcars)*.75))
train_cars <-mtcars[dt,]
test_cars <-mtcars[-dt,]

# check size dimensions of respective partions
n_train <- nrow(train_cars)[1]
n_test <- nrow(test_cars)[1]

train_size = n_train/(n_train+n_test)
test_size = n_test/(n_train+n_test)

cat('\n Train Size:', train_size,
    '\n Test Size:', test_size)


# Let us bring in a generalized linear model for this illustration.
cars_model <- glm(mpg ~., data = mtcars)
cars_predictions <- predict(cars_model, test_cars)

# computing model performance metrics
data.frame(R2 = R2(cars_predictions, test_cars$mpg), 
           RMSE = RMSE(cars_predictions, test_cars$mpg),
           MAE = MAE(cars_predictions, test_cars$mpg))

# In order to use the `trainControl()` function for cross-validation, we will
# bring in the `library(caret)`.
library(caret)

# Best model has lowest error(s)
# Now let us train the model with cross-validation
train_control <- trainControl(method = "cv", number = 5, savePredictions=TRUE)
cars_predictions <- train(mpg ~., data=mtcars, # glm model
                          method = 'glm',
                          trControl = train_control) # cross-validation
cars_predictions


############################
#### K-Means Clustering ####
############################


# A cluster is a collection of observations. We want to group these observations based on
# the most similar attributes.
# We use distance measures to measure similarity between clusters. 
# This is one of the most widely-used unsupervised learning techniques that groups "similar 
# data points together and discover underlying patterns. To achieve this objective, K-means 
# looks for a fixed number (k) of clusters in a dataset" (Garbade, 2018).

# Let us split the mtcars dataset into 3 clusters.
library(cluster)    # clustering algorithms
library(factoextra) # clustering algorithms & visualization
set.seed(222)
kmeans_cars <- kmeans(mtcars,      # dataset 
                      centers = 3, # number of centroids
                      nstart = 20) # number of random starts s/b > 1
kmeans_cars 

# Now let's visualize the cluster using the `fviz_cluster()` function from the `factoextra` library.

fviz_cluster(kmeans_cars, data = mtcars) + theme_classic() # from factoextra lib.


# But what is the appropriate number of clusters that we should generate? 
# Can we do better with more clusters?

# total sum of squares
kmeans_cars$totss 

# between sum of squares
kmeans_cars$betweenss

# within sum of squares
kmeans_cars$withinss 

# ratio for between sum of squares/ total sum of squares
kmeans_cars$betweenss/kmeans_cars$totss 

# Let's create a numeric vector populated with zeroes and ten spots long.
wss <- numeric(10)


# Can we do better? Let's run k-means from 1:10 clusters.
# This will effectively measure the homogeneity of the clusters as the number 
# of clusters increases.

# Now let us use a basic for-loop to run through k-means 10 times.
# K-means is iterated through each of these 10 clusters as follows:

for(i in 1:10) {
  wss[i] <- sum(kmeans(mtcars, 
                       centers=i)$withinss)
}

## Basic Elbow Method ##
# Now let's plot these within sum of squares using the elbow method, which is one of 
# the most commonly used approaches for finding the optimal k.
plot(wss, 
     type='b', 
     main ='Elbow Method for K-Means',
     col='blue') 

# Once we start to add clusters, the within sum of squares is reduced. Thus,
# the incremental reduction in within sum of squares is getting progressively smaller.
# We see that after approximately k = 3, each of the new clusters is not separating the
# data as well.


#################################
#### Hierarchical Clustering ####
#################################

# This is another form of unsupervised learning type of cluster analysis, which takes
# on a more visual method, working particularly well with smaller samples (i.e., n < 500), 
# such as this mtcars dataset. We start out with as many clusters as observations, and we go
# through a procedure of combining observations into clusters, and culminating with combining
# clusters together as a reduction method for the total number of clusters that are present.
# Moreover, the premise for combining clusters together is a direct result of:

## complete linkage - or largest Euclidean distance between clusters.
## single linkage - conversely, we look at the observations which are closest together (proximity).
## centroid linkage - we can the distance between the centroid of each cluster.
## group average (mean) linkage - taking the mean between the pairwise distances of the observations.

## Complete linkage is the most traditional approach. ##

# The tree structure that examines this hierarchical structure is called a dendogram.

auto_dist <- dist(mtcars, method ='euclidean', diag = FALSE)
auto_cluster <- hclust(auto_dist, method ='complete')

# plot the hierarchical cluster
plot(auto_cluster)
rect.hclust(auto_cluster, k = 3, border = 'red') # visualize cluster borders

# Our dendogram indicates which observation is within which cluster.

# We can cut our tree at let's say 3 clusters, segmenting them out as follows:
cut_tree <- cutree(auto_cluster, 3) # each obs. now belongs to cluster 1,2, or 3
mtcars$segment <- cut_tree # segment out the data

# Now we can view our segmented data in the workspace window as follows:
View(mtcars)




##################
#### Sources: ####
##################

# finnstats. (2021, October 31). What Does Cross Validation Mean? R-bloggers. 
#     https://www.r-bloggers.com/2021/10/cross-validation-in-r-with-example/
#
# Garbade, Michael. (2018, September 12). Understanding K-means Clustering in Machine Learning. Towards Data Science. 
#     https://towardsdatascience.com/understanding-k-means-clustering-in-machine-learning-6a6e67336aa1
#
# GeeksforGeeks. (2020, April 22). Scope of Variable in R. GeeksforGeeks. 
#     https://www.geeksforgeeks.org/scope-of-variable-in-r/
#
# Shmueli, G., Bruce, P. C., Yahav, I., Patel, N. R., & Lichtendahl Jr., K. C. (2018). 
#     Data mining for business analytics: Concepts, techniques, and applications in R. Wiley.

