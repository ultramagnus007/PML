# Practical Machine Learning Assignment
Author: UltraMagnus  
# Overview

# Loading the Data  

Reading the Data from csv files 'Tdata' contain the traning data and Sdata contains the 20 test case to be submitted.  


```r
dirpath="/home/lotus/Data_Science/08_Practical Machine Learning/GIT/"
Tdata<-read.table(paste0(dirpath,"pml-training.csv"), header = TRUE, sep = ",")
Sdata<-read.table(paste0(dirpath,"pml-testing.csv"), header = TRUE, sep = ",")
```

# Cleaning the Data  

1. Calculating the index of columns which contains the NA values, rmCol contains the 
index of those columns.

```r
library(dplyr)
```

```
## 
## Attaching package: 'dplyr'
## 
## The following objects are masked from 'package:stats':
## 
##     filter, lag
## 
## The following objects are masked from 'package:base':
## 
##     intersect, setdiff, setequal, union
```

```r
i<-0
rmCol<-NULL
for(i in 1:ncol(Sdata))
{
  if(sum(is.na(Sdata[, i]))  > 0)
		rmCol<-c(rmCol,i)
}
```

removing column containing NA values from both 


```r
Tdata<-select(Tdata, -rmCol)
Sdata<-select(Sdata, -rmCol)
```

removing unuseful colums  

```r
Tdata<-select(Tdata, -c(X,user_name, cvtd_timestamp, new_window, num_window))
Sdata<-select(Sdata, -c(X,user_name, cvtd_timestamp, new_window, num_window))
dim(Tdata)
```

```
## [1] 19622    55
```

```r
dim(Sdata)
```

```
## [1] 20 55
```

so finally both contains 55 columns  

# Creating Traning and Test Set
creating the Traning and Test set 70% for Training and 30% for Testing. 

```r
library(caret)
```

```
## Loading required package: lattice
## Loading required package: ggplot2
```

```r
set.seed(12321)
testIndex = createDataPartition(Tdata$classe, p = 0.70,list=FALSE)
training = Tdata[-testIndex,]
testing = Tdata[testIndex,]
```

# Training  

Training the training data using randomForest.  


```r
library(randomForest)
```

```
## randomForest 4.6-10
## Type rfNews() to see new features/changes/bug fixes.
## 
## Attaching package: 'randomForest'
## 
## The following object is masked from 'package:dplyr':
## 
##     combine
```

```r
modFit <- randomForest(classe ~ .,data=training)
```


#Testing 
Testing the model, 55th column is not used as it is 'problem_id' variable  

```r
confusionMatrix(testing$classe, predict(modFit,testing[,-55]))
```

```
## Confusion Matrix and Statistics
## 
##           Reference
## Prediction    A    B    C    D    E
##          A 3904    2    0    0    0
##          B   22 2636    0    0    0
##          C    0   32 2362    2    0
##          D    0    0   37 2211    4
##          E    0    0    3   14 2508
## 
## Overall Statistics
##                                          
##                Accuracy : 0.9916         
##                  95% CI : (0.9899, 0.993)
##     No Information Rate : 0.2858         
##     P-Value [Acc > NIR] : < 2.2e-16      
##                                          
##                   Kappa : 0.9893         
##  Mcnemar's Test P-Value : NA             
## 
## Statistics by Class:
## 
##                      Class: A Class: B Class: C Class: D Class: E
## Sensitivity            0.9944   0.9873   0.9833   0.9928   0.9984
## Specificity            0.9998   0.9980   0.9970   0.9964   0.9985
## Pos Pred Value         0.9995   0.9917   0.9858   0.9818   0.9933
## Neg Pred Value         0.9978   0.9969   0.9965   0.9986   0.9996
## Prevalence             0.2858   0.1944   0.1749   0.1621   0.1829
## Detection Rate         0.2842   0.1919   0.1719   0.1610   0.1826
## Detection Prevalence   0.2843   0.1935   0.1744   0.1639   0.1838
## Balanced Accuracy      0.9971   0.9926   0.9902   0.9946   0.9984
```

we can see accurary is 99.16% which is quite statisfactory.  so we can use above model for final test cases for submission.  

On 20 Test Cases Results are  

```r
result<-predict(modFit,Sdata[,-55])
result
```

```
##  1  2  3  4  5  6  7  8  9 10 11 12 13 14 15 16 17 18 19 20 
##  B  A  B  A  A  E  D  B  A  A  B  C  B  A  E  E  A  B  B  B 
## Levels: A B C D E
```

all test cases passed.
