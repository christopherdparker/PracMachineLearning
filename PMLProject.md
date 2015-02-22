# Practical Machine Learning: Predicting Activity Quality from Activity Monitors

## Executive summary
This documents the creation of a model for predicting activity quality from activity monitor data. The model is highly accurate according to cross-validation estimates of the error.

## Loading and preprocessing the data
I begin by loading the necessary packages, setting up parallel processing, and importing the data.


```r
require(dplyr)
```

```
## Loading required package: dplyr
## 
## Attaching package: 'dplyr'
## 
## The following object is masked from 'package:stats':
## 
##     filter
## 
## The following objects are masked from 'package:base':
## 
##     intersect, setdiff, setequal, union
```

```r
require(ggplot2)
```

```
## Loading required package: ggplot2
```

```r
require(doMC)
```

```
## Loading required package: doMC
## Loading required package: foreach
## Loading required package: iterators
## Loading required package: parallel
```

```r
require(caret)
```

```
## Loading required package: caret
## Loading required package: lattice
```

```r
registerDoMC(cores = 4)
testing = read.csv("pml-testing.csv", na.strings = "#DIV/0!")
training = read.csv("pml-training.csv", na.strings = "#DIV/0!")
```

With the data imported there is still an issue with several variables coming in as factor variables rather than numeric. R-bloggers helps me solve this problem by providing a varlist function for picking factor variables from the data. For more information on this you can visit: http://www.r-bloggers.com/selecting-subset-of-variables-in-data-frame/

```r
varlist <- function (df=NULL,type=c("numeric","factor","character"), pattern="", exclude=NULL) {
    vars <- character(0)
    if (any(type %in% "numeric")) {
        vars <- c(vars,names(df)[sapply(df,is.numeric)])
    }
    if (any(type %in% "factor")) {
        vars <- c(vars,names(df)[sapply(df,is.factor)])
    }  
    if (any(type %in% "character")) {
        vars <- c(vars,names(df)[sapply(df,is.character)])
    }  
    vars[(!vars %in% exclude) & grepl(vars,pattern=pattern)]
}
```

With this function we can now pick out the weird factor variables and turn them into numeric variables before putting the data back into the main data frame.

```r
tempfix = select(training,one_of(varlist(training,type="factor", exclude = c("user_name", "cvtd_timestamp",
                                                                             "new_window", "classe"))))
tempfine = select(training,-one_of(varlist(training,type="factor", exclude = c("user_name", "cvtd_timestamp",
                                                                             "new_window", "classe"))))
tempfix = apply(tempfix,2,function(x) {as.numeric(x)})
training = cbind(tempfine,tempfix)
```

And the same fix for the holdout data (the data we need to predict but never observe the true outcome).

```r
tempfix = select(testing,one_of(varlist(testing,type="factor", exclude = c("user_name", "cvtd_timestamp",
                                                                             "new_window"))))
tempfine = select(testing,-one_of(varlist(testing,type="factor", exclude = c("user_name", "cvtd_timestamp",
                                                                               "new_window"))))
tempfix = apply(tempfix,2,function(x) {as.numeric(x)})
testing = cbind(tempfine,tempfix)
```

Some variables are all NA in holdout so it doesn't make sense to use them for prediction

```r
todrop = sapply(testing,function(x) {sum(is.na(x))==20})
training = training[,!todrop]
testing = testing[,!-todrop]
```

## Building a prediction model
Now the data is in a reasonable form so we can run a model on the training data. We do this with repeated crosss-validation so we can get a good estimate of the out of sample error. For allow for 10 k-folds and repeat this 5 times. I used these numbers to balance bias, accuracy, and time to run the code.

```r
mymod = train(classe ~ . -raw_timestamp_part_1 -raw_timestamp_part_2 -cvtd_timestamp -new_window -num_window,
              data = training, method = "rf", trControl = trainControl(method = "repeatedcv", number = 10, repeats = 5
                                                                       , classProbs = TRUE, savePredictions = TRUE))
```

```
## Loading required package: randomForest
## randomForest 4.6-10
## Type rfNews() to see new features/changes/bug fixes.
```

```r
mymod
```

```
## Random Forest 
## 
## 19622 samples
##    59 predictor
##     5 classes: 'A', 'B', 'C', 'D', 'E' 
## 
## No pre-processing
## Resampling: Cross-Validated (10 fold, repeated 5 times) 
## 
## Summary of sample sizes: 17660, 17661, 17658, 17659, 17659, 17660, ... 
## 
## Resampling results across tuning parameters:
## 
##   mtry  Accuracy   Kappa      Accuracy SD   Kappa SD    
##    2    0.9985015  0.9981046  0.0011714468  0.0014818001
##   30    0.9998573  0.9998195  0.0002920005  0.0003693439
##   58    0.9997757  0.9997163  0.0003282364  0.0004151732
## 
## Accuracy was used to select the optimal model using  the largest value.
## The final value used for the model was mtry = 30.
```

Taking all of the training models together we can obtain an estimate of the out of sample error:

```r
allerrors = summarize(group_by(mymod$pred,Resample), aveerror = mean(pred == obs))
meanerror = mean(allerrors$aveerror)
meanerror
```

```
## [1] 0.9993782
```

## Predicting the test set
We can now predict the test set. Since this is going into github I am not actually uploading the files but holding them in a separate working directory.

```r
test_prediction<-predict(mymod, newdata=testing)
pml_write_files = function(x){
  n = length(x)
  for(i in 1:n){
    filename = paste0("problem_id_",i,".txt")
    write.table(x[i],file=filename,quote=FALSE,row.names=FALSE,col.names=FALSE)
  }
}
pml_write_files(test_prediction)
```
