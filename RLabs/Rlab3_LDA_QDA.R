### Machine Learning
### LDA e QDA
### Prof. Neylson Crepalde

# Libraries
library(MASS)
library(readr)
library(ISLR)
library(pROC)

data("Default")
Default = as_tibble(Default)
Default

# Making train and test sets
train = sample(1:nrow(Default), nrow(Default)*.8)


#### LDA
ldamodel = lda(default ~ balance + income + student, data = Default, subset = train)
ldamodel

yhat = predict(ldamodel, Default[-train, ], type = "response")$class

# Confusion matrix
table(Default$default[-train], yhat)
roc(response = Default$default[-train], predictor = as.numeric(as.factor(yhat)), plot = T,
    main = "ROC curve for LDA")


## QDA
qdamodel = qda(default ~ balance + income + student, data = Default, subset = train)
qdamodel

yhat = predict(qdamodel, Default[-train, ], type = "response")$class

# Confusion Matrix
table(Default$default[-train], yhat)
roc(response = Default$default[-train], predictor = as.numeric(as.factor(yhat)), plot = T,
    main = "ROC curve for QDA")
