###--------------clearing environment and setting up directory-------------------###

rm(list = ls())
setwd("K:/Data_Science/Project/Project_02")
getwd()

###------------------- importing libraries --------------------------------------###

libs = c("ggplot2", "geosphere", "corrgram", "DMwR", "caret", "rpart", "randomForest", "xgboost","stats", "sp", "pROC", "e1071")

#load Packages
lapply(libs, require, character.only = TRUE)
rm(libs)


###-------------------- loading data --------------------------------------------###

dfLoaner <- read.csv("01_Data/bank-loan.csv", header=TRUE, na.strings = c(" ", "", "NA"))

###----------------------overview of data----------------------------------------###

str(dfLoaner)
#'data.frame':	850 obs. of  9 variables:
#$ age     : int  41 27 40 41 24 41 39 43 24 36 ...
#$ ed      : int  3 1 1 1 2 2 1 1 1 1 ...
#$ employ  : int  17 10 15 15 2 5 20 12 3 0 ...
#$ address : int  12 6 14 14 0 5 9 11 4 13 ...
#$ income  : int  176 31 55 120 28 25 67 38 19 25 ...
#$ debtinc : num  9.3 17.3 5.5 2.9 17.3 10.2 30.6 3.6 24.4 19.7 ...
#$ creddebt: num  11.359 1.362 0.856 2.659 1.787 ...
#$ othdebt : num  5.009 4.001 2.169 0.821 3.057 ...
#$ default : int  1 0 0 0 1 0 0 0 1 0 ...


summary(dfLoaner)

dfLoaner$default <-as.factor(dfLoaner$default)

###------------------------distribution of individual variables------------------###
ggplot(dfLoaner, aes(x = age)) + geom_area( stat = 'count')
ggplot(dfLoaner, aes(x = ed)) + geom_area( stat = 'count')
ggplot(dfLoaner, aes(x = employ)) + geom_area( stat = 'count')
ggplot(dfLoaner, aes(x = address)) + geom_area( stat = 'count')
ggplot(dfLoaner, aes(x = income)) + geom_area( stat = 'count')
ggplot(dfLoaner, aes(x = debtinc)) + geom_area( stat = 'count')
ggplot(dfLoaner, aes(x = creddebt)) + geom_area( stat = 'count')
ggplot(dfLoaner, aes(x = othdebt)) + geom_area( stat = 'count')
ggplot(dfLoaner, aes(x = default)) + geom_area( stat = 'count')

###------------------------feature tranformation---------------------------------###

#Normalizing Values

preproc_x = preProcess(dfLoaner[,c(1:8)], method = 'range')
dfLoaner[,c(1:8)] <- predict(preproc_x, dfLoaner[,c(1:8)])


###----------------------------------feature selection---------------------------###
#correlation analysis
corrgram(dfLoaner, order = F,
         upper.panel=panel.pie, text.panel=panel.txt, main = "Correlation Plot")


#vif analysis for multicolinearity
vif(dfLoaner)
#Variables      VIF
#1       age 2.022854
#2        ed 1.263713
#3    employ 2.401604
#4   address 1.592315
#5    income 4.215396
#6   debtinc 3.370526
#7  creddebt 2.615818
#8   othdebt 3.861054
#9   default 1.405265
# no vif more than 5

###----------------------------model building------------------------------------###

#train-test split
dfTrain <- dfLoaner[which(!is.na(dfLoaner$default)),]
dfTest <- dfLoaner[which(is.na(dfLoaner$default)),]
set.seed(1000)
tr.idx = createDataPartition(dfTrain$default,p=0.7,list = FALSE)
to_train <- dfTrain[tr.idx, ]
to_test <- dfTrain[-tr.idx, ]
formula_y <- (default ~ age + ed + employ + address + income + debtinc + creddebt + othdebt) 

#Error Metric defination
err_metric=function(CM)
{
  TN =CM[1,1]
  TP =CM[2,2]
  FP =CM[1,2]
  FN =CM[2,1]
  precision =(TP)/(TP+FP)
  recall_score =(FP)/(FP+TN)
  f1_score=2*((precision*recall_score)/(precision+recall_score))
  
  print(paste("Precision value of the model: ",round(precision,2)))
  print(paste("Recall value of the model: ",round(recall_score,2)))
  print(paste("f1 score of the model: ",round(f1_score,2)))
}

#logistic regression
logModel <- glm(formula = formula_y, data=to_train, family=binomial)
summary(logModel)
logPred = predict(logModel,to_test[,-9], type="response")

cnf_mtr <- table(to_test[,9],logPred>0.5)
print(cnf_mtr)
#  FALSE  TRUE
#0   144    11
#1    28    26 

err_metric(cnf_mtr)
#[1] "Precision value of the model:  0.7"
#[1] "Recall value of the model:  0.07"
#[1] "f1 score of the model:  0.13"

roc_score <- roc(to_test[,9], logPred)
plot(roc_score, main="ROC Curve")

#decision tree
dtreeModel <- rpart(formula = formula_y, data=to_train, method='class', control=rpart.control(cp=0.01))
summary(dtreeModel)
dtreePred = predict(dtreeModel, to_test[,-9])

cnf_mtr <- table(to_test[,9],dtreePred[,2]>0.5)
print(cnf_mtr)
#  FALSE  TRUE
#0   142    13
#1    30    24 

err_metric(cnf_mtr)
#[1] "Precision value of the model:  0.65"
#[1] "Recall value of the model:  0.08"
#[1] "f1 score of the model:  0.15"

roc_score <- roc(to_test[,9], dtreePred[,2])
plot(roc_score, main="ROC Curve")

#support vector
svmModel <- svm(formula = formula_y, data=to_train, probability=TRUE)
summary(svmModel)
svmPred = predict(svmModel, to_test[,-9], probability=TRUE)

cnf_mtr <- table(to_test[,9],svmPred)
print(cnf_mtr)
#  FALSE  TRUE
#0   144    11
#1    32    22 

err_metric(cnf_mtr)
#[1] "Precision value of the model:  0.67"
#[1] "Recall value of the model:  0.07"
#[1] "f1 score of the model:  0.13"

roc_score <- roc(to_test[,9], attr(svmPred, "probabilities")[,2])
plot(roc_score, main="ROC Curve")

#naive bayes
nbModel <- naiveBayes(formula = formula_y, data=to_train, probability=TRUE)
summary(nbModel)
nbPred = predict(nbModel, to_test[,-9], type="raw")

cnf_mtr <- table(to_test[,9],nbPred[,2]>0.5)
print(cnf_mtr)
#  FALSE  TRUE
#0   147     8
#1    41    13 

err_metric(cnf_mtr)
#[1] "Precision value of the model:  0.62"
#[1] "Recall value of the model:  0.05"
#[1] "f1 score of the model:  0.1"

roc_score <- roc(to_test[,9], nbPred[,2])
plot(roc_score, main="ROC Curve")

#random forest
rfrstModel <- randomForest(formula = formula_y, data=to_train, importance=TRUE, ntree=500, nodesize=4)
summary(rfrstModel)
rfrstPred = predict(rfrstModel, to_test[,-9], type="prob")

cnf_mtr <- table(to_test[,9],rfrstPred[,2]>0.5)
print(cnf_mtr)
#  FALSE  TRUE
#0   143    12
#1    26    18 

err_metric(cnf_mtr)
#[1] "Precision value of the model:  0.7"
#[1] "Recall value of the model:  0.08"
#[1] "f1 score of the model:  0.14"

roc_score <- roc(to_test[,9], rfrstPred[,2])
plot(roc_score, main="ROC Curve")

#improving accuracy through XGBoost regression
to_train_matrix <- as.matrix(sapply(to_train[-9],as.numeric))
to_test_matrix <- as.matrix(sapply(to_test[-9],as.numeric))


XGBModel <- xgboost(data = to_train_matrix, label = as.matrix(to_train[9]), objective="binary:logistic", nrounds = 10, colsample_bytree=0.3)
summary(XGBModel)
xgbPred <- predict(XGBModel, to_test_matrix)

cnf_mtr <- table(to_test[,9],xgbPred>0.5)
print(cnf_mtr)
#  FALSE  TRUE
#0   137    18
#1    27    27 

err_metric(cnf_mtr)
#[1] "Precision value of the model:  0.6"
#[1] "Recall value of the model:  0.12"
#[1] "f1 score of the model:  0.19"

roc_score <- roc(to_test[,9], xgbPred)
plot(roc_score, main="ROC Curve")

###------------------------------final model-------------------------------------###
xgb_train <- as.matrix(sapply(dfTrain[-9],as.numeric))
xgb_test <- as.matrix(sapply(dfTest[-9],as.numeric))

finalModel <- xgboost(data = xgb_train, label = as.matrix(dfTrain[9]), objective="binary:logistic", nrounds = 10, colsample_bytree=0.3)

finalPred <- predict(finalModel, xgb_test)

final_Pred = data.frame("fare_amount" = finalPred>0.5)

#saving predictions in csv file
write.csv(final_Pred,"loandflt_xgb_output_r.csv",row.names = TRUE)

#saving model in dump format
saveRDS(finalModel, "./loandflt_xgbmodel_r")

####################################################################################
