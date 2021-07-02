# This is the pure code for early stage diabetes prediction problem

# Load package
rm(list=ls())
library(tidyverse)
library(ggpubr)
library(glmnet)
library(class)
library(e1071)
library(randomForest)
library(pROC)
library(ggplot2)
library(neuralnet)

## Define performance metrics
ca <- function(table){
  return((table[1]+table[4])/sum(table))
}
precision <- function(table){
  return(table[4]/(table[4]+table[2]))
}
sensitivity <- function(table){
  return(table[4]/(table[4]+table[3]))
}
specificity <- function(table){
  return(table[1]/(table[1]+table[2]))
}
f1 <- function(precision, sensitivity){
  return(2*precision*sensitivity/(precision+sensitivity))
}

## Read data and perform exploratory analysis
diabetes <- read.csv('~/Documents/R/diabetes_data_upload.csv')
diabetes <- diabetes %>%
  mutate(Gender = as.factor(ifelse(Gender=='Male', 1, 0)),
         Polyuria = as.factor(ifelse(Polyuria=='Yes', 1, 0)),
         Polydipsia = as.factor(ifelse(Polydipsia=='Yes', 1, 0)),
         sudden.weight.loss = as.factor(ifelse(sudden.weight.loss=='Yes', 1, 0)),
         weakness = as.factor(ifelse(weakness=='Yes', 1, 0)),
         Polyphagia = as.factor(ifelse(Polyphagia=='Yes', 1, 0)),
         Genital.thrush = as.factor(ifelse(Genital.thrush=='Yes', 1, 0)),
         visual.blurring = as.factor(ifelse(visual.blurring=='Yes', 1, 0)),
         Itching = as.factor(ifelse(Itching=='Yes', 1, 0)),
         Irritability = as.factor(ifelse(Irritability=='Yes', 1, 0)),
         delayed.healing = as.factor(ifelse(delayed.healing=='Yes', 1, 0)),
         partial.paresis = as.factor(ifelse(partial.paresis=='Yes', 1, 0)),
         muscle.stiffness = as.factor(ifelse(muscle.stiffness=='Yes', 1, 0)),
         Alopecia = as.factor(ifelse(Alopecia=='Yes', 1, 0)),
         Obesity = as.factor(ifelse(Obesity=='Yes', 1, 0)), 
         class = as.factor(ifelse(class=='Positive', 1, 0)))
# Exploratory table of categorical variable
table <- summary(diabetes[, -1])
# Boxplot of continuous variable
ggplot(data = diabetes) +
  geom_boxplot(mapping = aes(x=class, y=Age)) +
  scale_x_discrete()
# Bar plot of the association between continuous variable and categorical variable
vb <- ggplot(data=diabetes)+
  geom_histogram(mapping=aes(x=Age, fill=visual.blurring), binwidth=3) +
  scale_fill_manual(values = c('orange', 'chocolate'),
                    guide = guide_legend(nrow = 1, label.position = 'bottom', keywidth = 2)) +
  theme(legend.position = 'top')
mu <- ggplot(data=diabetes)+
  geom_histogram(mapping=aes(x=Age, fill=muscle.stiffness), binwidth=3) +
  scale_fill_manual(values = c('orange', 'chocolate'),
                    guide = guide_legend(nrow = 1, label.position = 'bottom', keywidth = 2)) +
  theme(legend.position = 'top')
se <- ggplot(data=diabetes)+
  geom_histogram(mapping=aes(x=Age, fill=Gender), binwidth=3) +
  scale_fill_manual(values = c('orange', 'chocolate'),
                    guide = guide_legend(nrow = 1, label.position = 'bottom', keywidth = 2)) +
  theme(legend.position = 'top')
cla <- ggplot(data=diabetes)+
  geom_histogram(mapping=aes(x=Age, fill=class), binwidth=3) +
  scale_fill_manual(values = c('orange', 'chocolate'),
                    guide = guide_legend(nrow = 1, label.position = 'bottom', keywidth = 2)) +
  theme(legend.position = 'top')
ggarrange(vb, mu, se, cla, labels=c('A', 'B', 'C', 'D'), 
          ncol = 2, nrow = 2)

## Logistic regression
# Association between predictors and response
LRmodel <- glm(class~., family = 'binomial', data = diabetes)
LRcoef <- data.frame('Feature'=colnames(diabetes)[-17],
                     'Estimate'=round(as.numeric(LRmodel$coefficients[-1]), 2),
                     'lwr.95'=round(as.numeric(confint(LRmodel)[, 1][-1]), 2),
                     'upr.95'=round(as.numeric(confint(LRmodel)[, 2][-1]), 2),
                     'p'=round(as.numeric(coef(summary(LRmodel))[, 4][-1]), 2))
# 10-fold cross validation splitting for all the following models
set.seed(1111)
n <- dim(diabetes)[1]
p <- dim(diabetes)[2]
k <- 10
split.sample <- sample(1:k, n, replace = TRUE)
# Modeling using cross validation
LRall <- data.frame('Class'=NA, 'Prediction'=NA)
for(i in 1:k){
  TrainDiabetes <- diabetes[split.sample!=i, ]
  TestDiabetes <- diabetes[split.sample==i, ]
  LRtrain <- glm(class~., family = 'binomial', data = TrainDiabetes)
  LRprediction <- as.factor(ifelse(predict(LRtrain, TestDiabetes[, -17], type = 'response') >= 0.5, 
                                   'Positive', 'Negative'))
  LRpredictiondf <- data.frame('Class'=TestDiabetes[, 17],
                               'Prediction'=LRprediction)
  LRall <- rbind(LRall, LRpredictiondf)
}
LRallTable <- table(LRall)
rownames(LRallTable) <- c('Negative', 'Positive')

## Logistic regression with lasso penalty
# Tune parameters
set.seed(1234)
DiabetesX <- model.matrix(class~., diabetes)
cvfit <- cv.glmnet(DiabetesX, diabetes$class, 
                   family = 'binomial', type.measure = 'class', 
                   nfolds = 10, alpha = 1)
lambdaLasso <- cvfit$lambda.1se
plot(cvfit, main = '10-fold cross validation with lasso')
LassoAll <- data.frame('Class'=NA, 'Prediction'=NA)
# 10-fold cross-validation
for(i in 1:k){
  TrainDiabetes <- diabetes[split.sample!=i, ]
  TestDiabetes <- diabetes[split.sample==i, ]
  # train the model
  TrainDiabetesX <- model.matrix(class~., TrainDiabetes)
  LassoModel <- glmnet(TrainDiabetesX, TrainDiabetes$class,
                       family = 'binomial', type.measure = 'class',
                       lambda = lambdaLasso, alpha = 1)
  # LassoCoef <- as.data.frame(coef(LassoModel)[-c(1, 2),])
  # colnames(LassoCoef) <- 'Estimate'
  LassoPrediction <- as.factor(predict(LassoModel, model.matrix(class~., TestDiabetes), type = 'class'))
  LassoAll <- rbind(LassoAll, data.frame('Class'=TestDiabetes$class,
                                         'Prediction'=LassoPrediction))
}
LassoTable <- table(LassoAll)
rownames(LassoTable) <- c('Negative', 'Positive')
colnames(LassoTable) <- c('Negative', 'Positive')

## KNN
# Need to reload the data, otherwise knn would have error.
diabetesKnn <- read.csv('~/Documents/R/diabetes_data_upload.csv') %>%
  mutate(Gender = ifelse(Gender == 'Female', 0, 
                         ifelse(Gender == 'Male', 1, NA)),
         class = as.factor(ifelse(class == 'Positive', 1, 
                                  ifelse(class == 'Negative', 0, NA))))
column_names <- colnames(diabetesKnn)
# Tune parameters
for (name in column_names[3:16]){
  diabetesKnn[name] <- ifelse(diabetesKnn[name] == 'Yes', 1, 0)
}
cvknn <- tune.knn(diabetesKnn[, -17], diabetesKnn[, 17], 
                  k=1:20, tunecontrol=tune.control(sampling='boot'), cross=10)
KnnAll <- data.frame('Class'=NA, 'Prediction'=NA)
# Modeling
for(i in 1:k){
  TrainDiabetes <- diabetesKnn[split.sample!=i, ]
  TrainDiabetes <- diabetesKnn[split.sample!=i, ]
  TestDiabetes <- diabetesKnn[split.sample==i, ]
  Knnprediction <- knn(TrainDiabetes[, -17], TestDiabetes[, -17],
                       TrainDiabetes[, 17], k=cvknn$best.parameters$k)
  KnnAll <- rbind(KnnAll, data.frame('Class'=TestDiabetes$class,
                                     'Prediction'=Knnprediction))
}
KnnTable <- table(KnnAll)
rownames(KnnTable) <- c('Negative', 'Positive')
colnames(KnnTable) <- c('Negative', 'Positive')

## Random forest
# Tune parameters
set.seed(9999)
forest.para <- tuneRF(diabetes[, -17], diabetes[, 17],
                      ntreeTry = 100, stepFactor = 0.5)
RFall <- data.frame('Class'=NA, 'Prediction'=NA)
# Modeling
set.seed(9999)
for(i in 1:k){
  TrainDiabetes <- diabetes[split.sample!=i, ]
  TestDiabetes <- diabetes[split.sample==i, ]
  RFmodel <- randomForest(class~., data = TrainDiabetes,
                          mtry = 4, ntree = 100, importance = TRUE)
  RFprediction <- predict(RFmodel, TestDiabetes[, -17], type = 'class')
  RFall <- rbind(RFall, data.frame('Class'=TestDiabetes[, 17],
                                   'Prediction'=RFprediction))
}
RFtable <- table(RFall)
rownames(RFtable) <- c('Negative', 'Positive')
colnames(RFtable) <- c('Negative', 'Positive')

## SVM
# Tuning parameters
set.seed(1123)
best_parameters <- tune.svm(class~., data = diabetes, 
                            gamma = seq(0, 0.15, by=0.1),
                            cost = seq(1, 10, by=1),
                            kernel = 'radial',
                            tunecontrol = tune.control(cross = 10))
best_gamma <- best_parameters$best.parameters[1]
best_cost <- best_parameters$best.parameters[2]
# Modeling
SVMall <- data.frame('Class'=NA, 'Prediction'=NA)
for(i in 1:k){
  TrainDiabetes <- diabetes[split.sample!=i, ]
  TestDiabetes <- diabetes[split.sample==i, ]
  SVMmodel <- svm(class~., data = TrainDiabetes, 
                  gamma = best_gamma, cost = best_cost,
                  type = 'C-classification', kernel = 'radial')
  SVMprediction <- predict(SVMmodel, TestDiabetes[, -17], type = 'class')
  SVMall <- rbind(SVMall, data.frame('Class'=TestDiabetes[, 17],
                                     'Prediction'=SVMprediction))
}
SVMtable <- table(SVMall)
rownames(SVMtable) <- c('Negative', 'Positive')
colnames(SVMtable) <- c('Negative', 'Positive')

## Naive Bayes classifier
NBall <- data.frame('Class'=NA, 'Prediction'=NA)
for(i in 1:k){
  TrainDiabetes <- diabetes[split.sample!=i, ]
  TestDiabetes <- diabetes[split.sample==i, ]
  NBmodel <- naiveBayes(class~., data = TrainDiabetes, 
                        type = 'class')
  NBprediction <- predict(NBmodel, TestDiabetes[, -17], 
                          type = 'class')
  NBall <- rbind(NBall, data.frame('Class'=TestDiabetes[, 17],
                                   'Prediction'=NBprediction))
}
NBtable <- table(NBall)
rownames(NBtable) <- c('Negative', 'Positive')
colnames(NBtable) <- c('Negative', 'Positive')

## Compare results
method <- c('Logistic regression', 'Lasso', 'KNN', 'Random forest',
            'SVM', 'Naive Bayes')
caall <- c(ca(LRallTable), ca(LassoTable), ca(KnnTable), ca(RFtable),
           ca(SVMtable), ca(NBtable))
precisionall <- c(precision(LRallTable), precision(LassoTable),
                  precision(KnnTable), precision(RFtable),
                  precision(SVMtable), precision(NBtable))
sensitivityall <- c(sensitivity(LRallTable), sensitivity(LassoTable),
                    sensitivity(KnnTable), sensitivity(RFtable),
                    sensitivity(SVMtable), sensitivity(NBtable))
specificityall <- c(specificity(LRallTable), specificity(LassoTable),
                    specificity(KnnTable), specificity(RFtable),
                    specificity(SVMtable), specificity(NBtable))
f1all <- f1(precisionall, sensitivityall)
result <- data.frame('Method'=method,
                     'CA'=round(caall, 3),
                     'Precision'=round(precisionall, 3),
                     'Sensitivity'=round(sensitivityall, 3),
                     'Specificity'=round(specificityall, 3),
                     'F1 score'=round(f1all, 3))
knitr::kable(result,
             caption = 'Performance metrics')
# compare f1 score
ggplot(data = result) +
  geom_col(mapping = aes(x=Method, y=F1.score))
#Importance of variables from random forest model
varImpPlot(RFmodel, type=2)