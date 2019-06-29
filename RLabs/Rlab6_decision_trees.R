## Machine Learning
## Ciências de Dados Izabela Hendrix
## Izabela Tech Open Day
## Prof. Dr. Neylson Crepalde
## RLab 6 - Tree based models
###################################

library(tree)
library(ISLR)
library(dplyr)
library(pROC)
library(readr)
library(purrr)

## Fitting Classification Trees ####

data("Carseats") 
Carseats = Carseats %>% as_tibble

summary(Carseats$Sales)

# Cria uma variável binária
Carseats$High = ifelse(Carseats$Sales <= 8, "No", "Yes") %>% as.factor # tem que ser factor


# T0
# Grow a full tree
tree.carseats = tree(High ~. -Sales, Carseats)
summary(tree.carseats)
plot(tree.carseats)
text(tree.carseats,pretty=1)

# Abordagem do train set test
set.seed(2)
train = sample(1:nrow(Carseats), nrow(Carseats)*0.5)
Carseats.test=Carseats[-train,]

tree.carseats = tree(High ~. -Sales, Carseats, subset=train)

# produz as predições
tree.pred = predict(tree.carseats, Carseats.test, type="class")
# Matriz de confusão
table(tree.pred, Carseats.test$High)
(104+50)/200  # Acurácia

auc(Carseats.test$High, as.numeric(tree.pred))

# Usando CV para escolher o melhor tamanho de árvore
set.seed(3)
cv.carseats = cv.tree(tree.carseats, FUN = prune.misclass)
names(cv.carseats)
cv.carseats
par(mfrow=c(1,2))
plot(cv.carseats$size,cv.carseats$dev,type="b")
plot(cv.carseats$k,cv.carseats$dev,type="b")

# Faz o prunning com o melhor resultado
prune.carseats=prune.misclass(tree.carseats,best=9)
par(mfrow=c(1,1))
plot(prune.carseats)
text(prune.carseats, pretty=0)
tree.pred = predict(prune.carseats, Carseats.test, type="class")
table(tree.pred,Carseats.test$High)
(97+58)/200 # Acurácia

auc(Carseats.test$High, as.numeric(tree.pred))

# Tentativa com o outro melhor resultado
prune.carseats = prune.misclass(tree.carseats, best=15)
plot(prune.carseats)
text(prune.carseats, pretty=0)
tree.pred = predict(prune.carseats, Carseats.test, type="class")
table(tree.pred,Carseats.test$High)
(102+53)/200 # Acurácia

auc(Carseats.test$High, as.numeric(tree.pred))

## Fitting Regression Trees ####
# Vamos trabalhar com o banco de dados Boston
library(MASS)
set.seed(1)
train = sample(1:nrow(Boston), nrow(Boston)/2)
Boston = Boston %>% as_tibble

tree.boston = tree(medv ~., Boston, subset=train)
summary(tree.boston)
plot(tree.boston)
text(tree.boston, pretty=0)

# Escolhendo o melhor tamanho com CV
cv.boston = cv.tree(tree.boston)
plot(cv.boston$size, cv.boston$dev, type='b')
# 1 tentativa = 5 folhas
prune.boston = prune.tree(tree.boston, best=5)
plot(prune.boston)
text(prune.boston, pretty=0)
yhat = predict(tree.boston, newdata = Boston[-train, ])
boston.test = Boston[-train, "medv"]
plot(yhat, boston.test$medv)
abline(0,1)
mean((yhat - boston.test$medv)^2) #MSE


## Bagging and Random Forests ####

library(randomForest)
set.seed(1)

bag.boston = randomForest(medv ~., data=Boston, subset=train,
                          mtry=13, importance=TRUE)
bag.boston
yhat.bag = predict(bag.boston, newdata=Boston[-train,])

plot(yhat.bag, boston.test$medv)
abline(0,1)
mean((yhat.bag-boston.test$medv)^2)

bag.boston = randomForest(medv~., data=Boston, subset=train,
                          mtry=13, ntree=25)
yhat.bag = predict(bag.boston, newdata=Boston[-train,])
mean((yhat.bag-boston.test$medv)^2)

set.seed(1)
rf.boston = randomForest(medv~., data=Boston,
                         subset=train, mtry=6, importance=TRUE)
yhat.rf = predict(rf.boston,newdata=Boston[-train,])
mean((yhat.rf-boston.test$medv)^2)
importance(rf.boston)
varImpPlot(rf.boston)


## Boosting ####

library(gbm)

set.seed(1)

boost.boston = gbm(medv~.,
                   data=Boston[train,],  # Apenas treino
                   distribution="gaussian", # Regressão
                   n.trees=5000,  # B
                   interaction.depth=4) # d
summary(boost.boston)

# Verificando o efeito marginal de rm e lstat
plot(boost.boston,i="rm")
plot(boost.boston,i="lstat")

# verifica o erro de teste
yhat.boost = predict(boost.boston, newdata=Boston[-train,], n.trees=5000)
mean((yhat.boost-boston.test$medv)^2)

# Agora vamos fazer o mesmo modelo de boosting com outro valor de lambda
boost.boston = gbm(medv~.,
                   data=Boston[train,],
                   distribution="gaussian",
                   n.trees=5000,
                   interaction.depth=4,
                   shrinkage=0.2,  # Valor de lambda
                   verbose=F)  # Não verboso

# Verifica o erro de teste
yhat.boost = predict(boost.boston, newdata=Boston[-train,], n.trees=5000)
mean((yhat.boost-boston.test$medv)^2)


#### Aplicações
# Random Forest Classifier: Credit Data
credit = read_csv("https://github.com/neylsoncrepalde/RFSVM/blob/master/UCI_Credit_Card.csv?raw=true")
credit$SEX = factor(credit$SEX, levels = c(1, 2), labels = c("Male", "Female"))
credit$EDUCATION = factor(credit$EDUCATION, levels = c(1,2,3,4,5,6),
                          labels = c("Grad School", "University", "High School",
                                     "Others", "Unknown", "Unknown"))
credit$MARRIAGE = factor(credit$MARRIAGE, levels = c(1,2,3),
                         labels = c("Married", "Single", "Other"))
credit$default.payment.next.month = factor(credit$default.payment.next.month,
                                           levels = c(0,1), labels = c("No", "Yes"))

table(credit$PAY_0)

## Verificando a variável resposta
table(credit$default.payment.next.month)
prop.table(table(credit$default.payment.next.month))*100

# Missing treatment
dim(credit)
dim(na.omit(credit))
# Não há tanto problema em jogar os missings fora

credit = na.omit(credit)

## Divide train test
set.seed(5)
train = sample(1:nrow(credit), nrow(credit)*.7)

fit = randomForest(default.payment.next.month ~. - ID, 
                   data=credit, 
                   subset=train,
                   mtry=5,   # sqrt(p) 
                   importance=TRUE)
fit

importance(fit)
varImpPlot(fit)

## Calculate train error and test error
yhattrain = predict(fit, type = "class")
yhattest = predict(fit, newdata = credit[-train, ], type = "class")

# Matriz de confusão
table(credit$default.payment.next.month[train], yhattrain)
table(credit$default.payment.next.month[-train], yhattest)

# Erro de treino
auc(credit$default.payment.next.month[train], as.numeric(yhattrain))
# Erro de teste
auc(credit$default.payment.next.month[-train], as.numeric(yhattest))


### Random Forests Regression
## Prevendo o limite de crédito
fit2 = randomForest(LIMIT_BAL ~. -ID - default.payment.next.month, 
                    data=credit, subset = train,
                    mtry=5, importance=T, ntree=1000)
fit2
importance(fit2)
varImpPlot(fit2)

yhattrain = predict(fit2)
yhattest = predict(fit2, newdata = credit[-train, ])

## RMSE
sqrt(mean((credit$LIMIT_BAL[train] - yhattrain)^2))

sqrt(mean((credit$LIMIT_BAL[-train] - yhattest)^2))

plot(fit2)




