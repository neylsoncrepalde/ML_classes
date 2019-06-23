## RLab 6 - Decision Trees

# Fitting Classification Trees

library(tree)
library(ISLR)
library(dplyr)

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
text(tree.carseats,pretty=0)

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

# Tentativa com o outro melhor resultado
prune.carseats = prune.misclass(tree.carseats, best=15)
plot(prune.carseats)
text(prune.carseats, pretty=0)
tree.pred = predict(prune.carseats, Carseats.test, type="class")
table(tree.pred,Carseats.test$High)
(102+53)/200 # Acurácia

# Fitting Regression Trees
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


