### Regressão Logística com os dados de Titanic
#install.packages("titanic")

library(titanic)
library(dplyr)
data("titanic_train")
head(titanic_train)

# Montando o modelo
model = glm(Survived ~ Sex + Age + factor(Pclass), data = titanic_train, family = binomial())
summary(model)

# Predizendo a probabilidade de sobrevivência de Neylson
b0 = coef(model)[1]
bsex = coef(model)[2]
bage = coef(model)[3]
b2classe = coef(model)[4]
b3classe = coef(model)[5]

exp(b0 + bsex*1 + bage*32 + b2classe*0 + b3classe*1) / 
  (1 + exp(b0 + bsex*1 + bage*32 + b2classe*0 + b3classe*1))

# E se neylson viajasse na primeira classe?
exp(b0 + bsex*1 + bage*32 + b2classe*0 + b3classe*0) / 
  (1 + exp(b0 + bsex*1 + bage*32 + b2classe*0 + b3classe*0))
  
