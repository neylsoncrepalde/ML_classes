### Lab Regressão Linear
### Machine Learning
### Graduação em Ciências de Dados
### Izabela Hendrix
### Prof. Neylson Crepalde

# Se os pacotes necessários não estiverem instalados, faça a instalação
if (! "ISLR" %in% installed.packages()) install.packages("ISLR")
if (! "MASS" %in% installed.packages()) install.packages("MASS")
if (! "dplyr" %in% installed.packages()) install.packages("dplyr")
if (! "ggplot2" %in% installed.packages()) install.packages("ggplot2")
if (! "readr" %in% installed.packages()) install.packages("readr")
if (! "texreg" %in% installed.packages()) install.packages("texreg")

# Carregando o pacote do livro ISLR
library(readr)
library(dplyr)
library(texreg)
library(ggplot2)

# Carrega os dados Advertisement
adv = read_csv("http://www-bcf.usc.edu/~gareth/ISL/Advertising.csv")

# Tira a primeira coluna que contém apenas o número da linha
adv = adv %>% select(-X1)

# Verifica os primeiros casos
head(adv)

# Monta uma regressão para cada tipo de investimento
reg1 = lm(sales ~ TV, data = adv)
reg2 = lm(sales ~ radio, data = adv)
reg3 = lm(sales ~ newspaper, data = adv)

# Monta uma regressão múltipla com as 3 variáveis
reg_completa = lm(sales ~ TV + radio + newspaper, data = adv)

# Verifica os resultados das regressões de forma detalhada
summary(reg1)
summary(reg2)
summary(reg3)
summary(reg_completa)

# Usa o pacote texreg para visualizar todas as regressões, uma ao lado da outra
# screenreg mostra na tela. htmlreg salva em html. texreg salva em latex
screenreg(list(reg1, reg2, reg3, reg_completa))

# Produz gráficos de avaliação dos resíduos
par(mfrow = c(2,2)) # Divide a tela de plotagem em 2 linhas e 2 colunas
plot(reg_completa)
par(mfrow = c(1,1)) # Volta a tela de plotagem para exibir 1 gráficos por vez

# Pega o intervalo de confiança para os betas da primeira regressão
confidence = confint(reg1)

# Pega o intervalo de confiança para os valores preditos
fitted_confint = predict(reg1, interval = "prediction")

plot(x=adv$TV, y=adv$sales)
lines(x=adv$TV, y=fitted_confint[,1], col = "red", lwd = 2)
lines(x=adv$TV, y=fitted_confint[,2], col = "blue", lwd = 2)
lines(x=adv$TV, y=fitted_confint[,3], col = "blue", lwd = 2)

# O mesmo gráfico com ggplot2
ggplot(adv, aes(x = TV)) +
  geom_point(aes(y = sales)) +
  geom_line(aes(y = fitted_confint[,1]), col = "red", lwd = 1) +
  geom_line(aes(y = fitted_confint[,2]), col = "blue", lwd = 1) +
  geom_line(aes(y = fitted_confint[,3]), col = "blue", lwd = 1)
