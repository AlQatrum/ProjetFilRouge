library(xlsx)
library(ggplot2)
library(tidyverse)
library(dplyr)
library(corrplot)
library(FactoMineR)
library(factoextra)
library(Hmisc)

#importation des individus depuis le presse-papier
individus <- read.table(file = "clipboard", sep = "\t", header=TRUE)
colnames(individus)<- c('Subject','age','gender','classe_age')
print(head(individus))

#table des echantillons en genre, inversement des colonnes et des lignes, ajout de l'âge
echantillons_d_s<- read.xlsx("Specieslevel_diversity.xlsx", sheetIndex=1)
echantillons_d_s <- rename(echantillons_d_s, Subject=sample)

echantillons_d_s <- merge(individus, echantillons_d_s, by="Subject")
echantillons_d_s <- select(echantillons_d_s, -(1:3))


#plot des deux indices de diversité en fonction de chaque classe d'âge
boxplot(echantillons_d_s['evenness_index'][echantillons_d_s['classe_age']=="1"], echantillons_d_s['evenness_index'][echantillons_d_s['classe_age']=="2"],echantillons_d_s['evenness_index'][echantillons_d_s['classe_age']=="3"], echantillons_d_s['evenness_index'][echantillons_d_s['classe_age']=="4"], echantillons_d_s['evenness_index'][echantillons_d_s['classe_age']=="5"], echantillons_d_s['evenness_index'][echantillons_d_s['classe_age']=="6"],echantillons_d_s['evenness_index'][echantillons_d_s['classe_age']=="7"], echantillons_d_s['evenness_index'][echantillons_d_s['classe_age']=="8"], echantillons_d_s['evenness_index'][echantillons_d_s['classe_age']=="9"], echantillons_d_s['evenness_index'][echantillons_d_s['classe_age']=="10"], main="diversité beta dde l'espece selon la classe d'age")

boxplot(echantillons_d_s['shannon_index'][echantillons_d_s['classe_age']=="1"], echantillons_d_s['shannon_index'][echantillons_d_s['classe_age']=="2"],echantillons_d_s['shannon_index'][echantillons_d_s['classe_age']=="3"], echantillons_d_s['shannon_index'][echantillons_d_s['classe_age']=="4"], echantillons_d_s['shannon_index'][echantillons_d_s['classe_age']=="5"], echantillons_d_s['shannon_index'][echantillons_d_s['classe_age']=="6"],echantillons_d_s['shannon_index'][echantillons_d_s['classe_age']=="7"], echantillons_d_s['shannon_index'][echantillons_d_s['classe_age']=="8"], echantillons_d_s['shannon_index'][echantillons_d_s['classe_age']=="9"], echantillons_d_s['shannon_index'][echantillons_d_s['classe_age']=="10"], main="diversité alpha de l'espece selon la classe d'age")


#test de significativité par analyse de la variance : anova
test_shan<-aov(classe_age ~ shannon_index, data=echantillons_d_g)
test_eve<-aov(classe_age ~ evenness_index, data=echantillons_d_g)
