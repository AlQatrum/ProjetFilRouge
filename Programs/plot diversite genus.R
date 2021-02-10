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
echantillons_d_g<- read.xlsx("diversity_genus_evenness_shannon.xlsx", sheetIndex=1)
echantillons_d_g <- rename(echantillons_d_g, Subject=sample)

echantillons_d_g <- merge(individus, echantillons_d_g, by="Subject")
echantillons_d_g <- select(echantillons_d_g, -(1:3))

#plot des deux indices de diversité en fonction de chaque classe d'âge
boxplot(echantillons_d_g['evenness_index'][echantillons_d_g['classe_age']=="1"], echantillons_d_g['evenness_index'][echantillons_d_g['classe_age']=="2"],echantillons_d_g['evenness_index'][echantillons_d_g['classe_age']=="3"], echantillons_d_g['evenness_index'][echantillons_d_g['classe_age']=="4"], echantillons_d_g['evenness_index'][echantillons_d_g['classe_age']=="5"], echantillons_d_g['evenness_index'][echantillons_d_g['classe_age']=="6"],echantillons_d_g['evenness_index'][echantillons_d_g['classe_age']=="7"], echantillons_d_g['evenness_index'][echantillons_d_g['classe_age']=="8"], echantillons_d_g['evenness_index'][echantillons_d_g['classe_age']=="9"], echantillons_d_g['evenness_index'][echantillons_d_g['classe_age']=="10"], main="diversité (evenness) du genre selon la classe d'age")

boxplot(echantillons_d_g['shannon_index'][echantillons_d_g['classe_age']=="1"], echantillons_d_g['shannon_index'][echantillons_d_g['classe_age']=="2"],echantillons_d_g['shannon_index'][echantillons_d_g['classe_age']=="3"], echantillons_d_g['shannon_index'][echantillons_d_g['classe_age']=="4"], echantillons_d_g['shannon_index'][echantillons_d_g['classe_age']=="5"], echantillons_d_g['shannon_index'][echantillons_d_g['classe_age']=="6"],echantillons_d_g['shannon_index'][echantillons_d_g['classe_age']=="7"], echantillons_d_g['shannon_index'][echantillons_d_g['classe_age']=="8"], echantillons_d_g['shannon_index'][echantillons_d_g['classe_age']=="9"], echantillons_d_g['shannon_index'][echantillons_d_g['classe_age']=="10"], main="diversité (shannon) du genre selon la classe d'age")

#test de significativité par analyse de la variance : anova
test_shan<-aov(classe_age ~ shannon_index, data=echantillons_d_g)
test_eve<-aov(classe_age ~ evenness_index, data=echantillons_d_g)
