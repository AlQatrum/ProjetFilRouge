library(xlsx)
library(ggplot2)
library(tidyverse)
library(dplyr)
library(corrplot)
library(FactoMineR)
library(factoextra)
library(Hmisc)

#table des individus
individus <- read.table(file = "clipboard", sep = "\t", header=TRUE)
colnames(individus)<- c('Subject','age','gender','classe_age')
print(head(individus))

#table des echantillons en genre, inversement des colonnes et des lignes, ajout de l'âge
echantillons_g<- read.csv("genus_level2.csv", sep=',', dec='.', header=TRUE)
echantillons_genus <- data.frame(t(echantillons_g[-1]))
colnames(echantillons_genus) <- echantillons_g[, 1]
echantillons_genus <- cbind(echantillons_genus, row=row.names(echantillons_genus))
echantillons_genus <- merge(individus, echantillons_genus, by.x="Subject", by.y='row')
echantillons_genus <- select(echantillons_genus, -c(2,3))

#une table par classe d'âge avec 25 individus aléatoires
classe1 <- filter(echantillons_genus, classe_age==1)
classe1 <- sample_n(classe1, 25)
classe2 <- filter(echantillons_genus, classe_age==2)
classe3 <- filter(echantillons_genus, classe_age==3)
classe3 <- sample_n(classe3, 25)
classe4 <- filter(echantillons_genus, classe_age==4)
classe4 <- sample_n(classe4, 25)
classe5 <- filter(echantillons_genus, classe_age==5)
classe5 <- sample_n(classe5, 25)
classe6 <- filter(echantillons_genus, classe_age==6)
classe6 <- sample_n(classe6, 25)
classe7 <- filter(echantillons_genus, classe_age==7)
classe7 <- sample_n(classe7, 25)
classe8 <- filter(echantillons_genus, classe_age==8)
classe8 <- sample_n(classe8, 25)
classe9 <- filter(echantillons_genus, classe_age==9)
classe9 <- sample_n(classe9, 25)
classe10 <- filter(echantillons_genus, classe_age==10)

echantillon_final <- rbind(classe1, classe2, classe3, classe4, classe5, classe6, classe7, classe8, classe9, classe10)

write.csv(echantillon_final, "selection_individus.csv")
