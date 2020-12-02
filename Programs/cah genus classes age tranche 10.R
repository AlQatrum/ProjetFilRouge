library(xlsx)
library(ggplot2)
library(tidyverse)
library(dplyr)
library(corrplot)
library(FactoMineR)

#table des individus
individus <- read.table(file = "clipboard", sep = "\t", header=TRUE)
colnames(individus)<- c('Subject','age','gender','classe_age')


#table des echantillons en genre, inversement des colonnes et des lignes, ajout de l'âge
echantillons_g<- read.csv("genus_level2.csv", sep=',', dec='.', header=TRUE)
echantillons_genus <- data.frame(t(echantillons_g[-1]))
colnames(echantillons_genus) <- echantillons_g[, 1]
echantillons_genus <- cbind(echantillons_genus, row=row.names(echantillons_genus))
echantillons_genus <- merge(individus, echantillons_genus, by.x="Subject", by.y='row')

table_cah <- select(echantillons_genus, -(1:3))

rownames(table_cah) <- make.names(table_cah[,1], unique = TRUE)
table_cah[,1] <- NULL 


table_cah_cr <- scale(table_cah,center=T,scale=T)

dist_table_cah <- dist(table_cah_cr)

cah <- hclust(dist_table_cah,method="ward.D2")

plot(cah, cex=0.5)
