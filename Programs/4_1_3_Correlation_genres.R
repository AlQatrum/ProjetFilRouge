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
echantillons_g<- read.csv("genus_level2.csv", sep=',', dec='.', header=TRUE)
echantillons_genus <- data.frame(t(echantillons_g[-1]))
colnames(echantillons_genus) <- echantillons_g[, 1]



#corrélation avec test de significativité 

test_correlation <- rcorr(as.matrix(echantillons_genus),type=c("pearson","spearman"))

#fonction qui permet d'aplatir la matrice de corrélation
flattenCorrMatrix <- function(cormat, pmat) {
  ut <- upper.tri(cormat)
  data.frame(
    row = rownames(cormat)[row(cormat)[ut]],
    column = rownames(cormat)[col(cormat)[ut]],
    cor  =(cormat)[ut],
    p = pmat[ut]
  )
}

#application de la fonction précédente à la matrice avec les tests de significativité
correlation_pvalue <- flattenCorrMatrix(test_correlation$r, test_correlation$P)

#séléction des paires de genres significativement corrélés avec une corrélation supérieure à 0,5
correlation_pvalue <- filter(correlation_pvalue , p <= 0.05)
correlation_pvalue <- filter(correlation_pvalue, cor>0.5)

#création d'un fichier csv
write.csv(correlation_pvalue, "correlations_genres_spearman.csv")

saveRDS(correlation_pvalue, "table_paires_genres_correles_signif.rds")

