library(xlsx)
library(ggplot2)
library(tidyverse)
library(dplyr)
library(corrplot)
library(FactoMineR)
library(factoextra)
library(Hmisc)



#importation des données (table avec toutes les informations age-sexe-espèce-diversité) et sélection des colonnes d'intérêt (espèces)
table<-read.csv("table_age_1_sex_espece_diversite_VF.csv", sep=",", dec=".", header=TRUE)
table<- select(table, -c(1,2,3,4,5,76,77))


#corrélation avec test de significativité 

test_correlation <- rcorr(as.matrix(table),type=c("pearson","spearman"))


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
write.csv(correlation_pvalue, "correlations_espèces_spearman.csv")

saveRDS(correlation_pvalue, "table_paires_genres_correles_signif.rds")

