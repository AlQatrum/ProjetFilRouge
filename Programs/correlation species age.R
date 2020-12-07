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
echantillons_genus <- select(echantillons_genus, -(1:3))

genus <- select(echantillons_genus, -1)
classe <- select(echantillons_genus, 1)

#test significativité

test_correlation <- rcorr(as.matrix(genus),as.matrix(classe),type=c("pearson","spearman"))


flattenCorrMatrix <- function(cormat, pmat) {
  ut <- upper.tri(cormat)
  data.frame(
    row = rownames(cormat)[row(cormat)[ut]],
    column = rownames(cormat)[col(cormat)[ut]],
    cor  =(cormat)[ut],
    p = pmat[ut]
  )
}

correlation_pvalue <- flattenCorrMatrix(test_correlation$r, test_correlation$P)

correlation_pvalue <- filter(correlation_pvalue, column=="classe_age")

correlation_pvalue <- filter(correlation_pvalue , p <= 0.05)
saveRDS(correlation_pvalue, "table_genres_correles_classeage_signif.rds")

