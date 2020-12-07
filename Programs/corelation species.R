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

variables <- select(echantillons_genus, -(1:4))


#matrice des corrélations
correlation <- cor(variables)
corrplot(correlation, is.corr=FALSE)


correlation_signif <- data.frame(Genus1=character(),
                                 Genus2=character(), 
                                 Correlation=factor()) 

for (l in 1:nrow(correlation)){
  for (c in 1:ncol(correlation)){
    if (correlation[l,c]>0.5 & correlation[l,c]!=1){
      correlation_signif <- rbind( correlation_signif, c(rownames(correlation)[l],
                                                         rownames(correlation)[c], 
                                                         correlation[l,c]))
    }
  }
} 

write.csv(correlation_signif, "correlation_significatives_classes_age.csv")



#test significativité

test_correlation <- rcorr(as.matrix(variables),type=c("pearson","spearman"))


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

correlation_pvalue <- filter(correlation_pvalue , p <= 0.05)
correlation_pvalue <- filter(correlation_pvalue, cor>0.5)
saveRDS(correlation_pvalue, "table_paires_genres_correles_signif.rds")

