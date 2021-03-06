library(xlsx)
library(ggplot2)
library(tidyverse)
library(dplyr)
library(corrplot)
library(FactoMineR)
library(factoextra)
library(Hmisc)

#importation de la table des individus depuis le presse papier
individus <- read.table(file = "clipboard", sep = "\t", header=TRUE)
colnames(individus)<- c('Subject','age','gender','classe_age')
print(head(individus))


#table des echantillons en genre, inversement des colonnes et des lignes, ajout de l'âge
echantillons_g<- read.csv("genus_level2.csv", sep=',', dec='.', header=TRUE)
echantillons_genus <- data.frame(t(echantillons_g[-1]))
colnames(echantillons_genus) <- echantillons_g[, 1]

echantillons_genus <- cbind(echantillons_genus, row=row.names(echantillons_genus))
echantillons_genus <- merge(individus, echantillons_genus, by.x="Subject", by.y='row')


#recupération des colonnes d'intérêt
table_acp <- select(echantillons_genus, -(1:3))

#ACP
res_acp <- PCA(table_acp, ncp=50, graph=FALSE)

#enregistement de l'ACP
saveRDS(res_acp, "res_acp_classesage_10.rds")

#affichage des résultats
print(res_acp)
eig_val <- get_eigenvalue(res_acp)
eig_val

#plots
fviz_pca_var(res_acp, col.var = "black")

fviz_pca_ind (res_acp, habillage="classe_age", label=FALSE)

fviz_pca_ind (res_acp, select.ind=list(classe_age=c(1,2,3,4,5)), habillage="classe_age", label=FALSE)





#selection des genres qui contribuent le plus aux cinquante premières composantes


#récupération des varaibles de l'ACP
var <- get_pca_var(res_acp)

variables_ <- as.data.frame(var$contrib)

#récupération des 50 premières dimentsions et des genres qui y contribuent
dimension1 <- select(variables_, Dim.1)
dimension1 <- filter(dimension1, Dim.1>1)

dimension2 <- select(variables_, Dim.2)
dimension2 <- filter(dimension2, Dim.2>1)

dimension3 <- select(variables_, Dim.3)
dimension3 <- filter(dimension3, Dim.3>1)

dimension4 <- select(variables_, Dim.4)
dimension4 <- filter(dimension4, Dim.4>1)

dimension5 <- select(variables_, Dim.5)
dimension5 <- filter(dimension5, Dim.5>1)

dimension6 <- select(variables_, Dim.6)
dimension6 <- filter(dimension6, Dim.6>1)

dimension7 <- select(variables_, Dim.7)
dimension7 <- filter(dimension7, Dim.7>1)

dimension8 <- select(variables_, Dim.8)
dimension8 <- filter(dimension8, Dim.8>1)

dimension9 <- select(variables_, Dim.9)
dimension9 <- filter(dimension9, Dim.9>1)

dimension10 <- select(variables_, Dim.10)
dimension10 <- filter(dimension10, Dim.10>1)

dimension11 <- select(variables_, Dim.11)
dimension11 <- filter(dimension11, Dim.11>1)

dimension12 <- select(variables_, Dim.12)
dimension12 <- filter(dimension12, Dim.12>1)

dimension13 <- select(variables_, Dim.13)
dimension13 <- filter(dimension13, Dim.13>1)

dimension14 <- select(variables_, Dim.14)
dimension14 <- filter(dimension14, Dim.14>1)

dimension15 <- select(variables_, Dim.15)
dimension15 <- filter(dimension15, Dim.15>1)

dimension16 <- select(variables_, Dim.16)
dimension16 <- filter(dimension16, Dim.16>1)

dimension17 <- select(variables_, Dim.17)
dimension17 <- filter(dimension17, Dim.17>1)

dimension18 <- select(variables_, Dim.18)
dimension18 <- filter(dimension18, Dim.18>1)

dimension19 <- select(variables_, Dim.19)
dimension19 <- filter(dimension19, Dim.19>1)

dimension20 <- select(variables_, Dim.20)
dimension20 <- filter(dimension20, Dim.20>1)

dimension21 <- select(variables_, Dim.21)
dimension21 <- filter(dimension21, Dim.21>1)

dimension22 <- select(variables_, Dim.22)
dimension22 <- filter(dimension22, Dim.22>1)

dimension23 <- select(variables_, Dim.23)
dimension23 <- filter(dimension23, Dim.23>1)

dimension24 <- select(variables_, Dim.24)
dimension24 <- filter(dimension24, Dim.24>1)

dimension25 <- select(variables_, Dim.25)
dimension25 <- filter(dimension25, Dim.25>1)

dimension26 <- select(variables_, Dim.26)
dimension26 <- filter(dimension26, Dim.26>1)

dimension27 <- select(variables_, Dim.27)
dimension27 <- filter(dimension27, Dim.27>1)

dimension28 <- select(variables_, Dim.28)
dimension28 <- filter(dimension28, Dim.28>1)

dimension29 <- select(variables_, Dim.29)
dimension29 <- filter(dimension29, Dim.29>1)

dimension30 <- select(variables_, Dim.30)
dimension30 <- filter(dimension30, Dim.30>1)

dimension31 <- select(variables_, Dim.31)
dimension31 <- filter(dimension31, Dim.31>1)

dimension32 <- select(variables_, Dim.32)
dimension32 <- filter(dimension32, Dim.32>1)

dimension33 <- select(variables_, Dim.33)
dimension33 <- filter(dimension33, Dim.33>1)

dimension34 <- select(variables_, Dim.34)
dimension34 <- filter(dimension34, Dim.34>1)

dimension35 <- select(variables_, Dim.35)
dimension35 <- filter(dimension35, Dim.35>1)

dimension36 <- select(variables_, Dim.36)
dimension36 <- filter(dimension36, Dim.36>1)

dimension37 <- select(variables_, Dim.37)
dimension37 <- filter(dimension37, Dim.37>1)

dimension38 <- select(variables_, Dim.38)
dimension38 <- filter(dimension38, Dim.38>1)

dimension39 <- select(variables_, Dim.39)
dimension39 <- filter(dimension39, Dim.39>1)

dimension40 <- select(variables_, Dim.40)
dimension40 <- filter(dimension40, Dim.40>1)

dimension41 <- select(variables_, Dim.41)
dimension41 <- filter(dimension41, Dim.41>1)

dimension42 <- select(variables_, Dim.42)
dimension42 <- filter(dimension42, Dim.42>1)

dimension43 <- select(variables_, Dim.43)
dimension43 <- filter(dimension43, Dim.43>1)

dimension44 <- select(variables_, Dim.44)
dimension44 <- filter(dimension44, Dim.44>1)

dimension45 <- select(variables_, Dim.45)
dimension45 <- filter(dimension45, Dim.45>1)

dimension46 <- select(variables_, Dim.46)
dimension46 <- filter(dimension46, Dim.46>1)

dimension47 <- select(variables_, Dim.47)
dimension47 <- filter(dimension47, Dim.47>1)

dimension48 <- select(variables_, Dim.48)
dimension48 <- filter(dimension48, Dim.48>1)

dimension49 <- select(variables_, Dim.49)
dimension49 <- filter(dimension49, Dim.49>1)

dimension50 <- select(variables_, Dim.50)
dimension50 <- filter(dimension50, Dim.50>1)

typeof(variables_)



#vecteur des genres des 50 dimensions d'intérêt
liste_dimension<-c(list(row.names(dimension1)),row.names(dimension2),row.names(dimension3),row.names(dimension4),row.names(dimension5),row.names(dimension6),row.names(dimension7),row.names(dimension8),row.names(dimension9),row.names(dimension10),row.names(dimension11),row.names(dimension12),row.names(dimension13),row.names(dimension14),row.names(dimension15),row.names(dimension16),row.names(dimension17),row.names(dimension18),row.names(dimension19),row.names(dimension20),row.names(dimension21),row.names(dimension22),row.names(dimension23),row.names(dimension24),row.names(dimension25),row.names(dimension26),row.names(dimension27),row.names(dimension28),row.names(dimension29),row.names(dimension30),row.names(dimension31),row.names(dimension32),row.names(dimension33),row.names(dimension34),row.names(dimension35),row.names(dimension36),row.names(dimension37),row.names(dimension38),row.names(dimension39),row.names(dimension40),row.names(dimension41),row.names(dimension42),row.names(dimension43),row.names(dimension44),row.names(dimension45),row.names(dimension46),row.names(dimension47),row.names(dimension48),row.names(dimension49),row.names(dimension50))

#initialisation d'une liste vide dans laquelle stocker les genres séléctionnés
genus_selectionne=NULL


#creation d'une fonction pour vérifier la nom présence d'un élement x dans une table
`%not in%` <- function (x, table) is.na(match(x, table, nomatch=NA_integer_))


#parcours de la liste des 50 dimensions pour récupérer les genres sans répétition avec la fonction précédente
for (i in 1:50) {
  presence=0
  for (l in 1:length((liste_dimension[[i]]))) {
    for (j in 1:50){
      if (liste_dimension[[i]][l] %in% liste_dimension[[j]]){
        presence=presence+1
      }
    }
  }
    if (presence>=1){
      if (liste_dimension[[i]][l] %not in% genus_selectionne){
        genus_selectionne[length(genus_selectionne)+1]=liste_dimension[[i]][l]
      }
    }
}


#affichage final
genus_selectionne<-genus_selectionne[1]+genus_selectionne[3:33]

