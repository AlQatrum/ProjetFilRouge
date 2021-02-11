library(xlsx)
library(ggplot2)
library(tidyverse)
library(dplyr)
library(corrplot)
library(FactoMineR)
library(Hmisc)

#importation de la table d'individus depuis le presse-papier
individus <- read.table(file = "clipboard", sep = "\t", header=TRUE)
colnames(individus)<- c('Subject','age','gender','classe_age')
print(head(individus))


#importation de la table des echantillons en es^èce, inversement des colonnes et des lignes, ajout de l'âge
echantillons_s<- read.csv("species_level_table.csv", sep=',', dec='.', header=TRUE)
echantillons_species <- data.frame(t(echantillons_s[-1]))
colnames(echantillons_species) <- echantillons_s[, 1]
echantillons_species <- cbind(echantillons_species, row=row.names(echantillons_species))
echantillons_species <- merge(individus, echantillons_species, by.x="Subject", by.y='row')

#séléection de la colonne copri et des informations sur l'individu
echantillons_species_copri <- select(echantillons_species, 1:4, s__copri)
#récupération des deux colonnes d'intérêt
echantillons_species_copri <- select(echantillons_species_copri, 4,5)
#garder les individus qui présentent copri
echantillons_species_copri <- filter(echantillons_species_copri, s__copri != 0)
#ne garder que la classe d'âge de ces individus
echantillons_species_copri_classe <- select(echantillons_species_copri, 1)

#récupération d'une table avec la classe d'âge de tous les individus
echantillons_classe <- select(echantillons_species, 4)

#à partir de la table "echantillons_species_copri_classe"
#trouver le nombre d'individus présentant copri dans chaque classe
copri_1 = nrow(subset(echantillons_species_copri_classe, classe_age==1))
copri_2 = nrow(subset(echantillons_species_copri_classe, classe_age==2))
copri_3 = nrow(subset(echantillons_species_copri_classe, classe_age==3))
copri_4 = nrow(subset(echantillons_species_copri_classe, classe_age==4))
copri_5 = nrow(subset(echantillons_species_copri_classe, classe_age==5))
copri_6 = nrow(subset(echantillons_species_copri_classe, classe_age==6))
copri_7 = nrow(subset(echantillons_species_copri_classe, classe_age==7))
copri_8 = nrow(subset(echantillons_species_copri_classe, classe_age==8))
copri_9 = nrow(subset(echantillons_species_copri_classe, classe_age==9))
copri_10 = nrow(subset(echantillons_species_copri_classe, classe_age==10))

#à partir de la table "echantillons_classe"
#trouver le nombre d'individus dans chaque classe
classe1=nrow(subset(echantillons_classe, classe_age==1))
classe2=nrow(subset(echantillons_classe, classe_age==2))
classe3=nrow(subset(echantillons_classe, classe_age==3))
classe4=nrow(subset(echantillons_classe, classe_age==4))
classe5=nrow(subset(echantillons_classe, classe_age==5))
classe6=nrow(subset(echantillons_classe, classe_age==6))
classe7=nrow(subset(echantillons_classe, classe_age==7))
classe8=nrow(subset(echantillons_classe, classe_age==8))
classe9=nrow(subset(echantillons_classe, classe_age==9))
classe10=nrow(subset(echantillons_classe, classe_age==10))

#faire le rapport pour avoir la proportion d'individus présentant copri
M1=copri_1/classe1
M2=copri_2/classe2
M3=copri_3/classe3
M4=copri_4/classe4
M5=copri_5/classe5
M6=copri_6/classe6
M7=copri_7/classe7
M8=copri_8/classe8
M9=copri_9/classe9
M10=copri_10/classe10

#création d'une table avec les proportions d'indidividus de chaque classe avec copri
data <- data.frame(c(M1,M2,M3,M4,M5,M6,M7,M8,M9,M10))
names(data)[1]<-"proportion_copri"
data <- cbind(data, c(1,2,3,4,5,6,7,8,9,10))
names(data)[2]<-"classe_age"

#création et affichage d'un plot
qplot(classe_age, data=echantillons_species_copri_classe, geom="histogram")
qplot(classe_age, data=echantillons_classe, geom="histogram")
hist(data)

p<-ggplot(data=data, aes(x=classe_age, y=proportion_copri))+geom_bar(stat="identity")
p
boxplot(data['proportion_copri'][data['classe_age']=="1"], data['proportion_copri'][data['classe_age']=="2"],data['proportion_copri'][data['classe_age']=="3"], data['proportion_copri'][data['classe_age']=="4"], data['proportion_copri'][data['classe_age']=="5"], data['proportion_copri'][data['classe_age']=="6"],data['proportion_copri'][data['classe_age']=="7"], data['proportion_copri'][data['classe_age']=="8"], data['proportion_copri'][data['classe_age']=="9"], data['proportion_copri'][data['classe_age']=="10"], main="proportion de population présentant des bactéries copri par classe d age")



library(ggpubr)
library(rstatix)

echantillons_species_copri$classe_age <- ordered(echantillons_species_copri$classe_age,levels = c("1", "2", "3","4","5","6","7","8","9","10"))
levels(echantillons_species_copri$classe_age)


ggboxplot(echantillons_species_copri, x = "classe_age", y = "s__copri", 
          order = c("1", "2", "3","4","5","6","7","8","9","10"),
          ylab = "proportion copri", xlab = "classe age")
signif <- kruskal.test(s__copri ~ classe_age, data = echantillons_species_copri)
signif
