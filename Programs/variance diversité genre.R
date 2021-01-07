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
echantillons_d_g<- read.xlsx("diversity_genus_evenness_shannon.xlsx", sheetIndex=1)
echantillons_d_g <- rename(echantillons_d_g, Subject=sample)

echantillons_d_g <- merge(individus, echantillons_d_g, by="Subject")
echantillons_d_g <- select(echantillons_d_g, -(1:3))


Shann_var_1=sd(filter(echantillons_d_g,classe_age==1)$shannon_index)
Shann_var_2=sd(filter(echantillons_d_g,classe_age==2)$shannon_index)
Shann_var_3=sd(filter(echantillons_d_g,classe_age==3)$shannon_index)
Shann_var_4=sd(filter(echantillons_d_g,classe_age==4)$shannon_index)
Shann_var_5=sd(filter(echantillons_d_g,classe_age==5)$shannon_index)
Shann_var_6=sd(filter(echantillons_d_g,classe_age==6)$shannon_index)
Shann_var_7=sd(filter(echantillons_d_g,classe_age==7)$shannon_index)
Shann_var_8=sd(filter(echantillons_d_g,classe_age==8)$shannon_index)
Shann_var_9=sd(filter(echantillons_d_g,classe_age==9)$shannon_index)
Shann_var_10=sd(filter(echantillons_d_g,classe_age==10)$shannon_index)

shann_var=c(Shann_var_1, Shann_var_2, Shann_var_3, Shann_var_4, Shann_var_5, Shann_var_6, Shann_var_7, Shann_var_8, Shann_var_9, Shann_var_10)


Eve_var_1=sd(filter(echantillons_d_g,classe_age==1)$evenness_index)
Eve_var_2=sd(filter(echantillons_d_g,classe_age==2)$evenness_index)
Eve_var_3=sd(filter(echantillons_d_g,classe_age==3)$evenness_index)
Eve_var_4=sd(filter(echantillons_d_g,classe_age==4)$evenness_index)
Eve_var_5=sd(filter(echantillons_d_g,classe_age==5)$evenness_index)
Eve_var_6=sd(filter(echantillons_d_g,classe_age==6)$evenness_index)
Eve_var_7=sd(filter(echantillons_d_g,classe_age==7)$evenness_index)
Eve_var_8=sd(filter(echantillons_d_g,classe_age==8)$evenness_index)
Eve_var_9=sd(filter(echantillons_d_g,classe_age==9)$evenness_index)
Eve_var_10=sd(filter(echantillons_d_g,classe_age==10)$evenness_index)

Eve_var=c(Eve_var_1, Eve_var_2, Eve_var_3, Eve_var_4, Eve_var_5, Eve_var_6, Eve_var_7, Eve_var_8, Eve_var_9, Eve_var_10)

x<-seq(1,10)

df<-data.frame(x, shann_var, Eve_var)

colors <- c("Shann_var"='blue', "eve_var"='red')

p<- ggplot(df, aes(x))+geom_point(aes(y=shann_var, color='Shann_var'))+geom_point(aes(y=Eve_var, color='eve_var'))+labs(x="classe age", y="Variance", color="Legend", title="Echelle du genre")+scale_color_manual(name = "Group",values = c( "Shann_var" = "blue", "eve_var" = "red"),labels = c("Shannon", "Evenness"))
p

