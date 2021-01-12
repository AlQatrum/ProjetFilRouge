########################################## Description ##############################################
#                                                                                        
#####################################################################################################


rm(list = ls())
### Libraries ###
library(data.table)
library(tibble)
library(tidyverse)
library(corrplot)

### Parameters ###
Rep <- 'C:/Users/alexi/OneDrive/Documents/DocumentsPersonnels/SynchroDropbox/Dropbox/IODAA2020/Agro/ProjetFilRouge/ProjetFilRouge/'

### Reading files ###
Genus <- fread(paste0(Rep,'Data/WorkData/WithoutAgeClasses/GenusWithAgeFull.csv')) %>% 
  column_to_rownames('Subject') 

Species <- fread(paste0(Rep,'Data/WorkData/WithoutAgeClasses/SpeciesWithAgeFull.csv')) %>% 
  column_to_rownames('Subject')
  

### Check for correlations between bacteria ###
BacteriaGenus <- Genus %>% 
  select(-Gender, -Age)



for i in 1:ncol(BacteriaGenus){
  for j in 1:ncol(BacteriaGenus){
    
    
  }
  
}


x <- c(44.4, 45.9, 41.9, 53.3, 44.7, 44.1, 50.7, 45.2, 60.1)
y <- c( 2.6,  3.1,  2.5,  5.0,  3.6,  4.0,  5.2,  2.8,  3.8)
result <- cor.test(CorrBacteriaGenus, method = "spearman")

result$statistic
#  S 
# 48 