########################################## Description ##############################################
#
#####################################################################################################


rm(list = ls())
### Libraries ###
library(data.table)
library(tibble)
library(tidyverse)
library(janitor)
library(FactoMineR)
library(factoextra)
library(corrplot)

### Parameters ###
Rep <- 'C:/Users/alexi/OneDrive/Documents/DocumentsPersonnels/SynchroDropbox/Dropbox/IODAA2020/Agro/ProjetFilRouge/ProjetFilRouge/'

### Functions ###
PCAWithoutAgeClasses <- function(Data, WhereToSavePlot){
  ### Analysis
  ResPCA <- PCA(Data, scale.unit = TRUE, ncp = 43, graph = TRUE) # 43 explaining 80 % variance
  EigenValues <- get_eigenvalue(ResPCA)
  Var <- get_pca_var(ResPCA)
  ResDesc <- dimdesc(ResPCA, axes = c(1,2), proba = 0.05)
  Individuals <- get_pca_ind(ResPCA)
  
  Results <- list(ResPCA = ResPCA, EigenValues = EigenValues, Variables = Var, DimensionDescription = ResDesc, Individuals = Individuals)
  
  ### Graphics
  pdf(file = WhereToSavePlot)
  fviz_eig(ResPCA, addlabels = TRUE, ylim = c(0, 50))       # Dimensions function of percentage of explained variance
  fviz_pca_var(ResPCA, col.var = "cos2",                    # Variables visualisation
               gradient.cols = c("#00AFBB", "#E7B800", "#FC4E07"),
               label = 'none'#,
               #repel = TRUE # Avoid text overlapping
               )                                      
  #corrplot(Var$cos2, is.corr=FALSE)                         # Cos2 of variables on all PC
  corrplot(Var$contrib, is.corr=FALSE)                      # Contribution of variables on all PC 
  fviz_cos2(ResPCA, choice = "var", axes = 1:2)             # Cos2 of variables on PC1 and 2 
  fviz_contrib(ResPCA, choice = "var", axes = 1, top = 10)  # Contribution of variables to PC1
  fviz_contrib(ResPCA, choice = "var", axes = 2, top = 10)  # Contribution of variables to PC2
  fviz_pca_ind (ResPCA, col.ind = "cos2",                   # Placing individuals on PC1 and 2
                gradient.cols = c("#00AFBB", "#E7B800", "#FC4E07"),
                label = 'none'#,
                #repel = TRUE # Avoid text overlapping
  )
  fviz_contrib(ResPCA, choice = "ind", axes = 1:2)          # Contribution of individuals on PC1 and 2
  fviz_cos2(ResPCA, choice = "ind")                         # Cos2 of individuals
  dev.off()
  return(Results)
}

### Executing PCA ###
### Preparing data
Genus <- fread(paste0(Rep,'Data/WorkData/WithoutAgeClasses/GenusWithAgeFull.csv')) %>% 
  column_to_rownames('Subject') %>% 
  select(-Gender)

GenusMale <- fread(paste0(Rep,'Data/WorkData/WithoutAgeClasses/GenusWithAgeFull.csv')) %>% 
  column_to_rownames('Subject') %>% 
  filter(Gender == 'male') %>% 
  select(-Gender)

GenusFemale <- fread(paste0(Rep,'Data/WorkData/WithoutAgeClasses/GenusWithAgeFull.csv')) %>% 
  column_to_rownames('Subject') %>% 
  filter(Gender == 'female') %>% 
  select(-Gender)

Species <- fread(paste0(Rep,'Data/WorkData/WithoutAgeClasses/SpeciesWithAgeFull.csv')) %>% 
  column_to_rownames('Subject') %>% 
  select(-Gender)

SpeciesMale <- fread(paste0(Rep,'Data/WorkData/WithoutAgeClasses/SpeciesWithAgeFull.csv')) %>% 
  column_to_rownames('Subject') %>% 
  filter(Gender == 'male') %>% 
  select(-Gender)

SpeciesFemale <- fread(paste0(Rep,'Data/WorkData/WithoutAgeClasses/SpeciesWithAgeFull.csv')) %>% 
  column_to_rownames('Subject') %>% 
  filter(Gender == 'female') %>% 
  select(-Gender)

### Doing PCA
PCAGenus <- PCAWithoutAgeClasses(Genus, paste0(Rep,'Reports/PCA/GenusResults.pdf'))
saveRDS(PCAGenus, paste0(Rep,'Results/PCA/GenusPCA.rds'))

PCAGenusMale <- PCAWithoutAgeClasses(GenusMale, paste0(Rep,'Reports/PCA/GenusMaleResults.pdf'))
saveRDS(PCAGenusMale, paste0(Rep,'Results/PCA/GenusMalePCA.rds'))

PCAGenusFemale <- PCAWithoutAgeClasses(GenusFemale, paste0(Rep,'Reports/PCA/GenusFemaleResults.pdf'))
saveRDS(PCAGenusFemale, paste0(Rep,'Results/PCA/GenusFemalePCA.rds'))

PCASpecies <- PCAWithoutAgeClasses(Genus, paste0(Rep,'Reports/PCA/SpeciesResults.pdf'))
saveRDS(PCASpecies, paste0(Rep,'Results/PCA/SpeciesPCA.rds'))

PCASpeciesMale <- PCAWithoutAgeClasses(SpeciesMale, paste0(Rep,'Reports/PCA/SpeciesMaleResults.pdf'))
saveRDS(PCASpeciesMale, paste0(Rep,'Results/PCA/SpeciesMalePCA.rds'))

PCASpeciesFemale <- PCAWithoutAgeClasses(SpeciesFemale, paste0(Rep,'Reports/PCA/SpeciesFemaleResults.pdf'))
saveRDS(PCASpeciesFemale, paste0(Rep,'Results/PCA/SpeciesFemalePCA.rds'))