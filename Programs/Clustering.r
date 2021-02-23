rm(list = ls())
### Libraries ###
library(data.table)
library(tibble)
library(tidyverse)
library(janitor)
library(mclust)
library(gplots)

### Parameters ###
Rep <- 'C:/Users/alexi/OneDrive/Documents/DocumentsPersonnels/SynchroDropbox/Dropbox/IODAA2020/Agro/ProjetFilRouge/'
GenusFile <- 'RemodelingData/GenusWithAgeFull.csv'
SpeciesFile <- 'RemodelingData/SpeciesWithAgeFull.csv'

### Reading data ###
GenusData <- fread(paste0(Rep,GenusFile)) %>% 
  as_tibble()
SpeciesData <- fread(paste0(Rep,SpeciesFile)) %>% 
  as_tibble()

### MClust ###
GenusMclust <- Mclust(data = GenusData[,-c(1,2,3)], G =  2:50)
GenusClassif <- bind_cols(Cluster = GenusMclust$classification, GenusData) 

pdf(paste0(Rep,'ClusteringViaModeleMixteHisto.pdf'))

ForHist <- GenusClassif %>% 
  filter(Cluster == 1)
#heatmap.2(x = ForHeatmap1, trace = 'none', main  = 'Group 0', labRow = GenusClassif %>% filter(Cluster == 1) %>% .$Age)
ggplot(ForHist,
       aes(x=Age, fill = Gender)) + 
  ggtitle("Group 0") +
  geom_histogram(binwidth = 1)

GenusClassif <- GenusClassif %>% 
  filter(Cluster == 2)
GenusMclust <- Mclust(data = GenusClassif[,-c(1,2,3,4)], G =  2:50)
GenusClassif <- bind_cols(Cluster = GenusMclust$classification, GenusClassif[,-1])

for(i in 1:GenusMclust$G){
  ForHist <- GenusClassif %>% 
    filter(Cluster == i) 
  # if(nrow(ForHeatmap) > 1){
  #   heatmap.2(x = ForHeatmap %>% 
  #               .[,-c(1,2,3,4)] %>% 
  #               as.matrix(), trace = 'none', main  = paste0('Group ',i), labRow = ForHeatmap$Age)
  # }
  p <- ggplot(ForHist,
         aes(x=Age, fill = Gender)) + 
    ggtitle(paste0("Group ",i)) +
    geom_histogram(binwidth = 1)
  print(p)
}

dev.off()


####################################################

















