rm(list = ls())
### Libraries ###
library(data.table)
library(tibble)
library(tidyverse)
library(janitor)

### Parameters ###
Rep <- 'C:/Users/alexi/OneDrive/Documents/DocumentsPersonnels/SynchroDropbox/Dropbox/IODAA2020/Agro/ProjetFilRouge/RemodelingData/'

# Reading data files ###
Ages <- fread(paste0(Rep,'AgeAndSex.txt'), header = TRUE) %>% 
  as_tibble() 
names(Ages) <- c('Subject', 'Age', 'Gender')
Ages[["Age"]] <- Ages[["Age"]] %>% 
  gsub(pattern = ',',replacement = '.',x = .) %>% 
  as.double()

Genus <- fread(paste0(Rep,'genus_level2.csv')) %>% 
  t() %>%
  as.data.frame() %>% 
  row_to_names(row_number = 1) %>% 
  rownames_to_column('Subject') %>% 
  as_tibble()

SpeciesNames <- fread(paste0(Rep,'species_level_table.csv')) %>% 
  as_tibble() %>% 
  select('index')

Species <- fread(paste0(Rep,'species_level_table.csv')) %>% 
  as_tibble() %>% 
  select(contains('Jap')) %>% 
  t() %>%
  as.data.frame() %>% 
  rownames_to_column('Subject') %>% 
  as_tibble()
colnames(Species) <- c('Subject', SpeciesNames[['index']])

GenusWithAge <- full_join(Ages, Genus)
SpeciesWithAge <- full_join(Ages, Species)

AgeClasses <- list(3,10,20,30,40,50,60,70,80,90,100)

GenusWithAgeClasses <- GenusWithAge %>% 
  mutate(AgeClass = )

GenusLearning <- GenusWithAge[sample(nrow(GenusWithAge), 2*nrow(GenusWithAge)/3),]
Remaining <- GenusWithAge[!(GenusWithAge$Subject %in% GenusLearning$Subject),]
GenusTest <- Remaining[sample(nrow(Remaining), nrow(Remaining)/2),]
GenusValidation <- Remaining[!(Remaining$Subject %in% GenusTest$Subject),]

SpeciesLearning <- SpeciesWithAge[sample(nrow(SpeciesWithAge), 2*nrow(SpeciesWithAge)/3),]
Remaining <- SpeciesWithAge[!(SpeciesWithAge$Subject %in% SpeciesLearning$Subject),]
SpeciesTest <- Remaining[sample(nrow(Remaining), nrow(Remaining)/2),]
SpeciesValidation <- Remaining[!(Remaining$Subject %in% SpeciesTest$Subject),]

fwrite(GenusWithAge, file = paste0(Rep,'GenusWithAgeFull.csv'))
fwrite(GenusLearning, file = paste0(Rep,'GenusLearning.csv'))
fwrite(GenusTest, file = paste0(Rep,'GenusTest.csv'))
fwrite(GenusValidation, file = paste0(Rep,'GenusValidation.csv'))

fwrite(SpeciesWithAge, file = paste0(Rep,'SpeciesWithAgeFull.csv'))
fwrite(SpeciesLearning, file = paste0(Rep,'SpeciesLearning.csv'))
fwrite(SpeciesTest, file = paste0(Rep,'SpeciesTest.csv'))
fwrite(SpeciesValidation, file = paste0(Rep,'SpeciesValidation.csv'))
