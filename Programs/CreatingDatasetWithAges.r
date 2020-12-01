############################################ Description ############################################
# This scripts aims to write new data files easier to handle. Each file generated possess one row   #
# by individuals of the study. For each individual, its 'name', gut microbiota composition ans      #
# its age (target variable) are present both for genus and species level. The script also split     #
# randomly the dataset in 3 : one part for training, the second part for testing, and the third for #
# validation.                                                                                       #
#####################################################################################################


rm(list = ls())
### Libraries ###
library(data.table)
library(tibble)
library(tidyverse)
library(janitor)

### Parameters ###
Rep <- 'C:/Users/alexi/OneDrive/Documents/DocumentsPersonnels/SynchroDropbox/Dropbox/IODAA2020/Agro/ProjetFilRouge/ProjetFilRouge/'

ProportionForTraining <- 2/3
ProportionForTesting <- 1/6  # The rest of the dataset will be used for validation

AgeClasses <- list(3,10,20,30,40,50,60,70,80,90,100)

# Reading data files ###
Ages <- fread(paste0(Rep,'Data/InitialData/AgeAndSex.txt'), header = TRUE) %>% 
  as_tibble() 
names(Ages) <- c('Subject', 'Age', 'Gender')
Ages[["Age"]] <- Ages[["Age"]] %>% 
  gsub(pattern = ',',replacement = '.',x = .) %>% 
  as.double()

Genus <- fread(paste0(Rep,'Data/InitialData/genus_level2.csv')) %>% 
  t() %>%
  as.data.frame() %>% 
  row_to_names(row_number = 1) %>% 
  rownames_to_column('Subject') %>% 
  as_tibble()

SpeciesNames <- fread(paste0(Rep,'Data/InitialData/species_level_table.csv')) %>% 
  as_tibble() %>% 
  select('index')

Species <- fread(paste0(Rep,'Data/InitialData/species_level_table.csv')) %>% 
  as_tibble() %>% 
  select(contains('Jap')) %>% 
  t() %>%
  as.data.frame() %>% 
  rownames_to_column('Subject') %>% 
  as_tibble()
colnames(Species) <- c('Subject', SpeciesNames[['index']])

GenusWithAge <- full_join(Ages, Genus)
SpeciesWithAge <- full_join(Ages, Species)



#GenusWithAgeClasses <- GenusWithAge %>% 
#  mutate(AgeClass = )

ToGet <- ProportionForTesting/ProportionForTraining # Getting the proportion of row to choose for testing

GenusLearning <- GenusWithAge[sample(nrow(GenusWithAge), ProportionForTraining*nrow(GenusWithAge)),]
Remaining <- GenusWithAge[!(GenusWithAge$Subject %in% GenusLearning$Subject),]
GenusTest <- Remaining[sample(nrow(Remaining), ToGet*nrow(Remaining)),]
GenusValidation <- Remaining[!(Remaining$Subject %in% GenusTest$Subject),]

SpeciesLearning <- SpeciesWithAge[sample(nrow(SpeciesWithAge), ProportionForTraining*nrow(SpeciesWithAge)),]
Remaining <- SpeciesWithAge[!(SpeciesWithAge$Subject %in% SpeciesLearning$Subject),]
SpeciesTest <- Remaining[sample(nrow(Remaining), ToGet*nrow(Remaining)),]
SpeciesValidation <- Remaining[!(Remaining$Subject %in% SpeciesTest$Subject),]

if(!dir.exists(paste0(Rep,'Data/WorkData'))){
  dir.create(paste0(Rep,'Data/WorkData'))
}

fwrite(GenusWithAge, file = paste0(Rep,'Data/WorkData/GenusWithAgeFull.csv'))
fwrite(GenusLearning, file = paste0(Rep,'Data/WorkData/GenusLearning.csv'))
fwrite(GenusTest, file = paste0(Rep,'Data/WorkData/GenusTest.csv'))
fwrite(GenusValidation, file = paste0(Rep,'Data/WorkData/GenusValidation.csv'))

fwrite(SpeciesWithAge, file = paste0(Rep,'Data/WorkData/SpeciesWithAgeFull.csv'))
fwrite(SpeciesLearning, file = paste0(Rep,'Data/WorkData/SpeciesLearning.csv'))
fwrite(SpeciesTest, file = paste0(Rep,'Data/WorkData/SpeciesTest.csv'))
fwrite(SpeciesValidation, file = paste0(Rep,'Data/WorkData/SpeciesValidation.csv'))
