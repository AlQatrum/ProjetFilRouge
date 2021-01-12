########################################## Description ##############################################
# This scripts aims to write new data files easier to handle. Each file generated possess one row   #
# by individuals of the study. For each individual, its 'name', gut microbiota composition ans      #
# its age (target variable) are present both for genus and species level. The script also split     #
# randomly the dataset in 3 : one part for training, the second part for testing, and the third for #
# validation. The script also creates the same datasets with age classes.                           #
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

AgeClasses <- list(10,80)

### Reading data files ###
### For the ages
Ages <- fread(paste0(Rep,'Data/InitialData/AgeAndSex.txt'), header = TRUE) %>% 
  as_tibble() 
names(Ages) <- c('Subject', 'Age', 'Gender')
Ages[["Age"]] <- Ages[["Age"]] %>% 
  gsub(pattern = ',',replacement = '.',x = .) %>% 
  as.double()

### For the microbiota (genus level)
Genus <- fread(paste0(Rep,'Data/InitialData/genus_level2.csv')) %>% 
  t() %>%
  as.data.frame() %>% 
  row_to_names(row_number = 1) %>% 
  rownames_to_column('Subject') %>% 
  as_tibble()

### For the microbiota (species level)
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

### Joining by individuals to have the age in the same data table than the microbiota ###
GenusWithAge <- full_join(Ages, Genus)
SpeciesWithAge <- full_join(Ages, Species)

### Dividing into age classes ###
GenusWithAgeClasses <- GenusWithAge %>% 
  mutate(AgeClass = cut(Age, breaks = AgeClasses))
SpeciesWithAgeClasses <- SpeciesWithAge %>% 
  mutate(AgeClass = cut(Age, breaks = AgeClasses))

### Sampling to create training, test and validation data ###
ToGet <- ProportionForTesting/ProportionForTraining # Getting the proportion of row to choose for testing

### Without age classes
#### Genus level
GenusLearning <- GenusWithAge[sample(nrow(GenusWithAge), ProportionForTraining*nrow(GenusWithAge)),]
Remaining <- GenusWithAge[!(GenusWithAge$Subject %in% GenusLearning$Subject),]
GenusTest <- Remaining[sample(nrow(Remaining), ToGet*nrow(Remaining)),]
GenusValidation <- Remaining[!(Remaining$Subject %in% GenusTest$Subject),]

#### Species level
SpeciesLearning <- SpeciesWithAge[sample(nrow(SpeciesWithAge), ProportionForTraining*nrow(SpeciesWithAge)),]
Remaining <- SpeciesWithAge[!(SpeciesWithAge$Subject %in% SpeciesLearning$Subject),]
SpeciesTest <- Remaining[sample(nrow(Remaining), ToGet*nrow(Remaining)),]
SpeciesValidation <- Remaining[!(Remaining$Subject %in% SpeciesTest$Subject),]

### With age classes
#### Genus level
GenusAgeClassesLearning <- GenusWithAgeClasses[sample(nrow(GenusWithAgeClasses), ProportionForTraining*nrow(GenusWithAgeClasses)),]
Remaining <- GenusWithAgeClasses[!(GenusWithAgeClasses$Subject %in% GenusAgeClassesLearning$Subject),]
GenusAgeClassesTest <- Remaining[sample(nrow(Remaining), ToGet*nrow(Remaining)),]
GenusAgeClassesValidation <- Remaining[!(Remaining$Subject %in% GenusAgeClassesTest$Subject),]

#### Species level
SpeciesAgeClassesLearning <- SpeciesWithAgeClasses[sample(nrow(SpeciesWithAgeClasses), ProportionForTraining*nrow(SpeciesWithAgeClasses)),]
Remaining <- SpeciesWithAgeClasses[!(SpeciesWithAgeClasses$Subject %in% SpeciesAgeClassesLearning$Subject),]
SpeciesAgeClassesTest <- Remaining[sample(nrow(Remaining), ToGet*nrow(Remaining)),]
SpeciesAgeClassesValidation <- Remaining[!(Remaining$Subject %in% SpeciesAgeClassesTest$Subject),]

### Writing files on the drive
if(!dir.exists(paste0(Rep,'Data/WorkData'))){
  dir.create(paste0(Rep,'Data/WorkData'))
}
if(!dir.exists(paste0(Rep,'Data/WorkData/WithoutAgeClasses'))){
  dir.create(paste0(Rep,'Data/WorkData/WithoutAgeClasses'))
}
if(!dir.exists(paste0(Rep,'Data/WorkData/WithAgeClasses'))){
  dir.create(paste0(Rep,'Data/WorkData/WithAgeClasses'))
}

### Without age classes
#### Genus level
fwrite(GenusWithAge, file = paste0(Rep,'Data/WorkData/WithoutAgeClasses/GenusWithAgeFull.csv'))
fwrite(GenusLearning, file = paste0(Rep,'Data/WorkData/WithoutAgeClasses/GenusLearning.csv'))
fwrite(GenusTest, file = paste0(Rep,'Data/WorkData/WithoutAgeClasses/GenusTest.csv'))
fwrite(GenusValidation, file = paste0(Rep,'Data/WorkData/WithoutAgeClasses/GenusValidation.csv'))

#### Species level
fwrite(SpeciesWithAge, file = paste0(Rep,'Data/WorkData/WithoutAgeClasses/SpeciesWithAgeFull.csv'))
fwrite(SpeciesLearning, file = paste0(Rep,'Data/WorkData/WithoutAgeClasses/SpeciesLearning.csv'))
fwrite(SpeciesTest, file = paste0(Rep,'Data/WorkData/WithoutAgeClasses/SpeciesTest.csv'))
fwrite(SpeciesValidation, file = paste0(Rep,'Data/WorkData/WithoutAgeClasses/SpeciesValidation.csv'))

### With age classes
#### Genus level
fwrite(GenusWithAgeClasses, file = paste0(Rep,'Data/WorkData/WithAgeClasses/GenusWithAgeFull.csv'))
fwrite(GenusAgeClassesLearning, file = paste0(Rep,'Data/WorkData/WithAgeClasses/GenusLearning.csv'))
fwrite(GenusAgeClassesTest, file = paste0(Rep,'Data/WorkData/WithAgeClasses/GenusTest.csv'))
fwrite(GenusAgeClassesValidation, file = paste0(Rep,'Data/WorkData/WithAgeClasses/GenusValidation.csv'))

#### Species level
fwrite(SpeciesWithAgeClasses, file = paste0(Rep,'Data/WorkData/WithAgeClasses/SpeciesWithAgeFull.csv'))
fwrite(SpeciesAgeClassesLearning, file = paste0(Rep,'Data/WorkData/WithAgeClasses/SpeciesLearning.csv'))
fwrite(SpeciesAgeClassesTest, file = paste0(Rep,'Data/WorkData/WithAgeClasses/SpeciesTest.csv'))
fwrite(SpeciesAgeClassesValidation, file = paste0(Rep,'Data/WorkData/WithAgeClasses/SpeciesValidation.csv'))