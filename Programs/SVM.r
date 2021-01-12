########################################## Description ##############################################
# This script aims to create a SVM separating young + old from the others.                          #
#####################################################################################################


rm(list = ls())
### Libraries ###
library(data.table)
library(tibble)
library(tidyverse)
library(e1071) # For smv

### Parameters ###
Rep <- 'C:/Users/alexi/OneDrive/Documents/DocumentsPersonnels/SynchroDropbox/Dropbox/IODAA2020/Agro/ProjetFilRouge/ProjetFilRouge/'
AdultAges <- '(10,80]'

### Data ###
GenusLearning <- fread(paste0(Rep,'Data/WorkData/WithAgeClasses/GenusLearning.csv'))
GenusTest <- fread(paste0(Rep,'Data/WorkData/WithAgeClasses/GenusTest.csv'))
GenusValidation <- fread(paste0(Rep,'Data/WorkData/WithAgeClasses/GenusValidation.csv'))

SpeciesLearning <- fread(paste0(Rep,'Data/WorkData/WithAgeClasses/SpeciesLearning.csv'))
SpeciesTest <- fread(paste0(Rep,'Data/WorkData/WithAgeClasses/SpeciesTest.csv'))
SpeciesValidation <- fread(paste0(Rep,'Data/WorkData/WithAgeClasses/SpeciesValidation.csv'))

### Creating only two classes ###
GenusLearning <- GenusLearning %>% 
  mutate(OldYoung = ifelse(test = AgeClass != AdultAges, yes = 1, no = -1)) %>% 
  as_tibble()
GenusTest <- GenusTest %>% 
  mutate(OldYoung = ifelse(test = AgeClass != AdultAges, yes = 1, no = -1)) %>% 
  as_tibble()
GenusValidation <- GenusValidation %>% 
  mutate(OldYoung = ifelse(test = AgeClass != AdultAges, yes = 1, no = -1)) %>% 
  as_tibble()

SpeciesLearning <- SpeciesLearning %>% 
  mutate(OldYoung = ifelse(test = AgeClass != AdultAges, yes = 1, no = -1)) %>% 
  as_tibble()
SpeciesTest <- SpeciesTest %>% 
  mutate(OldYoung = ifelse(test = AgeClass != AdultAges, yes = 1, no = -1)) %>% 
  as_tibble()
SpeciesValidation <- SpeciesValidation %>% 
  mutate(OldYoung = ifelse(test = AgeClass != AdultAges, yes = 1, no = -1)) %>% 
  as_tibble()

### Removing [ and ] in colnames ###
GenusColNames <- colnames(GenusLearning) %>% 
  map(.x = ., .f = ~str_remove_all(string = .x, pattern = "[^[:alnum:][:blank:]_]")) %>% 
  unlist()

colnames(GenusLearning) <- GenusColNames
colnames(GenusTest) <- GenusColNames
colnames(GenusValidation) <- GenusColNames

SpeciesColNames <- colnames(SpeciesLearning) %>% 
  map(.x = ., .f = ~str_remove_all(string = .x, pattern = "[^[:alnum:][:blank:]_]")) %>% 
  unlist()

colnames(SpeciesLearning) <- SpeciesColNames
colnames(SpeciesTest) <- SpeciesColNames
colnames(SpeciesValidation) <- SpeciesColNames

### Modelising SVM ###
GenusFormula <- colnames(GenusLearning)[4:105] %>% # Taking only columns of interest
  paste(collapse = '+') 
SpeciesFormula <- colnames(SpeciesLearning)[4:73] %>% # Taking only columns of interest
  paste(collapse = '+')

### Genus modelising
GenusSVM <- svm(OldYoung ~ eval(parse(text=GenusFormula)), data = GenusLearning, kernel = 'linear', scale = TRUE, probability = TRUE)
print(GenusSVM)
plot(GenusSVM, GenusLearning)

GenusTestPrediction <- predict(GenusSVM, GenusTest) %>% 
  cbind(GenusTest, .) %>% 
  select('OldYoung', '.')




### Species modelising
SpeciesSVM <- svm(OldYoung ~ eval(parse(text=SpeciesFormula)), data = SpeciesLearning, kernel = 'linear', scale = TRUE, probability = TRUE)
print(SpeciesSVM)
plot(SpeciesSVM, SpeciesLearning)

SpeciesTestPrediction <- predict(SpeciesSVM, SpeciesTest) %>% 
  cbind(SpeciesTest, .) %>% 
  select('OldYoung', '.')