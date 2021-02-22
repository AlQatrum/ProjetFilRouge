# ProjetFilRouge

Les données, ayant été fournies par les chercheurs les ayant produites pour l'article
Age-related changes in gut microbiota composition from newborn to centenarian: a cross-selectional study, Odamaki et al. BMC Microbiology 2016,
ne peuvent être disponible publiquement sur internet.
Voici un résumé des formats des tables utilisées dans les différents codes de ce git.


table individus :
age_gendre_jap.csv
453 x 4
Subject - age - gender - classe_age


tables des genres :
genus_level2.csv
102 x 454
genus x individus

table_age_2_sex_genre.csv
453 x 105
Subject - age - genre - genus(x102)


tables des espèces :
species_level_table.csv
70 x 454
species x individus

table_age_1_sex_espece_diversite_VF.csv
453 x 74
Subject - age - age2 (classe) - gender - species(x70)


tables diversité :
diversity_genus_evenness_shannon.csv
453 x 3
sample - evenness_index - shannon_index

Specieslevel_diversity.csv
453 x 3
sample - evenness_index - shannon_index
