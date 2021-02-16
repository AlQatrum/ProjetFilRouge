#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: annecharlottemichel
"""

import pandas as pd
import numpy as np

#table_espece = pd.read_excel('./to_AIRE_modifie.xlsx', sheet_name = 'spcecies')
table_espece = pd.read_csv('./species_level_table.csv')
#table_genre = pd.read_csv('./genus_level2.csv')
table_age_sex = pd.read_excel('./age_and_gender_jap.xlsx', sheet_name = 'Sheet1')
table_diversite = pd.read_excel('./diversity_genus_evenness_shannon.xlsx', sheet_name = 'Feuille1')
table_diversite_espece = pd.read_excel('./Specieslevel_diversity.xlsx', sheet_name = 'Feuille1')

#table_espece = table_espece.iloc[:, 6:]
#table_espece = table_espece.loc[(table_espece['Unnamed: 6'] != '__') & (table_espece['Unnamed: 6'] != 's__') ]
table_espece = table_espece.transpose()
table_espece = table_espece.reset_index()

header = table_espece.iloc[0]
header['index'] = 'Subject'

table_espece = table_espece[1:]
table_espece = table_espece.rename(columns = header)

table_age_sex = table_age_sex[['Subject \n(name of deposit data)', 'age', 
                               'gender']]
table_age_sex.columns = ['Subject', 'age', 'gender']

table_age_sex_espece_diversite = pd.merge(table_age_sex, table_espece, how = 'inner', 
                               left_on = ['Subject'], right_on = ['Subject'])

table_age_sex_espece_diversite = pd.merge(table_age_sex_espece_diversite, table_diversite_espece, how = 'inner', 
                               left_on = ['Subject'], right_on = ['sample'])
del table_age_sex_espece_diversite['sample']

# Classification age article
table_age_1_sex_espece_diversite = table_age_sex_espece_diversite.copy()
table_age_1_sex_espece_diversite['age'] = (
    np.select(
        condlist=[table_age_1_sex_espece_diversite['Subject'] == 'Japanese250',
                  table_age_1_sex_espece_diversite['Subject'] == 'Japanese273',
                  table_age_1_sex_espece_diversite['Subject'] == 'Japanese317',
                  table_age_1_sex_espece_diversite['age'] <0.5,
                  (table_age_1_sex_espece_diversite['age'] >= 0.5) &( table_age_1_sex_espece_diversite['age'] <= 1),
                  (table_age_1_sex_espece_diversite['age'] > 1) &( table_age_1_sex_espece_diversite['age'] <4),
                  (table_age_1_sex_espece_diversite['age'] >= 4) &( table_age_1_sex_espece_diversite['age'] <10),
                  (table_age_1_sex_espece_diversite['age'] >= 10) &( table_age_1_sex_espece_diversite['age'] <20),
                  (table_age_1_sex_espece_diversite['age'] >= 20) & (table_age_1_sex_espece_diversite['age'] <30),
                  (table_age_1_sex_espece_diversite['age'] >= 30) & (table_age_1_sex_espece_diversite['age'] <40),
                  (table_age_1_sex_espece_diversite['age'] >= 40) & (table_age_1_sex_espece_diversite['age'] <50),
                  (table_age_1_sex_espece_diversite['age'] >= 50) & (table_age_1_sex_espece_diversite['age'] <60),
                  (table_age_1_sex_espece_diversite['age'] >= 60) & (table_age_1_sex_espece_diversite['age'] <70),
                  (table_age_1_sex_espece_diversite['age'] >= 70) & (table_age_1_sex_espece_diversite['age'] <80),
                  (table_age_1_sex_espece_diversite['age'] >= 80) & (table_age_1_sex_espece_diversite['age'] <90),
                  (table_age_1_sex_espece_diversite['age'] >= 90) & (table_age_1_sex_espece_diversite['age'] <100),
                  table_age_1_sex_espece_diversite['age'] >=100],
        choicelist=['Weaning', 'Weaning','Preweaning', 'Preweaning', 'Weaning',
                    'Weaned - 3', '4 - 9', '10 - 19',
                    '20 - 29', '30 - 39', '40 - 49','50 - 59', '60 - 69',
                    '70 - 79', '80 - 89','90 - 99', '100 +' ]))

table_age_1_sex_espece_diversite.insert(2, 'age2', table_age_sex_espece_diversite['age'])

table_age_1_sex_espece_diversite.to_csv('table_age_1_sex_espece_diversite.csv')

# Classification decade
table_age_2_sex_espece_diversite = table_age_sex_espece_diversite.copy()
table_age_2_sex_espece_diversite['age'] = (
    np.select(
        condlist=[table_age_2_sex_espece_diversite['age'] <10,
                  (table_age_2_sex_espece_diversite['age'] >= 10) &( table_age_2_sex_espece_diversite['age'] <20),
                  (table_age_2_sex_espece_diversite['age'] >= 20) & (table_age_2_sex_espece_diversite['age'] <30),
                  (table_age_2_sex_espece_diversite['age'] >= 30) & (table_age_2_sex_espece_diversite['age'] <40),
                  (table_age_2_sex_espece_diversite['age'] >= 40) & (table_age_2_sex_espece_diversite['age'] <50),
                  (table_age_2_sex_espece_diversite['age'] >= 50) & (table_age_2_sex_espece_diversite['age'] <60),
                  (table_age_2_sex_espece_diversite['age'] >= 60) & (table_age_2_sex_espece_diversite['age'] <70),
                  (table_age_2_sex_espece_diversite['age'] >= 70) & (table_age_2_sex_espece_diversite['age'] <80),
                  (table_age_2_sex_espece_diversite['age'] >= 80) & (table_age_2_sex_espece_diversite['age'] <90),
                  (table_age_2_sex_espece_diversite['age'] >= 90) & (table_age_2_sex_espece_diversite['age'] <100),
                  table_age_2_sex_espece_diversite['age'] >=100],
        choicelist=['0 - 9', '10 - 19', '20 - 29', '30 - 39', '40 - 49',
                    '50 - 59', '60 - 69','70 - 79', '80 - 89','90 - 99', '100 - 109']))

table_age_2_sex_espece_diversite.to_csv('table_age_2_sex_espece_diversite.csv')

# Classification enfants (-20), adultes (21-69), personnes agés (+70)
table_age_3_sex_espece_diversite = table_age_sex_espece_diversite.copy()
table_age_3_sex_espece_diversite['age'] = (
    np.select(
        condlist=[table_age_3_sex_espece_diversite['age'] <= 20, table_age_3_sex_espece_diversite['age'] >= 70], 
        choicelist=['child', 'elderly'], 
        default = 'adult'))

table_age_3_sex_espece_diversite.to_csv('table_age_3_sex_espece_diversite.csv')


# table_age_4_sex_espece_diversite = table_age_sex_espece_diversite.copy()
# table_age_4_sex_espece_diversite["classe"] = pd.cut(table_age_4_sex_espece_diversite.age,
#                                [-1, 1, 80, np.inf],
#                                labels=['baby','other','elderly'])

# table_age_4_sex_espece_diversite.to_csv("table_age_4_sex_espece_diversite.csv")




table_age_7_sex_espece_diversite = table_age_sex_espece_diversite.copy()
decoupage = pd.cut(table_age_7_sex_espece_diversite.age,
                               [-1, 1, 10, 80, np.inf],
                               labels=['0-1', '1-10', '10-80', '+80'])
table_age_7_sex_espece_diversite.insert(2, 'classe', decoupage)

table_age_7_sex_espece_diversite.to_csv("table_age_7_sex_espece_diversite.csv")

table_age_8_sex_espece_diversite = table_age_sex_espece_diversite.copy()
decoupage = pd.cut(table_age_8_sex_espece_diversite.age,
                               [-1, 1, 10, 70, np.inf],
                               labels=['0-1', '1-10', '10-70', '+70'])
table_age_8_sex_espece_diversite.insert(2, 'classe', decoupage)

table_age_8_sex_espece_diversite.to_csv("table_age_8_sex_espece_diversite.csv")