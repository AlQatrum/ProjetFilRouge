#bibliotheque utiles
library(FactoMineR)
library(readxl)
library(factoextra)


#prep des donn�es

#importation des donn�es
data = read_excel("C:\\Users\\Hello\\Desktop\\IODAA\\fil_rouge\\to_AIRE.xlsx") 
data=t(data)
rownames(data)=data$index

#importation de l'�ge
age=read_excel("C:\\Users\\Hello\\Desktop\\IODAA\\fil_rouge\\age_and_gender_jap.xlsx")


#transformation en facteur pour l'acm
data=cbind(data, age$age)
dataprim <- lapply(data, as.factor)


#calcul de l'acm
res.mca <- MCA(data, graph=TRUE)




#visualisation

#graphe du pourcentage de variance expliqu�e
eig.val <- res.mca$eig
barplot(eig.val[, 2], 
        names.arg = 1:nrow(eig.val), 
        main = "Variances Explained by Dimensions (%)",
        xlab = "Principal Dimensions",
        ylab = "Percentage of variances",
        col ="steelblue")

#graphe des individus
fviz_mca_biplot (res.mca, repel = TRUE, 
                 ggtheme = theme_minimal())

#graphe des variables
fviz_mca_var (res.mca, choice = "mca.cor",
              repel = TRUE, 
              ggtheme = theme_minimal ())

