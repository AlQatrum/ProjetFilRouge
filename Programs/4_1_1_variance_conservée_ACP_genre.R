
library("readxl")
library("FactoMineR")
library("factoextra")

#importation des données
data = read_excel("C:\\Users\\Hello\\Desktop\\IODAA\\fil_rouge\\to_AIRE.xlsx") 
#data=t(data)
rownames(data)= data[,1]$index
data[,1]=NULL
data= t(data)

#calcul de l'ACP
res.pca <- PCA(data, graph = TRUE)
eig.val <- get_eigenvalue(res.pca)
#visualisation de la part de variance conservé
fviz_eig(res.pca, addlabels = TRUE, ylab = "Part de la variance conservée", 
         xlab= "Composantes principales", linecolor = "black",
         main = "Part de la variance conservée par composante principale")
fviz_pca_var(res.pca, col.var = "black")
