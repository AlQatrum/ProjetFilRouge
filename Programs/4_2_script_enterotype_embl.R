#bibliotheque utiles
library('cluster')
library('clusterSim')
library('ade4')
library(readxl)

#prep des données

#importation des données
data = read_excel("C:\\Users\\Hello\\Desktop\\IODAA\\fil_rouge\\to_AIRE.xlsx") 
#data=t(data)
rownames(data)= data[,1]$index
data[,1]=NULL



#Jensen-Shannon divergence
JSD<- function(x,y) sqrt(0.5 * KLD(x, (x+y)/2) + 0.5 * KLD(y, (x+y)/2))
#Kullback-Leibler divergence
KLD <- function(x,y) sum(x * log(x/y))


#calcul de la matrice de distance
dist.JSD <- function(inMatrix, pseudocount=0.000001, ...) {
  #définition des distances
  KLD <- function(x,y) sum(x *log(x/y))
  JSD<- function(x,y) sqrt(0.5 * KLD(x, (x+y)/2) + 0.5 * KLD(y, (x+y)/2))
  
  #creation de la matrice de resultats
  matrixColSize <- length(colnames(inMatrix))
  matrixRowSize <- length(rownames(inMatrix))
  colnames <- colnames(inMatrix)
  resultsMatrix <- matrix(0, matrixColSize, matrixColSize)
  
  inMatrix = apply(inMatrix,1:2,function(x) ifelse (x==0,pseudocount,x))

  for(i in 1:matrixColSize) {
    for(j in 1:matrixColSize) { 
      #calcul de la distance
      resultsMatrix[i,j]=JSD(as.vector(inMatrix[,i]),
                             as.vector(inMatrix[,j]))
    }
  }
  colnames -> colnames(resultsMatrix) -> rownames(resultsMatrix)
  #on accole l'attribut dist a la matrice donnée en argument
  as.dist(resultsMatrix)->resultsMatrix
  attr(resultsMatrix, "method") <- "dist"
  return(resultsMatrix) 
}


#calcul de la matrice de distance
data.dist=dist.JSD(data)






#Utilisation du Partitionning arround medioids (une variante du kmeans)

pam.clustering=function(x,k) { # x is a distance matrix and k the number of clusters
  require(cluster)
  cluster = as.vector(pam(as.dist(x), k, diss=TRUE)$clustering)
  return(cluster)
}

#clustering pour k = 3
data.cluster=pam.clustering(data.dist, k=3) #ATTENTION LES NUMEROS DE CLUSTERS SONT DONNES ALEATOIREMENT






#Calcul du nombre de cluster optimaux

require(clusterSim)
#nclusters = index.G1(t(data), data.cluster, d = data.dist, centrotypes = "medoids")

  
#calcul de l'induice CH pour tout les nombre de cluster possibles  
nclusters=NULL

for (k in 1:20) { 
  if (k==1) {
    nclusters[k]=NA 
  } else {
    data.cluster_temp=pam.clustering(data.dist, k)
    nclusters[k]=index.G1(t(data),data.cluster_temp,  d = data.dist,
                          centrotypes = "medoids")
  }
}

#plot du ch index
plot(nclusters, main="CH-index pour différents clustering" ,type="h", xlab="k clusters", ylab="CH index")
df_cluster=data.frame(nclusters, seq(1, length(nclusters)))
colnames(df_cluster)=c("CH_index","n_cluster")
f <- ggplot(df_cluster, aes(x = n_cluster, y = CH_index)) + geom_col()+
  labs(title="CH index par nombre de cluster",x="nombre de cluster", y = "CH index")
f


#validation des clusters
obs.silhouette=mean(silhouette(data.cluster, data.dist)[,3])




#on enlève les genres faiblement présent (aka abondance trop faible)
#on ne s'en ai pas servi ici
noise.removal <- function(dataframe, percent=0.01, top=NULL){
  dataframe->Matrix
  bigones <- rowSums(Matrix)*100/(sum(rowSums(Matrix))) > percent 
  Matrix_1 <- Matrix[bigones,]
  print(percent)
  return(Matrix_1)
}


data.denoized=noise.removal(data, percent=0.01)




#PLOT
obs.pca=dudi.pca(data.frame(t(data)), scannf=F, nf=10)
obs.bet=bca(obs.pca, fac=as.factor(data.cluster), scannf=F, nf=k-1) 

s.class(obs.bet$ls, fac=as.factor(data.cluster), grid=F, sub="BCA de la distance entre les individus clusterisée")
s.class(obs.bet$ls, fac=as.factor(data.cluster), grid=F, cell=0, cstar=0, col=c(4,2,3))
s.label(obs.bet$ls)

obs.pcoa=dudi.pco(data.dist, scannf=F, nf=3)
s.class(obs.pcoa$li, fac=as.factor(data.cluster), grid=F, sub="PCoA de la distance entre les individus clusterisée")
s.class(obs.pcoa$li, fac=as.factor(data.cluster), grid=F,sub="PCoA de la distance entre les individus clusterisée", cell=0, cstar=0, col=c(3,2,4))
s.label(obs.pcoa$li)





# age des differents clusters

#prepartion des données supplémentaires
age = read.table("C:\\Users\\Hello\\Desktop\\IODAA\\fil_rouge\\to_age.csv", sep=",")
longevite=as.vector(age$V2)
longevite=longevite[2:length(longevite)]

#stockage des ages dans des vecteurs
age_1=c()
age_2=c()
age_3=c()
age_groupe= data.frame()  
for (k in 1:length(data.cluster)) {
  if (data.cluster[k]==1){  
    age_1=c(age_1, longevite[k])
  }
  else if (data.cluster[k]==2){
    age_2=c(age_2, longevite[k])
  }
  else if (data.cluster[k]==3){
    age_3=c(age_3, longevite[k])
  }
}

#traitement des résultats
age_1=type.convert(age_1, dec = ",")
age_2=type.convert(age_2, dec = ",")
age_3=type.convert(age_3, dec = ",")
age_groupe= data.frame(c(age_1,age_2,age_3), c(rep(1,length(age_1) ), rep(2, length(age_2) ), rep(3, length(age_3)) ) ) 
colnames(age_groupe)=c("age","cluster")


#plot des mins

#
min=c(min(age_1), min(age_2), min(age_3))
barplot(min, main="age minimum par Enterotype",
        xlab="Enterotype", ylab="Effectif")



#boxplot des ages dans les clusters
library(ggplot2)

age_groupe$cluster=as.factor(age_groupe$cluster)
p <- ggplot(age_groupe, aes(x=cluster, y=age, fill=cluster)) + labs(title="Age par Entérotype")+geom_violin(trim=FALSE)
  #geom_jitter(shape=16, position=position_jitter(0.2))
p




#test de student non appariés entre les clusters
t.test(age_1, age_2) #non significatif (les numeros de clusters étant attribué au hasard, cela peut varier)
t.test(age_1, age_3) #significatif
t.test(age_2, age_3) #significatif




#determiner quel cluster pour quel genre

#ensemble des noms de lignes contenant les genres interessants
genre_Prevotella = c("k__Bacteria|p__Bacteroidetes|c__Bacteroidia|o__Bacteroidales|f__Prevotellaceae|g__Prevotella",
                     "k__Bacteria|p__Bacteroidetes|c__Bacteroidia|o__Bacteroidales|f__[Paraprevotellaceae]|g__[Prevotella]") 
genre_Bacteroides = c("k__Bacteria|p__Bacteroidetes|c__Bacteroidia|o__Bacteroidales|f__Bacteroidaceae|g__Bacteroides")
                      #"k__Bacteria|p__Firmicutes|c__Clostridia|o__Clostridiales|f__Lachnospiraceae|g__Roseburia")
genre_Ruminococcus =c("k__Bacteria|p__Firmicutes|c__Clostridia|o__Clostridiales|f__Lachnospiraceae|g__[Ruminococcus]",
                      "k__Bacteria|p__Firmicutes|c__Clostridia|o__Clostridiales|f__Ruminococcaceae|g__Ruminococcus")
                      #"k__Bacteria|p__Verrucomicrobia|c__Verrucomicrobiae|o__Verrucomicrobiales|f__Verrucomicrobiaceae|g__Akkermansia",
                      #"k__Bacteria|p__Bacteroidetes|c__Bacteroidia|o__Bacteroidales|f__Rikenellaceae|g__Alistipes")
  
  

#vecteur contenant la somme des abondances par genre et par cluster
sum_prevotella = c(0,0,0)
sum_bacteroides = c(0,0,0)
sum_ruminococcus = c(0,0,0)

#on remplis ces vecteurs avec la somme des abondances pour chaque genre par individus
for (k in 1:length(data.cluster)) {
  if (data.cluster[k]==1){  
    sum_prevotella[1]=sum_prevotella[1]  + sum(data[genre_Prevotella,k])
    sum_bacteroides[1]=sum_bacteroides[1]  + sum(data[genre_Bacteroides,k])
    sum_ruminococcus[1]=sum_ruminococcus[1]  + sum(data[genre_Ruminococcus,k])
  } 
  else if (data.cluster[k]==2){  
    sum_prevotella[2]=sum_prevotella[2]  + sum(data[genre_Prevotella,k])
    sum_bacteroides[2]=sum_bacteroides[2]  + sum(data[genre_Bacteroides,k])
    sum_ruminococcus[2]=sum_ruminococcus[2]  + sum(data[genre_Ruminococcus,k])
  }
  else if (data.cluster[k]==3){  
    sum_prevotella[3]=sum_prevotella[3]  + sum(data[genre_Prevotella,k])
    sum_bacteroides[3]=sum_bacteroides[3]  + sum(data[genre_Bacteroides,k])
    sum_ruminococcus[3]=sum_ruminococcus[3]  + sum(data[genre_Ruminococcus,k])
  }
}

#on fait l'abondance moyenne par genre
sum_prevotella[1]=sum_prevotella[1]/length(age_1)
sum_bacteroides[1]=sum_bacteroides[1]/length(age_1)
sum_ruminococcus[1]=sum_ruminococcus[1]/length(age_1)

sum_prevotella[2]=sum_prevotella[2]/length(age_2)
sum_bacteroides[2]=sum_bacteroides[2]/length(age_2)
sum_ruminococcus[2]=sum_ruminococcus[2]/length(age_2)

sum_prevotella[3]=sum_prevotella[3]/length(age_3)
sum_bacteroides[3]=sum_bacteroides[3]/length(age_3)
sum_ruminococcus[3]=sum_ruminococcus[3]/length(age_3)

sum_prevotella
sum_bacteroides
sum_ruminococcus


#a priori (suceptible de changer selon la run)
#le cluster Prevotella est le 1
#le cluster Bacteroides est le 2
#le clsuter Ruminococcus est le 3 




#plot du statut des bébés selon l'enterotypes

#importation des données
wean = read.csv("C:\\Users\\Hello\\Desktop\\IODAA\\fil_rouge\\table_age_1_sex_genre_diversite.csv") 
wean = data.frame(wean$age, wean$X,  data.cluster)

#remplissage des vecteurs
weaned=c()
preweaning=c()
weaning=c()
for(k in c(1:length(wean$wean.age) )){
  if (wean$wean.age[k]=="Weaned - 3"){
     weaned=c(weaned, wean$data.cluster[k]) 
  }else if(wean$wean.age[k]== "Preweaning"){
    preweaning=c(preweaning, wean$data.cluster[k])
  }else if(wean$wean.age[k]=="Weaning"){
    weaning=c(weaning, wean$data.cluster[k])
  }
} 

#création d'un dataframe pour faciliter les plots
col_weaned=c( rep("weaned", 3),
             rep("preweaning",3),
             rep("weaning",3 ))
wean_number_per_cluster= c(length(weaned[weaned==1]),length(weaned[weaned==2]),length(weaned[weaned==3]),
                        length(preweaning[preweaning==1]),length(preweaning[preweaning==2]),length(preweaning[preweaning==3]),
                        length(weaning[weaning==1]),length(weaning[weaning==2]),length(weaning[weaning==3]))
wean_cluster=c(1,2,3,1,2,3,1,2,3)
weaned_graph= data.frame(col_weaned,wean_number_per_cluster, wean_cluster)
colnames(weaned_graph)=c("état du sevrage", "nombre de bébés", "entérotype")


#plot de l'état du sevrage en fonction du cluster
ggplot(weaned_graph, aes(x=entérotype, y=`nombre de bébés`, fill=`état du sevrage`)) + 
  geom_bar(stat="identity") + labs(title="Nombre de bébés par catégories par entérotypes")

