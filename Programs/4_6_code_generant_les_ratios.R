library("readxl")
library("FactoMineR")
library("factoextra")

#preparation des données
data_age = read_excel("C:/Users/Hello/Desktop/IODAA/fil_rouge/age_and_gender_jap.xlsx")
data_jap = read_excel("C:/Users/Hello/Desktop/IODAA/fil_rouge/to_AIRE.xlsx")
#traitent
rownames(data_jap)=data_jap$index
data_jap$index=NULL

#ratio=mapply('/', data_jap, data_jap)



#calcul des ratios
df=data.frame()
l=c()
for (i in rownames(data_jap)){
  for (j in rownames(data_jap)){
    #on ajoute ligne par ligne les ratios (*1000 pour limiter les Inf)
    df=rbind(df, (data_jap[i,]*1000)/(data_jap[j,]*1000))
    #nom des ratios
    l=c(l,paste(toString(i),"_div_",toString(j)))
  }}
rownames(df)=l

write.csv(df, "C:/Users/Hello/Desktop/IODAA/fil_rouge/ratio_genre.csv")





#fonction qui permet de dire si un point du dataframe est un NA
is.nan.data.frame <- function(x)
  do.call(cbind, lapply(x, is.nan))

#les point étant des NA ou des Inf sont remplacé par des 0
df[is.nan(df)] <- 0
df[df=="Inf"] <- 0
#on supprime les lignes dont la somme vaut 0
df=df[rowSums(df[])>0,]

#on écrit différents fichiers
write.csv(df, "C:/Users/Hello/Desktop/IODAA/fil_rouge/ratio_genre_trié.csv")
df=rbind(df ,data$age)
rownames(df)[rownames(df) == "11364"]="age"
write.csv(df, "C:/Users/Hello/Desktop/IODAA/fil_rouge/ratio_genre_trié_avec_age.csv")


df=t(df)
#correction de problèmes résiduels
df[,"k__Bacteria|__|__|__|__|__ _div_ k__Bacteria|__|__|__|__|__"] <- as.numeric(df[,"k__Bacteria|__|__|__|__|__ _div_ k__Bacteria|__|__|__|__|__"]) 
df[,"k__Bacteria|__|__|__|__|__ _div_ k__Bacteria|p__Actinobacteria|c__Actinobacteria|o__Actinomycetales|f__Actinomycetaceae|g__Actinomyces"] <- as.numeric(df[,"k__Bacteria|__|__|__|__|__ _div_ k__Bacteria|p__Actinobacteria|c__Actinobacteria|o__Actinomycetales|f__Actinomycetaceae|g__Actinomyces"])
df[,"k__Bacteria|__|__|__|__|__ _div_ k__Bacteria|p__Actinobacteria|c__Actinobacteria|o__Actinomycetales|f__Propionibacteriaceae|g__Propionibacterium"] <- as.numeric(df[,"k__Bacteria|__|__|__|__|__ _div_ k__Bacteria|p__Actinobacteria|c__Actinobacteria|o__Actinomycetales|f__Propionibacteriaceae|g__Propionibacterium"])
df[,"k__Bacteria|__|__|__|__|__ _div_ k__Bacteria|p__Actinobacteria|c__Actinobacteria|o__Bifidobacteriales|f__Bifidobacteriaceae|g__"] <- as.numeric(df[, "k__Bacteria|__|__|__|__|__ _div_ k__Bacteria|p__Actinobacteria|c__Actinobacteria|o__Bifidobacteriales|f__Bifidobacteriaceae|g__" ])                                                                                                                                 
df[, "k__Bacteria|__|__|__|__|__ _div_ k__Bacteria|p__Actinobacteria|c__Actinobacteria|o__Bifidobacteriales|f__Bifidobacteriaceae|g__Bifidobacterium"]<- as.numeric(df[, "k__Bacteria|__|__|__|__|__ _div_ k__Bacteria|p__Actinobacteria|c__Actinobacteria|o__Bifidobacteriales|f__Bifidobacteriaceae|g__Bifidobacterium"])
df[, "k__Bacteria|__|__|__|__|__ _div_ k__Bacteria|p__Actinobacteria|c__Coriobacteriia|o__Coriobacteri"]<- as.numeric(df[, "k__Bacteria|__|__|__|__|__ _div_ k__Bacteria|p__Actinobacteria|c__Coriobacteriia|o__Coriobacteri"])                                                                                                                                                     
#dernier fichier       
df =read.table("C:/Users/Hello/Desktop/IODAA/fil_rouge/ratio_genre_trié_avec_age_transpose.csv")


