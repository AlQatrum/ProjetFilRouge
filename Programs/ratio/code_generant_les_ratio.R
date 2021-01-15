library("readxl")
library("FactoMineR")
library("factoextra")

data_age = read_excel("C:/Users/Hello/Desktop/IODAA/fil_rouge/age_and_gender_jap.xlsx")
data_jap = read_excel("C:/Users/Hello/Desktop/IODAA/fil_rouge/to_AIRE.xlsx")

rownames(data_jap)=data_jap$index
data_jap$index=NULL

ratio=mapply('/', data_jap, data_jap)



df=data.frame()
l=c()
for (i in rownames(data_jap)){
  for (j in rownames(data_jap)){
    df=rbind(df, (data_jap[i,]*1000)/(data_jap[j,]*1000))
    l=c(l,paste(toString(i),"_div_",toString(j)))
  }}
rownames(df)=l

write.csv(df, "C:/Users/Hello/Desktop/IODAA/fil_rouge/ratio_genre.csv")

is.nan.data.frame <- function(x)
  do.call(cbind, lapply(x, is.nan))

df[is.nan(df)] <- 0

df[df=="Inf"] <- 0

df=df[rowSums(df[])>0,]

write.csv(df, "C:/Users/Hello/Desktop/IODAA/fil_rouge/ratio_genre_trié.csv")
df=rbind(df ,data$age)
rownames(df)[rownames(df) == "11364"]="age"
write.csv(df, "C:/Users/Hello/Desktop/IODAA/fil_rouge/ratio_genre_trié_avec_age.csv")


df=t(df)
write.csv(df, "C:/Users/Hello/Desktop/IODAA/fil_rouge/ratio_genre_trié_avec_age_transpose.csv")



df[,"k__Bacteria|__|__|__|__|__ _div_ k__Bacteria|__|__|__|__|__"] <- as.numeric(df[,"k__Bacteria|__|__|__|__|__ _div_ k__Bacteria|__|__|__|__|__"]) 
df[,"k__Bacteria|__|__|__|__|__ _div_ k__Bacteria|p__Actinobacteria|c__Actinobacteria|o__Actinomycetales|f__Actinomycetaceae|g__Actinomyces"] <- as.numeric(df[,"k__Bacteria|__|__|__|__|__ _div_ k__Bacteria|p__Actinobacteria|c__Actinobacteria|o__Actinomycetales|f__Actinomycetaceae|g__Actinomyces"])
df[,"k__Bacteria|__|__|__|__|__ _div_ k__Bacteria|p__Actinobacteria|c__Actinobacteria|o__Actinomycetales|f__Propionibacteriaceae|g__Propionibacterium"] <- as.numeric(df[,"k__Bacteria|__|__|__|__|__ _div_ k__Bacteria|p__Actinobacteria|c__Actinobacteria|o__Actinomycetales|f__Propionibacteriaceae|g__Propionibacterium"])
df[,"k__Bacteria|__|__|__|__|__ _div_ k__Bacteria|p__Actinobacteria|c__Actinobacteria|o__Bifidobacteriales|f__Bifidobacteriaceae|g__"] <- as.numeric(df[, "k__Bacteria|__|__|__|__|__ _div_ k__Bacteria|p__Actinobacteria|c__Actinobacteria|o__Bifidobacteriales|f__Bifidobacteriaceae|g__" ])                                                                                                                                 
df[, "k__Bacteria|__|__|__|__|__ _div_ k__Bacteria|p__Actinobacteria|c__Actinobacteria|o__Bifidobacteriales|f__Bifidobacteriaceae|g__Bifidobacterium"]<- as.numeric(df[, "k__Bacteria|__|__|__|__|__ _div_ k__Bacteria|p__Actinobacteria|c__Actinobacteria|o__Bifidobacteriales|f__Bifidobacteriaceae|g__Bifidobacterium"])
df[, "k__Bacteria|__|__|__|__|__ _div_ k__Bacteria|p__Actinobacteria|c__Coriobacteriia|o__Coriobacteri"]<- as.numeric(df[, "k__Bacteria|__|__|__|__|__ _div_ k__Bacteria|p__Actinobacteria|c__Coriobacteriia|o__Coriobacteri"])                                                                                                                                                     
       
df =read.table("C:/Users/Hello/Desktop/IODAA/fil_rouge/ratio_genre_trié_avec_age_transpose.csv")
                                                                                                                                              
res.pca <- PCA(df, graph = TRUE)
eig.val <- get_eigenvalue(res.pca)
fviz_eig(res.pca, addlabels = TRUE, ylim = c(0, 50))
fviz_pca_var(res.pca, col.var = "black")
