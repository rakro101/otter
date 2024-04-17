### library
library(data.table)
library(igraph)
library(ggplot2)

### data
load("BeyondBlooms2024/data/F4-TimeSeriesData-euk.Rdata")

con=fread("BeyondBlooms2024/tables/BeyondBlooms2024_Louvain_1_Pearson_FFT_Hellinger_True_14__complete_network_table_0.7_0.05.csv",
          sep = ";",header = T,drop=1)

ELA_default=fread("BeyondBlooms2024/tables/Main_Table_2_NonProjection_stablestate_latex_table_short.csv",sep = ",",header=T)
rownames(ELA_default)=ELA_default$ASV

meta=fread("BeyondBlooms2024/tables/Enriched_Paper_Meta.csv",
           sep = ",",header = T)

ccm_withpopa_pwert <- fread("BeyondBlooms2024/tables/BeyondBlooms2024_Hellinger_True_14_PV_CCM_CON_MAP_Network.csv",
                            sep = ";",header = T,drop=c(1))

#ToDo CON Centrality (non directed?)

### define my tax -> combination of phylum und class. use phylum as general taxa, decompose Dinoflagelatta by class
meta$my_tax=meta$Phylum
i=grep("Dinoflagellata",meta$Phylum)
meta$my_tax[i]=meta$Class[i]

### add sum over normalized abundace and calculate % abundace for each asv
mi=match(meta$Nodes,rownames(TimeSeriesData$abundMatRaw))
mm=apply(TimeSeriesData$abundMatRaw[mi,],2,function(x)sqrt(x/sum(x)))
meta$relAbund=rowSums(mm)/sum(rowSums(mm))

### add ELA info to meta
meta$ELA=0
meta$ELA[match(ELA_default$ASV,meta$Nodes)]=1


mi=match(ccm_withpopa_pwert$from,meta$Nodes)
ccm_withpopa_pwert$from_month=meta$MaxMonth[mi]
ccm_withpopa_pwert$from_Class=meta$Class[mi]
ccm_withpopa_pwert$from_Cluster=meta$cluster_names[mi]

mi=match(ccm_withpopa_pwert$to,meta$Nodes)
ccm_withpopa_pwert$to_month=meta$MaxMonth[mi]
ccm_withpopa_pwert$to_Class=meta$Class[mi]
ccm_withpopa_pwert$to_Cluster=meta$cluster_names[mi]

i=which(ccm_withpopa_pwert$`p-value`<0.05)
ccmp=ccm_withpopa_pwert[i,]
ccmp$sameClu=0
ccmp$sameClu[ccmp$from_clu==ccmp$to_clu]=1

# jetzt nehmen wir CCM aber hohes edge ist hier doch gut also 1-corr oder?
inet=graph_from_data_frame(ccmp[,c(1,2,3)],directed = T)
E(inet)$weight=ccmp$corr
cn=components(inet)
inet=induced.subgraph(inet,which(cn$membership==1))

bw=betweenness(inet,directed = T,normalized = T)
#get pvalue for betweenness
pval_bw=sapply(bw,function(x)t.test(bw,mu = x,alternative = "l")$p.value)
pAdj_bw=p.adjust(pval_bw,method = "fdr")

#get closeness
cln=closeness(inet,normalized = T,mode="total")
pval_cln=sapply(cln,function(x)t.test(cln,mu = x,alternative = "l")$p.value)
#z_cln=sapply(cln,function(x)(x-mean(cln))/sd(cln))
pAdj_cln=p.adjust(pval_cln,method = "fdr")

##### get meta info for ccm
km=meta[match(V(inet)$name,meta$Nodes),]
km$pval_cln=pAdj_cln[match(km$Nodes,names(pAdj_cln))]
km$cln=cln[match(km$Nodes,names(cln))]
#km$zscore_cln=z_cln[match(km$Nodes,names(z_cln))]
km$bw=bw[match(km$Nodes,names(bw))]
#km$zscore_bw=z_bw[match(km$Nodes,names(z_bw))]
km$pval_bw=pAdj_bw[match(km$Nodes,names(pAdj_bw))]

#### get only significant values
pp=km[km$ELA==1 & km$pval_cln<0.05,]
write.csv2(pp,file="BeyondBlooms2024/tables/Main_Table_2___Keystone_Species_OriginalSimCon.csv",
            quote = F)