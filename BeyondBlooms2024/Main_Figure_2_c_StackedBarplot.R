#library
library(data.table)
library(igraph)
library(ggplot2)

#Load data
load("../../../MOSAiC_Data/4Y/F4-TimeSeriesData-euk.Rdata")

meta=fread("../../../MOSAiC_Data/Manuscripts/NW_ELA_Paper/Networktables/newNWfiles/MetaData.csv",
           sep = ";",header = T,drop=c(1,2,3))

ccm_withpopa_pwert <- fread("../../../MOSAiC_Data/Manuscripts/NW_ELA_Paper/Networktables/newNWfiles/ccm_withpopa_pwert_non_pruned_ALL.csv",
                            sep = ";",header = T,drop=c(1,2))

ELA_default=fread("../../../MOSAiC_Data/Manuscripts/NW_ELA_Paper/ELA/ELA_all_cluster_defaultCondition.csv",sep = ",",header=T)
rownames(ELA_default)=ELA_default$ASV


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

getClusTaxAbd=function(cls,meta){
  i=which(meta$Cluster==cls)
  kk=meta[i,]
  gg=aggregate(kk$relAbund,list(kk$my_tax),sum)
  gg$y=gg$x/sum(gg$x)
  gg=gg[order(gg$y,decreasing=T),]
  gg$cm=cumsum(gg$y)
  return(gg)
}

mi=match(ccm_withpopa_pwert$from,meta$Nodes)
ccm_withpopa_pwert$from_month=meta$MaxMonth[mi]
ccm_withpopa_pwert$from_Class=meta$Class[mi]
ccm_withpopa_pwert$from_Cluster=meta$Cluster[mi]

mi=match(ccm_withpopa_pwert$to,meta$Nodes)
ccm_withpopa_pwert$to_month=meta$MaxMonth[mi]
ccm_withpopa_pwert$to_Class=meta$Class[mi]
ccm_withpopa_pwert$to_Cluster=meta$Cluster[mi]

i=which(ccm_withpopa_pwert$`p-value`<0.05)
ccmp=ccm_withpopa_pwert[i,]
ccmp$sameClu=0
ccmp$sameClu[ccmp$from_clu==ccmp$to_clu]=1

inet=graph_from_data_frame(ccmp[,c(1,2,3)],directed = T)
E(inet)$weight=ccmp$corr
cn=components(inet)
inet=induced.subgraph(inet,which(cn$membership==1))

#calculate closeness for cluster
tt=induced.subgraph(inet,which(is.element(V(inet)$name,meta$Nodes[meta$Cluster=="10-H"])))
tcn=components(tt)
tt=induced.subgraph(tt,which(tcn$membership==1))
tcl=closeness(tt,mode="all",normalized = T)

### barplot cluster taxa
tt=table(meta$my_tax,meta$Cluster)
tt=tt[,1:10]
rs=rowSums(tt)
rs=rs/sum(rs)
top_tt=tt[names(head((sort(rs,decreasing=T)),10)),]
top_tt=top_tt[c("Ochrophyta","Haptophyta","Chlorophyta","Cercozoa","Dinophyceae","Sagenista","Opalozoa","Radiolaria","Ciliophora","Syndiniales"),]
xx=reshape2::melt(top_tt)
colnames(xx)=c("taxa","cluster","proportion")

fill_col=c("#2ecc71", "#33ffcc", "#669966", "#aa6e28", "#ff9933", "#ffa500", "#ffb300", "#d95b43", "#9932CC", "#9999FF")
ggplot(xx,aes(x=cluster,y=proportion,fill=taxa))+geom_bar(stat = "identity",position="fill")+theme_light()+scale_fill_manual(values = fill_col)

