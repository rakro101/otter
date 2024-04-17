unavailable <- setdiff(c("Rcpp","RcppArmadillo","doParallel","tidyverse",'gsubfn', 'zoo','snow','plyr', 'gtools','ggsci','igraph', 'tidygraph','RColorBrewer',"stringdist","ggplot2", "reshape2", "vegan"), rownames(installed.packages()))
install.packages(unavailable)


# install rELA package
#install.packages("Rcpp")
#install.packages("RcppArmadillo")
#install.packages("doParallel")
#install.packages("rELA/rELA.v0.43.tar.gz", type = "source")

library("ggplot2")
library("reshape2")
library("Rcpp")
library("RcppArmadillo")
library("doParallel")
library('tidyverse')
library('gsubfn')
library('zoo')
library('snow')
library('plyr')
library('gtools')
library('ggsci')
library('igraph')
library('tidygraph')
library('RColorBrewer')
library("stringdist")
library("rELA")
library("vegan")

#source("rELA/rELA/R/visualization.R")
## Energy Landscape Analysis
set.seed(42)
### Download data
threshold = 0.05
threshold_interact = 0.05
threads = 20
reps =512
eid_ = "temp"

#Relative abundance threshold = 0.04 
#Occurrence threshold (lower) = 0.01 
#Occurrence threshold (upper) = 0.99 


figure_path ="/Users/raphaelkronberg/PycharmProjects/ELLEN/CCMN_Paper_Submission/figures/woSeason_final_proj_eo/"
table_path =  "/Users/raphaelkronberg/PycharmProjects/ELLEN/CCMN_Paper_Submission/tables/woSeason_final_proj_eo/"

cluster_no ="03-L-TestCASE-SommerrMonthMedian"
#cluster_no ="04-L-Fig5_Project_Condtions__final_"
#cluster_no ="05-L-Fig5_Project_Condtions__final_"

month_to_grep = "-06-|-07-|-08-"
cluster_no= paste0(cluster_no, '_', month_to_grep)

# Specify the path for your own data in the code below, if necessary.
#baseabtable = read.csv("CCMN_Paper/data/Raw_Ela_Cluster_03-L.csv", sep=';')
#baseabtable = read.csv("CCMN_Paper/data/Raw_Ela_Cluster_04-L.csv", sep=';')
baseabtable = read.csv("/Users/raphaelkronberg/PycharmProjects/ELLEN/CCMN_Paper_Submission/data/Raw_Ela_Cluster_03LW.csv", sep=';')
rownames(baseabtable) = baseabtable$Unnamed..0
baseabtable$Unnamed..0 = NULL

# take the 100 most abundance asvs
col_sums <- colSums(baseabtable)
sorted_indices <- order(col_sums, decreasing = TRUE)
top_100_abundant_asvs <- baseabtable[,sorted_indices[1:min(ncol(baseabtable),100)]]
dim(top_100_abundant_asvs)
baseabtable = top_100_abundant_asvs


basemetadata <- read.csv("CCMN_Paper_Submission/data/Raw_Env_data.csv", sep=';')
rownames(basemetadata) = basemetadata$date
basemetadata$date = NULL


# arctic conditions
basemetadata_arctic <- read.csv("rELA/data/RELA/Arctic_Env.csv", sep=';')
rownames(basemetadata_arctic) = basemetadata_arctic$date
basemetadata_arctic$date = NULL



#basemetadata <- head(basemetadata, 146)
head(baseabtable, 7)

head(basemetadata, 7)


# To group species with the same presence/absence pattern into one group,
#set grouping to 1 and specify a number between 0 and 1 for grouping_th.
# If 0, only species with the same presence/absence pattern will be grouped together.


basemetadata_scaled <- scale(basemetadata)
basemetadata = basemetadata_scaled



basemetadata_arctic <- scale(basemetadata_arctic)



pruning_pars = c(0.01, 0.01, 0.99) # 0.001, #0.01, #0.005

list[ocmat, abmat, enmat, samplelabel, specieslabel, factorlabel] <-Formatting(baseabtable,basemetadata, normalizeq=1, parameters=pruning_pars, grouping=0, grouping_th=0.)


enmat_arctic =as.matrix(basemetadata_arctic)


#################
# Heatmap
ocmat_melted <- melt(ocmat)
new_colnames <- c("Sample_day", "ASV", "Binar_Abund")  # Replace with your desired column names
colnames(ocmat_melted) <- new_colnames



str(ocmat_melted)
# Save the result_df data frame to a CSV file with indexed file name
#csv_file_path <- paste0(table_path, "OCMAT_for", cluster_no,".csv")
#write.csv(ocmat_melted, file = csv_file_path, row.names = FALSE, quote = FALSE)

heatmap_plot <- ggplot(ocmat_melted, aes(ASV, Sample_day, fill= Binar_Abund)) +
  geom_tile() +
  theme_minimal() +  # Using a minimal theme to start with
  theme(panel.grid = element_blank(),axis.text.x = element_text(angle = 90, hjust = 1))  # Remove panel grid lines

# Specify the filename and path
img_file_path <- paste0(figure_path, cluster_no, "heatmap_plot", ".png")   # Change this to your desired file path and name

# Save the ggplot object as an image with a specific DPI (e.g., 300 DPI)
ggsave(filename = img_file_path, plot = heatmap_plot, dpi = 300)
################

heatmap_plot

# Parameter fitting
#runSA: ocmatrix, env (environmental parameters; with>SA / without>fullSA), qth (threshold for stopping computation), rep (number of fitting processes to compute mean parameter value), threads (number of parallel threads)

sa <- runSA(ocmat=as.matrix(ocmat), enmat=as.matrix(enmat), qth=10^-5, rep=reps, threads=threads)

string_list = list("F4", "EGC")

for (string in string_list) {
  
  if (string == "F4") {
    # f4
    #enmat_env = apply(enmat[grep(month_to_grep, rownames(enmat)),], 2, mean)
    enmat_env = apply(enmat[grep(month_to_grep,rownames(enmat)),], 2, median)
    enmat_env__ = apply(enmat[grep(month_to_grep,rownames(basemetadata)),], 2, median)
    print(enmat_env__)
    print("Sommerbedingunen:")
    print(enmat_env)
    
    
  } else {
    # egc
    # winter EGC min temp
    enmat_env = apply(enmat[grep(month_to_grep,rownames(enmat)),], 2, median)
    # pruning_pars = c(0.01, 0.01, 0.99)
    enmat_env[1] = 0 #MLD
    enmat_env[2] = 40 #PAR fromf4
    enmat_env[3] = -2  #'temp'
    enmat_env[4] = 34 # sal -> zoo plankton
    enmat_env[5] = 1 # PW_Frac
    enmat_env[6] = 350 #O2_conc
    enmat_env[7] = 30 # depth
    enmat_env <- scale(t(enmat_env), center = attr(basemetadata_scaled, "scaled:center"), scale = attr(basemetadata_scaled, "scaled:scale"))
    enmat_env = t(enmat_env)
  }
  
  cluster_no= paste0(cluster_no, '_', string)
  
  cat("Current string: ", string, "\n")
  list[he,je,ge,hge] <- sa2params(sa, env =enmat_env)
  
  ## Analysis and visualization of energy landscape
  # ELA function
  
  elanp <- ELA(sa, env=enmat_env,
               SS.itr=20000, FindingTip.itr=10000, # <- the number of steps for finding stable states and tipping points (basically no need to change)
               threads=threads, reporting=TRUE)
  
  ela <- ELPruning(elanp, th=threshold, threads=threads)
  
  list[stablestates, stablen, tippingpoints, tippingen] <- ela[[1]]
  
  
  ## Stable states
  stablestates
  
  # Convert an integer representing a stable state (ssid) to a binary vector
  # ssid -> binary vector
  bin = as.list(lapply(stablestates, function(x){id2bin(x, ncol(ocmat))}))
  names(bin) <- stablestates
  bin
  print(bin)
  #Convert a binary vector to a ssid
  as.vector(sapply(bin, bin2id))
  
  # Table of SSID, Energy, Community composition
  sstable <- as.data.frame(cbind(stablestates, stablen, t(as.data.frame(bin)))) %>%
    'colnames<-'(c('ID', 'Energy', colnames(ocmat))) %>%
    'rownames<-'(1: length(stablestates))
  sstable
  
  write.csv(x = sstable, file=paste0(table_path,"/", string, "_", cluster_no, "Raw_sstable.csv")) 
  
  # ID and energy of tipping points
  as.data.frame(tippingpoints)
  as.data.frame(tippingen)
  
  # Energy of any community composition
  cEnergy(ocmat[1,], hge, je)
  
  # Find the stable state for a community composition
  Bi(ocmat[1,], hge, je)
  
  
  # Disconnectivity graph
  showDG(ela[[1]], ocmat, "test")
  
  #source("viso.R")
  
  # Visualization of species' interaction
  #showIntrGraph(ela[[1]], sa,env=enmat_env, th=0.01, # <- Threshold for links to be displayed
  #annot_adj=c(0.75, 2.00))
  
  env= enmat_env
  showIntrGraph(ela[[1]],sa, th=0.01, annot_adj=c(0.75, 2.00))
  #### Data for 3d Plot
  
  
  # Calculate the stable state for each community composition
  result <- apply(ocmat, 1, function(row) cEnergy(row, hge, je))
  ss_c <- apply(ocmat, 1, function(row) Bi(row, hge, je)[[1]])
  ss_c1 <- apply(ocmat, 1, function(row) Bi(row, hge, je)[[2]])
  
  row_names <- rownames(ocmat) # Assuming row names are set in ocmat
  
  
  # NMDS
  
  
  csv_file_path <- paste0(table_path, string, "_", "Raw_OCMAT_for__", cluster_no,".RData")
  save(ocmat, file = csv_file_path)
  
  
  # for numerical stability
  nmds = metaMDS(ocmat+0.00000001, distance = "bray", try = 100, trymax = 100)
  nmds
  
  plot(nmds, display = c("sites"))
  
  NMDS = scores(nmds)
  
  nmds_sites =as.data.frame(NMDS$sites)
  
  # Combine the results into a dataframe
  result_df <- data.frame(time = row_names, Energy = result, rel.MDS1 = nmds_sites$NMDS1, rel.MDS2 = nmds_sites$NMDS2, TargetStableState = ss_c, TargetStableStateEn = ss_c1)
  
  result_df$EGAP = result_df$TargetStableStateEn-result_df$Energy
  cat(sum(result_df$EGAP))
  
  # Save the result_df data frame to a CSV file with indexed file name
  csv_file_path <- paste0(table_path, string, "_", "Raw_NMDS_coordinates_for", cluster_no,".csv")
  write.csv(result_df, file = csv_file_path, row.names = FALSE, quote = FALSE)
  rowSums(ocmat)
  # loop
  print("end of script")
}


