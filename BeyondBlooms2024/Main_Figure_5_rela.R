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


## Energy Landscape Analysis
set.seed(42)

# set pathes to save figures and tables.
figure_path ="CCMN_Paper_Submission/figures/woSeason_final/"
table_path =  "CCMN_Paper_Submission/tables/woSeason_final/"


# set the thresholds:
threshold = 0.05
threshold_interact = 0.05
threads = 20
reps =512



# Modus:
mod = "withoutSeason"
#mod = "Season"
#month_to_grep = "-12-|-01-|-02-"
#cluster_no= paste0(cluster_no, '_', month_to_grep)


# Pathes to the BaseTable for each Cluster (ASVs Abdundances)
all_clusters = list(
"CCMN_Paper_Submission/data/Raw_Ela_Cluster_03LW.csv",
"CCMN_Paper_Submission/data/Raw_Ela_Cluster_10HS.csv",
"CCMN_Paper_Submission/data/Raw_Ela_Cluster_09HS.csv",
"CCMN_Paper_Submission/data/Raw_Ela_Cluster_06TS.csv",
"CCMN_Paper_Submission/data/Raw_Ela_Cluster_08TS.csv",
"CCMN_Paper_Submission/data/Raw_Ela_Cluster_01TA.csv",
"CCMN_Paper_Submission/data/Raw_Ela_Cluster_05LW.csv",
"CCMN_Paper_Submission/data/Raw_Ela_Cluster_02TA.csv",
"CCMN_Paper_Submission/data/Raw_Ela_Cluster_04LW.csv",
"CCMN_Paper_Submission/data/Raw_Ela_Cluster_07TS.csv"
)


### loop throuh all clusters:

for (cluster_file in all_clusters) {
  # Extract cluster name and season using string manipulation
  # create a name prefix
  cluster_no ="REla_with_env_"
  parts <- unlist(strsplit(cluster_file, "/"))
  filename <- parts[length(parts)]
  cluster_parts <- unlist(strsplit(filename, "_"))
  
  cluster_name <- unlist(strsplit(cluster_parts[4], ".csv"))[1]  # Extract the cluster name
  season <- substr(cluster_parts[4], 3, 4)  # Extract the season (e.g., "L" or "H")
  
  # Print or use cluster_name and season in your specific actions
  cat("Cluster Name:", cluster_name, "Season:", season, "\n")
  
  # select a relative abudance based on the cluster season
  if (season == "LW") {
    month_to_grep = "-12-|-01-|-02-"
    threshold_abu <- 0.01
  } else if (season == "HS") {
    month_to_grep = "-06-|-07-|-08-"
    threshold_abu <- 0.02
  } else if (season == "TA") {
    month_to_grep = "-09-|-10-|-11-"
    threshold_abu <- 0.02
  } else if (season == "TS") {
    month_to_grep = "-03-|-04-|-05-"
    threshold_abu <- 0.02
  } else {
    result <- "error"
  }

  cluster_no = paste0(cluster_no, '_mod_', mod, '_', cluster_name, '_', month_to_grep, '_tr', threshold_abu)
  cat("ClusterFileName", cluster_no)
  
  baseabtable = read.csv(cluster_file, sep=';')
  rownames(baseabtable) = baseabtable$Unnamed..0
  baseabtable$Unnamed..0 = NULL
  
  # take the 100 most abundance asvs
  #View(baseabtable)
  col_sums <- colSums(baseabtable)
  sorted_indices <- order(col_sums, decreasing = TRUE)
  top_100_abundant_asvs <- baseabtable[,sorted_indices[1:min(ncol(baseabtable),100)]]
  dim(top_100_abundant_asvs)
  baseabtable = top_100_abundant_asvs
  
  # read the Environmen Data
  basemetadata <- read.csv("CCMN_Paper_Submission/data/Raw_Env_data.csv", sep=';')
  rownames(basemetadata) = basemetadata$date
  #View(basemetadata)
  basemetadata$date = NULL
  basemetadata_orgin =basemetadata
  # select the frist 7 ENV Parameters
  basemetadata <-   subset(basemetadata, select =1:7)
  
  
  # Grep if only season
  if (mod != "Season") {
    print("No grep to season")
  
  } else {
    baseabtable = baseabtable[grep(month_to_grep,rownames(baseabtable)),]
    dim(baseabtable)
    basemetadata = basemetadata[grep(month_to_grep,rownames(basemetadata)),]
    dim(basemetadata)
  }
  
  
  
  # if you dont want to use all sample uncoment the head()
  #basemetadata <- head(basemetadata, 146)
  
  # show the first seven rows:
  head(baseabtable, 7)
  head(basemetadata, 7)
  
  
  # To group species with the same presence/absence pattern into one group,
  #set grouping to 1 and specify a number between 0 and 1 for grouping_th.
  # If 0, only species with the same presence/absence pattern will be grouped together.
  
  # standartize the env parameters, to rescale new data for projections 
  basemetadata_scaled <- scale(basemetadata)
  basemetadata = basemetadata_scaled
  
  
  # (Relative abundance threshold,  Occurrence threshold (lower) ,Occurrence threshold (upper))
  pruning_pars = c(threshold_abu, 0.01, 0.99)
  
  # format the tables and normalize the abudances using x / sum(x) (normalize =1)
  list[ocmat, abmat, enmat, samplelabel, specieslabel, factorlabel] <-Formatting(baseabtable,basemetadata, normalizeq=1, parameters=pruning_pars, grouping=0, grouping_th=0.)
  
  
  ############################################################################
  # Heatmap of the binar abundances
  ocmat_melted <- melt(ocmat)
  new_colnames <- c("Sample_day", "ASV", "Binar_Abund")  # Replace with your desired column names
  colnames(ocmat_melted) <- new_colnames
  str(ocmat_melted)
  # Save the result_df data frame to a CSV file with indexed file name
  csv_file_path <- paste0(table_path, "OCMAT_for", cluster_no,".csv")
  write.csv(ocmat_melted, file = csv_file_path, row.names = FALSE, quote = FALSE)
  
  heatmap_plot <- ggplot(ocmat_melted, aes(ASV, Sample_day, fill= Binar_Abund)) +
    geom_tile() +
    theme_minimal() +  # Using a minimal theme to start with
    theme(panel.grid = element_blank(),axis.text.x = element_text(angle = 90, hjust = 1))  # Remove panel grid lines
  
  # Specify the filename and path
  img_file_path <- paste0(figure_path, cluster_no, "heatmap_plot", ".png")   
  
  # Save the ggplot object as an image with a specific DPI (e.g., 300 DPI)
  ggsave(filename = img_file_path, plot = heatmap_plot, dpi = 300)
  ############################################################################
  
  heatmap_plot
  
  # Parameter fitting
  #runSA: ocmatrix, env (environmental parameters; with>SA / without>fullSA), 
  # qth (threshold for stopping computation), 
  # rep (number of fitting processes to compute mean parameter value), 
  # threads (number of parallel threads)
  
  sa <- runSA(ocmat=as.matrix(ocmat), enmat=as.matrix(enmat), qth=10^-5, rep=reps, threads=threads)


  # get the fixed env as median from the current season of the cluster
  
  enmat_env = apply(enmat[grep(month_to_grep,rownames(enmat)),], 2, median)
  
  
  cat("fixxed environmental condition",enmat_env)
  
  
  # look at the estimate params h hidden evn, species-species interactions, env- species interactions
  list[he,je,ge,hge] <- sa2params(sa, env =enmat_env)
  
  # write them as csv
  csv_file_path <- paste0(table_path, "Species_Interaction_J_Matrix", cluster_no,".RData")
  save(je, file = csv_file_path)
  csv_file_path <- paste0(table_path, "Species_Env_Interaction_G_Matrix", cluster_no,".RData")
  save(ge, file = csv_file_path)
  csv_file_path <- paste0(table_path, "Combined_Hidden_Species_Env_Interaction_Env_HG_Matrix", cluster_no,".RData")
  save(hge, file = csv_file_path)
  
  
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
  
  # Initialize an empty list
  sample_list <- list()
  
  # Loop through each row and convert it to a list
  for (i in 1:nrow(ocmat)) {
    sample_list[[i]] <- as.vector(ocmat[i, ])
  }
  sample_ids = as.vector(sapply(sample_list, bin2id))
  
  # Table of SSID, Energy, Community composition
  sstable <- as.data.frame(cbind(stablestates, stablen, t(as.data.frame(bin)))) %>%
    'colnames<-'(c('ID', 'Energy', colnames(ocmat))) %>%
    'rownames<-'(1: length(stablestates))
  sstable
  
  write.csv(x = sstable, file=paste0(table_path,"/", cluster_no, "Raw_sstable.csv")) 
  
  # ID and energy of tipping points
  as.data.frame(tippingpoints)
  as.data.frame(tippingen)
  
  # Energy of any community composition
  cEnergy(ocmat[1,], he, je)
  cEnergy(ocmat[1,], hge, je) # with env parameters
  # Find the stable state for a community composition
  Bi(ocmat[1,], he, je)
  Bi(ocmat[1,], hge, je) # with env parameters
  ## Principal component analysis
  # Observed community compositions plotted on a PC1,2 plane and color-coded by their stable states
  
  
  # Disconnectivity graph
  showDG(ela[[1]], ocmat, "test")
  
  
  # Visualization of species' interaction
  # showINt has no input parameter called env
  env= enmat_env
  showIntrGraph(ela[[1]],sa, th=0.01, annot_adj=c(0.75, 2.00))
  #### Data for 3d Plot
  
  

  
  # Calculate the stable state for each community composition
  result <- apply(ocmat, 1, function(row) cEnergy(row, hge, je))
  ss_c <- apply(ocmat, 1, function(row) Bi(row, hge, je)[[1]])
  ss_c1 <- apply(ocmat, 1, function(row) Bi(row, hge, je)[[2]])
  row_names <- rownames(ocmat) # Assuming row names are set in ocmat
  
  
  # NMDS
  csv_file_path <- paste0(table_path, "Raw_OCMAT_for__", cluster_no,".RData")
  save(ocmat, file = csv_file_path)
  
  nmds = metaMDS(ocmat+0.00000001, distance = "bray", try = 50, trymax = 100)
  nmds
  
  plot(nmds, display = c("sites"))
  
  NMDS = scores(nmds)
  nmds_sites =as.data.frame(NMDS$sites)
  

  # Combine the results into a dataframe
  #result_df <- data.frame(time = row_names, Energy = result, rel.MDS1 = adv$PC1, rel.MDS2 = adv$PC2, TargetStableState = ss_c, TargetStableStateEn = ss_c1)
  result_df <- data.frame(time = row_names,
                          Energy = result,
                          rel.MDS1 = nmds_sites$NMDS1,
                          rel.MDS2 = nmds_sites$NMDS2,
                          TargetStableState = ss_c,
                          TargetStableStateEn = ss_c1,
                          MLD = as.data.frame(basemetadata_orgin)$MLD,
                          PAR_satellite = as.data.frame(basemetadata_orgin)$PAR_satellite,
                          temp = as.data.frame(basemetadata_orgin)$temp,
                          sal = as.data.frame(basemetadata_orgin)$sal,
                          PW_frac = as.data.frame(basemetadata_orgin)$PW_frac,
                          O2_conc = as.data.frame(basemetadata_orgin)$O2_conc,
                          depth = as.data.frame(basemetadata_orgin)$depth,
                          Encoded_SS = sample_ids,
                          Coded_SS =as.data.frame(ocmat)
                          )
  result_df$EGAP = result_df$TargetStableStateEn-result_df$Energy
  cat(sum(result_df$EGAP))

  # Save the result_df data frame to a CSV file with indexed file name
  csv_file_path <- paste0(table_path, "Raw_NMDS_coordinates_for_", cluster_no,".csv")
  write.csv(result_df, file = csv_file_path, row.names = FALSE, quote = FALSE)
  print("end of script")

  stbwe <- Stability(sa, ocmat, enmat=enmat, threads=threads)
  csv_file_path_stab <- paste0(table_path, "Stability_", cluster_no,".csv")
  write.csv(stbwe, file = csv_file_path_stab, row.names = FALSE, quote = FALSE)
  head(stbwe, 5)

  cat("#################################End", "\n")
}

length(bin)