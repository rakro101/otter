setwd("/Users/ellen/Downloads/Update/ELA/CCMN_Paper_Submission/tables/woSeason_final/differentConditions/")

library(readr)

# Directory containing the files
directory <- "/Users/ellen/Downloads/Update/ELA/CCMN_Paper_Submission/tables/woSeason_final"

# List files that start with the specified name pattern
files <- list.files(path = directory, pattern = "^REla_with_env__mod_withoutSeason", full.names = TRUE)

# Initialize a list to store data frames
data_list <- list()

# Loop through each file and read it into R
for (file in files) {
  # Read the file into a data frame (adjust read function based on file format)
  df <- read.csv(file)  # Example assuming CSV files
  
  # Store the data frame in the list
  data_list[[file]] <- df
}

load("/Users/ellen/Documents/Uni/MOSAiC/MOSAiC_Data/4Y/F4-TESTTimeSeriesDataRAW-euk.Rdata")


# Initialize an empty list to store taxastableAtlantic data frames
all_taxastableAtlantic <- list()


# Iterate through each dataset
for (ind in seq_along(data_list)) {
  dt <- data_list[[ind]]
  da <- colSums(dt[, 4:ncol(dt)])
  zwei <- names(da)[which(unname(da) >= 1)]
  
  taxastableAtlantic <- TimeSeriesData$taxa_info[which(rownames(TimeSeriesData$taxa_info) %in% zwei), ]
  
  # Set cluster values
  taxastableAtlantic$cluster <- rep(0, nrow(taxastableAtlantic))
  taxastableAtlantic$cluster[which(rownames(taxastableAtlantic) %in% zwei)] <- ind
  
  # Save the modified taxastableAtlantic to a file or perform further operations
  # Example: saveRDS(taxastableAtlantic, file = paste0("taxastableAtlantic_", ind, ".rds"))
  # Store the modified taxastableAtlantic in the list
  all_taxastableAtlantic[[ind]] <- taxastableAtlantic
}

# Combine all taxastableAtlantic data frames into one
combined_taxastableAtlantic <- do.call(rbind, all_taxastableAtlantic)


write.table(combined_taxastableAtlantic, file = "/Users/ellen/Downloads/Update/ELA/CCMN_Paper_Submission/tables/Taxatable_StableState.csv", sep=";", quote = F)




