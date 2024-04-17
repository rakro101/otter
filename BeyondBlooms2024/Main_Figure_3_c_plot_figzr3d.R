# Libraries
unavailable <- setdiff(c("tidyverse","viridis","patchwork","tidyverse",'hrbrthemes', 'circlize'), rownames(installed.packages()))
install.packages(unavailable)

library(tidyverse)
library(viridis)
library(patchwork)
library(hrbrthemes)
library(circlize)

#devtools::install_github("mattflor/chorddiag")
library(chorddiag)
# Load dataset from github
#data <- read.table("paper/tables/matrix_cluster_cluster_distance_ALL_clean.csv", sep=",", header=TRUE)
data <- read.table("BeyondBlooms2024/tables/Main_Figure_S3_D_table.csv", sep=";", header=TRUE)
# short names
colnames(data) <- c("to_clu",data$to_clu)
rownames(data) <- data$to_clu
data$to_clu <- NULL

data[is.na(data)] <- 0
View(data)
mat <- as.matrix(data)

# Set the diagonal elements of the matrix to 0
diag(mat) <- 0
data <- data.frame(mat)
colnames(data) <- rownames(data)
#View(data)
# I need a long format
data_long <- data %>%
  rownames_to_column %>%
  gather(key = 'key', value = 'value', -rowname)

# parameters
circos.clear()
circos.par(start.degree = 90, gap.degree = 4, track.margin = c(-0.1, 0.1), points.overflow.warning = FALSE)
par(mar = rep(0, 4))

# color palette
mycolor <- viridis(10, alpha = 1, begin = 0, end = 1, option = "D")
mycolor <- c("#00B09D", "#FF6B6B", "#00203F","#6A0DAD", "#FD8D3C","#8E354A","#DDA0DD","#FFD700","#00695C","#0074E4")

# Base plot
chordDiagram(
  x = data_long,
  grid.col = mycolor,
  transparency = 0.25,
  directional = 1,
  direction.type = c("arrows", "diffHeight"),
  diffHeight  = -0.04,
  annotationTrack = "grid",
  annotationTrackHeight = c(0.05, 0.1),
  link.arr.type = "big.arrow",
  link.sort = TRUE,
  link.largest.ontop = TRUE)

# Add text and axis
circos.trackPlotRegion(
  track.index = 1,
  bg.border = NA,
  panel.fun = function(x, y) {

    xlim = get.cell.meta.data("xlim")
    sector.index = get.cell.meta.data("sector.index")

    # Add names to the sector.
    circos.text(
      x = mean(xlim),
      y = 3.2,
      labels = sector.index,
      facing = "bending",
      cex = 2.5
      )

  }

)

