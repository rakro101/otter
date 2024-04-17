unavailable <- setdiff(c("mgcv","plotly","htmlwidgets","webshot2"), rownames(installed.packages()))
install.packages(unavailable)
library(mgcv)
library(plotly)
library(htmlwidgets)
# Create an empty list
plot_list <- list()

# List of file paths
file_pathes <- c(
  "/Users/raphaelkronberg/PycharmProjects/ELLEN/CCMN_Paper_Submission/tables/woSeason_final/Raw_NMDS_coordinates_for_REla_with_env__mod_withoutSeason_01TA_-09-|-10-|-11-_tr0.02.csv",
  "/Users/raphaelkronberg/PycharmProjects/ELLEN/CCMN_Paper_Submission/tables/woSeason_final/Raw_NMDS_coordinates_for_REla_with_env__mod_withoutSeason_02TA_-09-|-10-|-11-_tr0.02.csv",#damn
  "/Users/raphaelkronberg/PycharmProjects/ELLEN/CCMN_Paper_Submission/tables/woSeason_final/Raw_NMDS_coordinates_for_REla_with_env__mod_withoutSeason_03LW_-12-|-01-|-02-_tr0.01.csv",
  "/Users/raphaelkronberg/PycharmProjects/ELLEN/CCMN_Paper_Submission/tables/woSeason_final/Raw_NMDS_coordinates_for_REla_with_env__mod_withoutSeason_04LW_-12-|-01-|-02-_tr0.01.csv",
  "/Users/raphaelkronberg/PycharmProjects/ELLEN/CCMN_Paper_Submission/tables/woSeason_final/Raw_NMDS_coordinates_for_REla_with_env__mod_withoutSeason_05LW_-12-|-01-|-02-_tr0.01.csv",
  "/Users/raphaelkronberg/PycharmProjects/ELLEN/CCMN_Paper_Submission/tables/woSeason_final/Raw_NMDS_coordinates_for_REla_with_env__mod_withoutSeason_06TS_-03-|-04-|-05-_tr0.02.csv",
  "/Users/raphaelkronberg/PycharmProjects/ELLEN/CCMN_Paper_Submission/tables/woSeason_final/Raw_NMDS_coordinates_for_REla_with_env__mod_withoutSeason_07TS_-03-|-04-|-05-_tr0.02.csv",
  "/Users/raphaelkronberg/PycharmProjects/ELLEN/CCMN_Paper_Submission/tables/woSeason_final/Raw_NMDS_coordinates_for_REla_with_env__mod_withoutSeason_08TS_-03-|-04-|-05-_tr0.02.csv",
  "/Users/raphaelkronberg/PycharmProjects/ELLEN/CCMN_Paper_Submission/tables/woSeason_final/Raw_NMDS_coordinates_for_REla_with_env__mod_withoutSeason_09HS_-06-|-07-|-08-_tr0.02.csv",
  "/Users/raphaelkronberg/PycharmProjects/ELLEN/CCMN_Paper_Submission/tables/woSeason_final/Raw_NMDS_coordinates_for_REla_with_env__mod_withoutSeason_10HS_-06-|-07-|-08-_tr0.02.csv"
)

# just plot only one!
# ToDo: attention something is here Cashed. -> plot 1 by 1
file_pathes <- c(

  "/Users/raphaelkronberg/PycharmProjects/ELLEN/CCMN_Paper_Submission/tables/woSeason_final/Raw_NMDS_coordinates_for_REla_with_env__mod_withoutSeason_10HS_-06-|-07-|-08-_tr0.02.csv"
  
)

figure_path ="CCMN_Paper_Submission/figures/woSeason_final/"

#
basemetadata <- read.csv("CCMN_Paper_Submission/data/Raw_Env_data.csv", sep=';')
#View(basemetadata)
seasons = basemetadata$date
years = basemetadata$date
dim(basemetadata)
seasons[grep("-12-|-01-|-02-",seasons)] = '#7FACD3'
seasons[grep("-03-|-04-|-05-",seasons)] = '#85C29D'
seasons[grep("-06-|-07-|-08-",seasons)] = '#DCCDE1'
seasons[grep("-09-|-10-|-11-",seasons)] = '#E4985D'



# ( "circle" | "circle-open" | "cross" | "diamond" | "diamond-open" | "square" | "square-open" | "x" )
years[grep("2016",years)] = "circle" #"x"
years[grep("2017",years)] = "circle" #"diamond"
years[grep("2018",years)] = "circle" #"cross"
years[grep("2019",years)] = "circle" #"square"
years[grep("2020",years)] = "circle"

for (file_path in file_pathes) {
  tube_select_ <- regmatches(file_path, regexpr("mod_withoutSeason_(.{4})", file_path))
  tube_select_ = substr(tube_select_, nchar(tube_select_) - 3, nchar(tube_select_))
  df <- read.csv(file = file_path)
  print(tube_select_)
  df$time=rownames(df)
  dim(df)[[1]]
  number_of_replicates=1
  number_of_samples <- dim(df)[[1]]/number_of_replicates
  
  df$color <-  rep(hcl.colors(number_of_samples, "Spectral"), number_of_replicates)
  
  size_of_numbers = 30
  

  df$season = seasons
  df$years = years
  # remove outliers
  print(dim(df))
  #df<- df[!df$rel.MDS1 >= mean(df$"rel.MDS1")+2*sd(df$"rel.MDS1"), ]
  #df<- df[!df$rel.MDS2 >= mean(df$"rel.MDS2")+2*sd(df$"rel.MDS2"), ]
  #df<- df[!df$rel.MDS1 <= mean(df$"rel.MDS1")-2*sd(df$"rel.MDS1"), ]
  #df<- df[!df$rel.MDS2 <= mean(df$"rel.MDS2")-2*sd(df$"rel.MDS2"), ]
  df <- df %>%
    filter(abs(rel.MDS1 - mean(rel.MDS1)) <= 2 * sd(rel.MDS1),
           abs(rel.MDS2 - mean(rel.MDS2)) <= 2 * sd(rel.MDS2))
  print(dim(df))
  #df<- df[!df$rel.MDS1 >= 2*sd(df$"rel.MDS1"), ]
  #dim(df_filtered)
  
  ss1_name= "SS1: O1t"
  ss2_name = "SS2: O9x"
  ss3_name= "SS3: 1uV"
  ss4_name = "SS4: EWB"
  
  
  X_axis_label = 'NMDS1'
  Y_axis_label = 'NMDS2'
  Z_axis_label = 'Energy'
  
  energy_cut=0
  energy_cut_max=200
  energy_cut_min = -600
  
  ## -- Surface slope
  mod <- gam(Energy ~ te(rel.MDS1, k=number_of_replicates) + te(rel.MDS2, k=number_of_replicates) + ti(rel.MDS1, rel.MDS2, k=number_of_replicates), data=df)
  
  mds1.seq <- seq(min(df$rel.MDS1, na.rm=TRUE), max(df$rel.MDS1, na.rm=TRUE), length=number_of_samples)
  mds2.seq <- seq(min(df$rel.MDS2, na.rm=TRUE), max(df$rel.MDS2, na.rm=TRUE), length=number_of_samples)
  
  predfun <- function(x,y){
    newdat <- data.frame(rel.MDS1 = x, rel.MDS2=y)
    predict(mod, newdata=newdat)
  }
  fit <- outer(mds1.seq, mds2.seq, Vectorize(predfun))
  dim(fit)
  # restrict the plot
  if (energy_cut == 1) {
    fit[fit > energy_cut_max] <- energy_cut_max
    fit[fit < -energy_cut_min] <- -energy_cut_min
  }
  ###
  ## -- Plotly
  cs <- scales::rescale(quantile(fit, probs=seq(0,1,0.25)), to=c(0,1))
  
  names(cs) <-NULL
  frame6 <- plot_ly(data=df, x = ~rel.MDS1, y= ~rel.MDS2, z= ~Energy) %>% 
    add_trace(data=df, x = ~rel.MDS1, y= ~rel.MDS2, z= ~Energy,
              type = "scatter3d", mode = "markers",
              marker = list(color = ~df$season,
                            symbol="diamond",#~df$years,
                            size=5, legendgrouptitle=list(text='Energy', font='Arial'),
                            line=list(width=1,color='black')),
              name="Samples",
              opacity = 1)  %>% 
    add_trace(data = df, x = ~mds1.seq, y = ~mds2.seq, z = ~fit,
              type = "surface", showscale = TRUE,
              hidesurface = FALSE, opacity = 0.7,
              colorscale = list(
                cs,
                c('blue', 'lightblue', 'slategray', 'tan', 'indianred')
              ),
              contours = list(
                z = list(
                  show = TRUE,
                  start = min(t(fit)),
                  end = max(t(fit)),
                  usecolormap = TRUE,
                  size = 0.7,
                  width = 3
                )
              )
    )%>%
    layout( title = list(text = paste0("Cluster ", tube_select_), y = 0.9),
            scene = list(xaxis = list(title = X_axis_label, showticklabels=TRUE,nticks=10, linewidth=7, gridwidth=3),
                         yaxis = list(title = Y_axis_label, showticklabels=TRUE, nticks=10, linewidth=7, gridwidth =3),
                         zaxis = list(title = Z_axis_label, showticklabels=FALSE, nticks=10, linewidth=7, gridwidth =3),
                         aspectratio = list(x = .9, y = .9, z = 0.9),
                         font='Arial') )	
  
  
  
  frame6  

  name = paste0(figure_path, 'Figure_4_', tube_select_, 'Cluster_3d_', "_Energy_ELLEN_final_sub.html")
  saveWidget(frame6, name)
  
  
  # Append the frame6 plot to the list
  plot_list <- c(plot_list, list(frame6))
}
# enter the cluster number for plotting
plot_list[[1]]