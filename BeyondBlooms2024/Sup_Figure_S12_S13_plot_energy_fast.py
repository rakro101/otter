import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime

pathes = [
    "/Users/raphaelkronberg/PycharmProjects/ELLEN/CCMN_Paper_Submission/tables/woSeason_final/Raw_NMDS_coordinates_for_REla_with_env__mod_withoutSeason_10HS_-06-|-07-|-08-_tr0.02.csv",
    "/Users/raphaelkronberg/PycharmProjects/ELLEN/CCMN_Paper_Submission/tables/woSeason_final/Raw_NMDS_coordinates_for_REla_with_env__mod_withoutSeason_03LW_-12-|-01-|-02-_tr0.01.csv",
    "/Users/raphaelkronberg/PycharmProjects/ELLEN/CCMN_Paper_Submission/tables/woSeason_final_proj/F4_Raw_NMDS_coordinates_forTest_10-H-Fig5_Project_Condtions__final__-06-|-07-|-08-_F4.csv",
    "/Users/raphaelkronberg/PycharmProjects/ELLEN/CCMN_Paper_Submission/tables/woSeason_final_proj/F4_Raw_NMDS_coordinates_for03-L-Fig5_Project_Condtions__final__-12-|-01-|-02-_F4.csv",
    "/Users/raphaelkronberg/PycharmProjects/ELLEN/CCMN_Paper_Submission/tables/woSeason_final_proj_eo/F4_Raw_NMDS_coordinates_forTestCASE_10-H-WinterMonthMedian_-12-|-01-|-02-_F4.csv",
    "/Users/raphaelkronberg/PycharmProjects/ELLEN/CCMN_Paper_Submission/tables/woSeason_final_proj_eo/F4_Raw_NMDS_coordinates_for03-L-TestCASE-SommerrMonthMedian_-06-|-07-|-08-_F4.csv",
    "/Users/raphaelkronberg/PycharmProjects/ELLEN/CCMN_Paper_Submission/tables/woSeason_final_proj/EGC_Raw_NMDS_coordinates_forTest_10-H-Fig5_Project_Condtions__final__-06-|-07-|-08-_F4_EGC.csv",
    "/Users/raphaelkronberg/PycharmProjects/ELLEN/CCMN_Paper_Submission/tables/woSeason_final_proj/EGC_Raw_NMDS_coordinates_for03-L-Fig5_Project_Condtions__final__-12-|-01-|-02-_F4_EGC.csv",
    "/Users/raphaelkronberg/PycharmProjects/ELLEN/CCMN_Paper_Submission/tables/woSeason_final_proj_eo/EGC_Raw_NMDS_coordinates_forTestCASE_10-H-WinterMonthMedian_-12-|-01-|-02-_F4_EGC.csv",
    "/Users/raphaelkronberg/PycharmProjects/ELLEN/CCMN_Paper_Submission/tables/woSeason_final_proj_eo/EGC_Raw_NMDS_coordinates_for03-L-TestCASE-SommerrMonthMedian_-06-|-07-|-08-_F4_EGC.csv",
    # "/Users/raphaelkronberg/PycharmProjects/ELLEN/CCMN_Paper_Submission/tables/woSeason_final/Raw_NMDS_coordinates_for_REla_with_env__mod_withoutSeason_01-F_-09-|-10-|-11-_tr0.02.csv",
    # "/Users/raphaelkronberg/PycharmProjects/ELLEN/CCMN_Paper_Submission/tables/woSeason_final/Raw_NMDS_coordinates_for_REla_with_env__mod_withoutSeason_08-M_-03-|-04-|-05-_tr0.02.csv",
]

pathes2 = [
    "/Users/raphaelkronberg/PycharmProjects/ELLEN/CCMN_Paper_Submission/tables/woSeason_final/Raw_NMDS_coordinates_for_REla_with_env__mod_withoutSeason_01TA_-09-|-10-|-11-_tr0.02.csv",
    "/Users/raphaelkronberg/PycharmProjects/ELLEN/CCMN_Paper_Submission/tables/woSeason_final/Raw_NMDS_coordinates_for_REla_with_env__mod_withoutSeason_08TS_-03-|-04-|-05-_tr0.02.csv",
    "/Users/raphaelkronberg/PycharmProjects/ELLEN/CCMN_Paper_Submission/tables/woSeason_final/Raw_NMDS_coordinates_for_REla_with_env__mod_withoutSeason_10HS_-06-|-07-|-08-_tr0.02.csv",
    "/Users/raphaelkronberg/PycharmProjects/ELLEN/CCMN_Paper_Submission/tables/woSeason_final/Raw_NMDS_coordinates_for_REla_with_env__mod_withoutSeason_03LW_-12-|-01-|-02-_tr0.01.csv",
]

name_dict = {
    1: "Summercluster real",
    2: "Wintercluster real",
    3: "Summercluster  Atlantic Summer",
    4: "Wintercluster Atlantic Winter",
    5: "Sommercluster Atlantic Winter",
    6: "Wintercluster Atlantic Summer",
    7: "Summercluster Arctic Summer",
    8: "Wintercluster Arctic Winter",
    9: "Summercluster Arctic Winter",
    10: "Wintercluster Arctic Summer",
}

name_dict = {
    1: "Summer Cluster (10HS) Env: Original",
    2: "Winter Cluster (03LW) Env: Original",
    3: "Summer Cluster (10HS) Env: Atlantic Summer",
    4: "Winter Cluster (03LW) Env: Atlantic Winter",
    5: "Summer Cluster (10HS) Env: Atlantic Winter",
    6: "Winter Cluster (03LW) Env: Atlantic Summer",
    7: "Summer Cluster (10HS) Env: Arctic Summer",
    8: "Winter Cluster (03LW) Env: Arctic Winter",
    9: "Summer Cluster (10HS) Env: Arctic Winter",
    10: "Winter Cluster (03LW) Env: Arctic Summer",
    11: "Autumn Cluster (01TA) Env: Original",
    12: "Spring Cluster (08TA) Env: Original",
    13: "Summer Cluster (10HS) Env: Original",
    14: "Winter Cluster (03LW) Env: Original",
}

color_dict = {
    1: "#0199F2",
    2: "#3A5A72",
    3: "#0199F2",
    4: "#3A5A72",
    5: "#0199F2",
    6: "#3A5A72",
    7: "#0199F2",
    8: "#3A5A72",
    9: "#0199F2",
    10: "#3A5A72",
    11: "#4CB5AA",
    12: "#FABB2C",
    13: "#0199F2",
    14: "#3A5A72",
}

if __name__ == "__main__":

    titles = [p.split("/")[-1].split(".")[0] for p in pathes]

    pathes_eo = pathes  # Your list of file paths goes here

    # Create a figure with 10 subplots arranged in 2 rows and 5 columns
    fig, axes = plt.subplots(5, 2, figsize=(20, 20))

    # Flatten the axes array
    # so we can access each subplot individually
    axes = axes.flatten()

    # Loop through each file path and plot data in a subplot
    for i, path in enumerate(pathes_eo):
        # Read data from the CSV file
        df_03_eo_f4 = pd.read_csv(pathes_eo[i], sep=",")
        df_03_eo_f4["time"] = pd.to_datetime(df_03_eo_f4["time"])
        df_03_eo_f4[["time", "Energy"]].plot(
            kind="line",
            x="time",
            y="Energy",
            title=name_dict[i + 1],
            ax=axes[i],
            color=color_dict[i + 1],
        )
        # df_03_eo_f4[["time", "Energy"]].plot(kind="line", x="time", y="Energy", title=titles[i], ax=axes[i], color=color_dict[i+1])
        axes[i].set_ylabel("Energy")
        # Increase font size for x and y ticks
        axes[i].tick_params(axis="x", labelsize=16)
        axes[i].tick_params(axis="y", labelsize=16)
        axes[i].set_xlabel("Time", fontsize=16)
        axes[i].set_ylabel("Energy", fontsize=16)
        # Plot horizontal lines at specific x-values
        for y in [2016, 2017, 2018, 2019, 2020]:
            if y != 2016:
                axes[i].axvline(
                    x=datetime.strptime(f"31.03.{y}", "%d.%m.%Y"),
                    color="black",
                    linestyle="--",
                )  # Line at 21.12
            if y != 2020:
                axes[i].axvline(
                    x=datetime.strptime(f"01.10.{y}", "%d.%m.%Y"),
                    color="black",
                    linestyle="--",
                )  # Line at 21.06

        for y in range(2016, 2021):
            if y != 2020:
                axes[i].axvspan(
                    datetime.strptime(f"01.10.{y}", "%d.%m.%Y"),
                    datetime.strptime(f"31.03.{y+1}", "%d.%m.%Y"),
                    color="lightgrey",
                    alpha=0.5,
                )

    # Adjust layout to prevent overlap
    plt.tight_layout()
    plt.savefig("figures/Sup_Figure_S13_Energy_plots_subplots.png", dpi=300)
    plt.show()
    ######

    pathes_eo = pathes2  # Your list of file paths goes here

    # Create a figure with 10 subplots arranged in 2 rows and 5 columns
    fig, axes = plt.subplots(2, 2, figsize=(20, 8))

    # Flatten the axes array
    # so we can access each subplot individually
    axes = axes.flatten()

    # Loop through each file path and plot data in a subplot
    for i, path in enumerate(pathes_eo):
        # Read data from the CSV file
        df_03_eo_f4 = pd.read_csv(pathes_eo[i], sep=",")
        df_03_eo_f4["time"] = pd.to_datetime(df_03_eo_f4["time"])
        df_03_eo_f4[["time", "Energy"]].plot(
            kind="line",
            x="time",
            y="Energy",
            title=name_dict[i + 11],
            ax=axes[i],
            color=color_dict[i + 11],
        )
        axes[i].set_ylabel("Energy")
        axes[i].tick_params(axis="x", labelsize=16)
        axes[i].tick_params(axis="y", labelsize=16)
        axes[i].set_xlabel("Time", fontsize=16)
        axes[i].set_ylabel("Energy", fontsize=16)
        # Plot horizontal lines at specific x-values
        for y in [2016, 2017, 2018, 2019, 2020]:
            if y != 2016:
                axes[i].axvline(
                    x=datetime.strptime(f"31.03.{y}", "%d.%m.%Y"),
                    color="black",
                    linestyle="--",
                )  # Line at 21.12
            if y != 2020:
                axes[i].axvline(
                    x=datetime.strptime(f"01.10.{y}", "%d.%m.%Y"),
                    color="black",
                    linestyle="--",
                )  # Line at 21.06

        # Add light grey background between October 1st and March 31st
        for y in range(2016, 2021):
            if y != 2020:
                axes[i].axvspan(
                    datetime.strptime(f"01.10.{y}", "%d.%m.%Y"),
                    datetime.strptime(f"31.03.{y+1}", "%d.%m.%Y"),
                    color="lightgrey",
                    alpha=0.5,
                )

    # Adjust layout to prevent overlap
    plt.tight_layout()
    plt.savefig(
        "figures/Sup_Figure_S12_Energy_plots_subplots_real_01TA_08TS.png", dpi=300
    )
    plt.show()
