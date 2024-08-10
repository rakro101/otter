import numpy as np
import pandas as pd

np.random.seed(42)

if __name__ == "__main__":
    # Permutation Pvalue
    df_permu = pd.read_csv(
        "tables/BeyondBlooms2024_MV_Hellinger_True_14_Pruned_Permutaion_CON_Network.csv",
        sep=";",
    )
    print(df_permu.head())
    # Pearson Pvalue
    df_p_val = pd.read_csv(
        "tables/BeyondBlooms2024_MV_Hellinger_True_14_Pearson_FFT__complete_network_table_0_None.csv",
        sep=";",
    )
    print(df_p_val.head())
    if True:
        df_p_val = df_p_val[df_p_val["p-value"] < 0.05]
        df_all = pd.merge(df_permu, df_p_val, on=["to", "from"], how="inner")
        print(df_all.columns)

        exp1 = df_all["p-value_x"]
        exp2 = df_all["p-value_y"]

        print(len(exp1))
        print(len(exp2))
        from scipy.stats import pearsonr

        # Calculate the Pearson correlation coefficient and p-value
        corr_coefficient, p_value = pearsonr(exp1, exp2)

        # Display the result
        print(f"Pearson Correlation Coefficient: {corr_coefficient}")
        print(f"P-value: {p_value}")
        import statsmodels.api as sm

        # Sample DataFrame with two columns
        X = sm.add_constant(exp1)

        # Fit the linear regression model
        model = sm.OLS(exp2, X).fit()

        # Get the summary of the regression
        print(model.summary())

        # Calculate the adjusted R-squared
        adj_r2 = 1 - (1 - model.rsquared) * (len(exp2) - 1) / (len(exp2) - 2 - 1)

        print(f"R-squared: {model.rsquared}")
        print(f"Adjusted R-squared: {adj_r2}")

        import matplotlib.pyplot as plt

        # plt.scatter(exp1, exp2, label=f'Original Data: Corr:{round(corr_coefficient,2)},p:{p_value} ')
        plt.scatter(
            exp1,
            model.predict(X),
            color="purple",
            label=f"Regression r2:{round(model.rsquared, 2)} y = {round(model.params[1], 5)}x +{round(model.params[0], 7)}",
        )
        # Add a constant term to the independent variable (X)
        # X = sm.add_constant(exp1)
        X = exp1
        # Fit the linear regression model
        model = sm.OLS(exp2, X).fit()

        # Get the summary of the regression
        print(model.summary())

        # Calculate the adjusted R-squared
        adj_r2 = 1 - (1 - model.rsquared) * (len(exp2) - 1) / (len(exp2) - 2 - 1)

        print(f"R-squared: {model.rsquared}")
        print(f"Adjusted R-squared: {adj_r2}")

        import matplotlib.pyplot as plt

        plt.scatter(
            exp1,
            exp2,
            label=f"Original Data: Corr:{round(corr_coefficient, 2)},p:{p_value} ",
        )

        # Plot the regression line
        # print('#####', len(model.predict(X)))
        plt.scatter(
            exp1,
            model.predict(X),
            color="red",
            label=f"Regression r2:{round(model.rsquared, 2)} y = {round(model.params[0], 5)}x ",
        )

        # Fit the GLM with an Exponential family distribution
        model = sm.GLM(exp2, sm.add_constant(exp1), family=sm.families.Poisson()).fit()

        # Display the summary of the regression
        print(model.summary())

        # Plot the original data
        # plt.scatter(exp1, exp2, label='Original Data')
        from sklearn.metrics import r2_score

        r2 = r2_score(exp2, model.predict(sm.add_constant(exp1)))
        print("R2 GLM", r2)
        # Plot the fitted values from the Exponential GLM
        plt.scatter(
            exp1,
            model.predict(sm.add_constant(exp1)),
            color="green",
            label=f"GLM r2:{round(r2, 2)}",
        )

        # Add labels and a legend
        plt.xlabel("permutation p-value [x]")
        plt.ylabel("(pearson) p-value [y]")
        plt.title("Compared permutation p-value with pearson p-value for the CON")
        plt.legend()

        plt.savefig(
            "figures/Sup_Figure_S10_pearson_pvalue_vs_permutation_value_ALL.png"
        )

        # Show the plot
        plt.show()
