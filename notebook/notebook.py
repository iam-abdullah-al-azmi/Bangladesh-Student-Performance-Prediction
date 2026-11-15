import marimo

__generated_with = "0.25.2"
app = marimo.App()


@app.cell
def _():
    import numpy as np
    import pandas as pd
    import seaborn as sns
    import matplotlib.pyplot as plt
    from data_profiling import ProfileReport

    # Data Preprocessing Libraries

    from sklearn.pipeline import Pipeline
    from sklearn.impute import SimpleImputer
    from sklearn.compose import ColumnTransformer
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler, OneHotEncoder

    # Regression Models

    from sklearn.linear_model import Ridge, LinearRegression
    from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor

    from sklearn.ensemble import VotingRegressor, StackingRegressor

    # Evaluation Metrics

    from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

    import warnings

    warnings.filterwarnings("ignore")
    return (
        ColumnTransformer,
        GradientBoostingRegressor,
        LinearRegression,
        OneHotEncoder,
        Pipeline,
        ProfileReport,
        RandomForestRegressor,
        Ridge,
        SimpleImputer,
        StackingRegressor,
        StandardScaler,
        VotingRegressor,
        mean_absolute_error,
        mean_squared_error,
        np,
        pd,
        plt,
        r2_score,
        sns,
        train_test_split,
    )


@app.cell
def _(pd):
    df = pd.read_csv("../data/bangladesh_student_performance.csv")
    return (df,)


@app.cell
def _(df):
    df
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Data Profiling
    """)
    return


@app.cell
def _(ProfileReport, df):
    profile_report = ProfileReport(
        df, title="Bangladesh Student Performance Prediction"
    )
    return (profile_report,)


@app.cell
def _(profile_report):
    profile_report.to_file("profile_report.html")
    return


@app.cell
def _(df):
    df.columns
    return


@app.cell
def _(df):
    df.columns = [
        "date",
        "st_gender",
        "st_age",
        "st_address",
        "st_fam_size",
        "pstatus",
        "m_edu",
        "f_edu",
        "m_job",
        "f_job",
        "st_relationship",
        "smoker",
        "tuition_fee",
        "time_with_friends",
        "ssc_result",
        "hsc_result",
    ]
    return


@app.cell
def _(df):
    df.columns
    return


@app.cell
def _(df):
    if "date" in df.columns:
        df.drop(columns=["date"], inplace=True)
    return


@app.cell
def _(df):
    df.shape
    return


@app.cell
def _(df, plt, sns):
    fig, axes = plt.subplots(nrows=2, ncols=3, figsize=(14, 6))

    sns.histplot(data=df, x="st_age", kde=True, ax=axes[0, 0])
    sns.histplot(data=df, x="m_edu", kde=True, ax=axes[0, 1])
    sns.histplot(data=df, x="f_edu", kde=True, ax=axes[0, 2])

    sns.histplot(data=df, x="tuition_fee", kde=True, ax=axes[1, 0])
    sns.histplot(data=df, x="time_with_friends", kde=True, ax=axes[1, 1])
    sns.histplot(data=df, x="ssc_result", kde=True, ax=axes[1, 2])

    fig.suptitle("Distribution of Numerical Features", fontsize=14)
    plt.tight_layout()
    plt.show()
    return


@app.cell
def _(df):
    X = df.drop("hsc_result", axis=1)
    y = df["hsc_result"]
    return X, y


@app.cell
def _(X):
    numerical_features = X.select_dtypes(include=["int64", "float64"]).columns
    return (numerical_features,)


@app.cell
def _(X):
    categorical_features = X.select_dtypes(include=["object"]).columns
    return (categorical_features,)


@app.cell
def _(numerical_features):
    numerical_features
    return


@app.cell
def _(categorical_features):
    categorical_features
    return


@app.cell
def _(Pipeline, SimpleImputer, StandardScaler):
    numerical_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    return (numerical_transformer,)


@app.cell
def _(OneHotEncoder, Pipeline, SimpleImputer):
    categorical_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(handle_unknown="ignore")),
        ]
    )
    return (categorical_transformer,)


@app.cell
def _(
    ColumnTransformer,
    categorical_features,
    categorical_transformer,
    numerical_features,
    numerical_transformer,
):
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numerical_transformer, numerical_features),
            ("cat", categorical_transformer, categorical_features),
        ]
    )
    return (preprocessor,)


@app.cell
def _(X, train_test_split, y):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    return X_test, X_train, y_test, y_train


@app.cell
def _(GradientBoostingRegressor, LinearRegression, RandomForestRegressor):
    reg_lr = LinearRegression()
    reg_rf = RandomForestRegressor(n_estimators=100, random_state=42)
    reg_gb = GradientBoostingRegressor(n_estimators=100, random_state=42)
    return reg_gb, reg_lr, reg_rf


@app.cell
def _(VotingRegressor, reg_gb, reg_lr, reg_rf):
    voting_reg = VotingRegressor(
        estimators=[("lr", reg_lr), ("rf", reg_rf), ("gb", reg_gb)]
    )
    return (voting_reg,)


@app.cell
def _(Ridge, StackingRegressor, reg_gb, reg_rf):
    stacking_reg = StackingRegressor(
        estimators=[("rf", reg_rf), ("gb", reg_gb)], final_estimator=Ridge()
    )
    return (stacking_reg,)


@app.cell
def _(reg_gb, reg_lr, reg_rf, stacking_reg, voting_reg):
    model_to_train = {
        "Linear Regression": reg_lr,
        "Random Forest": reg_rf,
        "Gradient Boosting": reg_gb,
        "Voting Ensemble": voting_reg,
        "Stacking Ensemble": stacking_reg,
    }
    return (model_to_train,)


@app.cell
def _(
    Pipeline,
    X_test,
    X_train,
    mean_absolute_error,
    mean_squared_error,
    model_to_train,
    np,
    preprocessor,
    r2_score,
    y_test,
    y_train,
):
    result = []

    for name, model in model_to_train.items():
        pipe = Pipeline([("preprocessor", preprocessor), ("model", model)])

        pipe.fit(X_train, y_train)

        y_pred = pipe.predict(X_test)

        r2 = r2_score(y_test, y_pred)
        mae = mean_absolute_error(y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))

        result.append(
            {
                "Model": name,
                "R2 Score": r2,
                "MAE": mae,
                "RMSE": rmse,
            }
        )
    return (result,)


@app.cell
def _(pd, result):
    result_df = pd.DataFrame(result)
    return (result_df,)


@app.cell
def _(result_df):
    result_df
    return


@app.cell
def _(model_to_train, result_df):
    best_model_name = result_df.iloc[0]["Model"]
    best_model_obj = model_to_train[best_model_name]
    return (best_model_obj,)


@app.cell
def _(Pipeline, best_model_obj, preprocessor):
    final_pipe = Pipeline([("preprocessor", preprocessor), ("model", best_model_obj)])
    return (final_pipe,)


@app.cell
def _(X_train, final_pipe, y_train):
    final_pipe.fit(X_train, y_train)
    return


@app.cell
def _(X_test, final_pipe):
    y_final_pred = final_pipe.predict(X_test)
    return (y_final_pred,)


@app.cell
def _(plt, sns, y_final_pred, y_test):
    plt.figure(figsize=(14, 6))
    sns.scatterplot(x=y_test, y=y_final_pred, alpha=0.6, color="teal")
    plt.plot([2, 5], [2, 5], color="red", linestyle="--")

    plt.xlabel("Actual HSC Result")
    plt.ylabel("Predicted HSC Result")

    plt.title("Actual vs Predicted HSC Result")
    plt.tight_layout()
    plt.grid(True)
    plt.show()
    return


if __name__ == "__main__":
    app.run()
