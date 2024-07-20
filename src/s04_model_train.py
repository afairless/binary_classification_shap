#! /usr/bin/env python3

import joblib
import numpy as np
import polars as pl
from pathlib import Path
from dataclasses import dataclass

from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split as skl_data_split 
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import balanced_accuracy_score

from scipy.stats import (
    randint as sci_randint, 
    uniform as sci_uniform, 
    loguniform as sci_loguniform)


if __name__ == '__main__':

    from s01_generate_data import create_data_01_with_parameters

    from common import (
        write_list_to_text_file, 
        get_binary_classification_threshold_and_best_metric)

else:

    from src.s01_generate_data import create_data_01_with_parameters

    from src.common import (
        write_list_to_text_file, 
        get_binary_classification_threshold_and_best_metric)


@dataclass
class SplitData:
    train_x: pl.DataFrame
    train_y: pl.Series
    test_x: pl.DataFrame
    test_y: pl.Series


def binarize_response_variable(
    df: pl.DataFrame, y_colname: str) -> dict[str, pl.DataFrame]:
    """
    Binarize the response/outcome variable in 'df' by its quartiles, and return
        resulting dataframes
    """

    quartile_str = ['25%', '50%', '75%']
    quartile_df = (
        df[y_colname]
        .describe()
        .filter(pl.col('statistic').is_in(quartile_str)))

    dfs = {}
    for q, q_value in quartile_df.iter_rows():

        new_y_colname = 'y' + q[:2]
        temp_df = df.with_columns(
            pl.when(pl.col(y_colname).gt(q_value))
            .then(1)
            .otherwise(0)
            .alias(new_y_colname)).drop(y_colname)

        dfs.update({new_y_colname: temp_df})

    return dfs


def one_hot_encode_str_columns(df: pl.DataFrame) -> pl.DataFrame:
    """
    One-hot encode string columns in 'df' and return a new DataFrame with the
        one-hot encoded columns concatenated with the original columns that were
        not one-hot encoded
    """

    str_colnames = [col for col in df.columns if df[col].dtype == pl.Utf8]
    # use list instead of subtracting sets to maintain column order
    other_colnames = [col for col in df.columns if col not in str_colnames]

    if not str_colnames:
        return df

    df_str = df.select(str_colnames)

    enc = OneHotEncoder(sparse_output=False)
    enc = enc.fit(df_str)
    arr_str_enc = enc.transform(df_str)
    df_str_enc = pl.DataFrame(arr_str_enc) 

    # convert ndarray to list to satisfy linter
    colnames = enc.get_feature_names_out().tolist()
    df_str_enc.columns = colnames 

    if not other_colnames:
        return df_str_enc

    df2 = pl.concat([df_str_enc, df[other_colnames]], how='horizontal')

    return df2


def split_data_train_test(
    df: pl.DataFrame, y_colname: str, train_size: float=0.8,
    random_state: int=7028934) -> SplitData:
    """
    Split the data into training and testing sets
    """

    train_data, test_data = skl_data_split(
        df, train_size=train_size, random_state=random_state) 
    assert isinstance(train_data, pl.DataFrame)
    assert isinstance(test_data, pl.DataFrame)
    assert len(train_data) > len(test_data)

    train_y = train_data[y_colname]
    train_x = train_data.drop(y_colname)
    test_y = test_data[y_colname]
    test_x = test_data.drop(y_colname)

    split_data = SplitData(train_x, train_y, test_x, test_y)

    return split_data


def train_model(
    data: SplitData, n_iter: int=10, cv: int=5
    ) -> tuple[RandomizedSearchCV, HistGradientBoostingClassifier]:
    """
    Train gradient-boosted tree model 
    Do hyperparameter optimization with a random search
    Refit the model on training data with the best hyperparameters and return 
        that model
    """

    model = HistGradientBoostingClassifier()

    parameter_space = [{
        'learning_rate': sci_loguniform(0.005, 0.2),
        'max_iter': sci_randint(50, 3000),
        'max_leaf_nodes': [31, 63, 127],
        'max_depth': sci_randint(5, 30),
        'min_samples_leaf': sci_randint(10, 200),
        'l2_regularization': sci_loguniform(0.0001, 0.2),
        'scoring': ['balanced_accuracy'],
        'tol': [1e-6],
        'validation_fraction': sci_uniform(0.1, 0.3),
        'n_iter_no_change': sci_randint(10, 30),
        'warm_start': [False],
        'early_stopping': [True],
        'random_state': [134267]}]
 
    search = RandomizedSearchCV(
        model, param_distributions=parameter_space, n_iter=n_iter, cv=cv, 
        refit=True, verbose=2, n_jobs=-1, random_state=30447)

    search_result = search.fit(data.train_x, data.train_y)

    final_model = HistGradientBoostingClassifier(**search_result.best_params_)
    final_model.fit(data.train_x, data.train_y)

    return search_result, final_model


def main():

    # from sklearn.metrics import get_scorer_names
    # get_scorer_names()


    ##################################################
    # SET CONFIGURATION
    ##################################################

    pl.Config.set_tbl_rows(10)
    pl.Config.set_tbl_cols(16)

    output_path = Path.cwd() / 'output' / 'model'
    output_path.mkdir(exist_ok=True, parents=True)

    mvn_components = create_data_01_with_parameters()

    colnames = ['x' + str(i) for i in range(mvn_components.cases_data.shape[1])]
    colnames[-1] = 'y0'
    df = pl.DataFrame(mvn_components.cases_data)
    df.columns = colnames
    
    dfs_dict = binarize_response_variable(df, colnames[-1])


    ##################################################
    # TRAIN AND SAVE MODELS
    ##################################################

    for new_y_colname, df2 in dfs_dict.items():

        data_x_y = split_data_train_test(df2, new_y_colname, train_size=0.85)

        search_result, model = train_model(data_x_y, n_iter=1000, cv=8)

        y_pred_proba_cross_valid = (
            search_result.predict_proba(data_x_y.train_x)[:, 1])
        best_threshold_metric_cross_valid = (
            get_binary_classification_threshold_and_best_metric(
                data_x_y.train_y.to_numpy(), y_pred_proba_cross_valid, 
                balanced_accuracy_score))

        y_pred_proba_test = model.predict_proba(data_x_y.test_x)[:, 1]
        best_threshold_metric_test = (
            get_binary_classification_threshold_and_best_metric(
                data_x_y.test_y.to_numpy(), y_pred_proba_test, 
                balanced_accuracy_score))

        descriptives = [
            new_y_colname,
            'best_threshold_metric_cross_valid',
            str(best_threshold_metric_cross_valid),
            'best_threshold_metric_test',
            str(best_threshold_metric_test)]


        ##################################################
        # SAVE DATA, MODEL, AND METRICS
        ##################################################

        train_df = pl.concat(
            [data_x_y.train_x, pl.DataFrame(data_x_y.train_y)], 
            how='horizontal')
        output_filename = new_y_colname + '_train_df.parquet'
        output_filepath = output_path / output_filename
        train_df.write_parquet(output_filepath)

        test_df = pl.concat(
            [data_x_y.test_x, pl.DataFrame(data_x_y.test_y)], 
            how='horizontal')
        output_filename = new_y_colname + '_test_df.parquet'
        output_filepath = output_path / output_filename
        test_df.write_parquet(output_filepath)

        output_filename = new_y_colname + '_df.parquet'
        output_filepath = output_path / output_filename
        df2.write_parquet(output_filepath)

        output_filename = new_y_colname + '_df.csv'
        output_filepath = output_path / output_filename
        df2.head(10).write_csv(output_filepath)

        output_filename = new_y_colname + '_metrics.txt'
        output_filepath = output_path / output_filename
        write_list_to_text_file(descriptives, output_filepath, True)

        output_filename = new_y_colname + '_model.pickle'
        output_filepath = output_path / output_filename
        joblib.dump(model, output_filepath)


if __name__ == '__main__':
    main()
