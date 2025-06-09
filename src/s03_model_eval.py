#! /usr/bin/env python3

import json
import joblib
import numpy as np
import polars as pl
from pathlib import Path
from typing import Callable
from dataclasses import dataclass, field, fields, asdict

from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.metrics import recall_score
from sklearn.metrics import precision_score
from sklearn.metrics import f1_score
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve
from sklearn.metrics import balanced_accuracy_score
from sklearn.metrics import brier_score_loss
from sklearn.metrics import matthews_corrcoef

from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.metrics import RocCurveDisplay
from sklearn.metrics import PrecisionRecallDisplay

import shap
import scipy as sp
from scipy.spatial.distance import pdist
from scipy.cluster.hierarchy import fcluster
from sklearn.manifold import TSNE
from matplotlib import pyplot as plt


if __name__ == '__main__':
    from common import (
        write_list_to_text_file, 
        get_binary_classification_threshold_and_best_metric)
else:
    from src.common import (
        write_list_to_text_file, 
        get_binary_classification_threshold_and_best_metric)


@dataclass
class BinaryClassificationMetrics:

    y_true: np.ndarray
    y_pred: np.ndarray
    y_pred_proba: np.ndarray = field(default_factory=lambda: np.array([]))
    classification_threshold: float = np.nan

    true_positive: float = np.nan
    true_negative: float = np.nan
    false_positive: float = np.nan
    false_negative: float = np.nan

    population: float = np.nan
    prevalence: float = np.nan

    accuracy: float = np.nan
    recall: float = np.nan
    sensitivity: float = np.nan
    true_positive_rate: float = np.nan
    tpr: float = np.nan
    specificity: float = np.nan
    true_negative_rate: float = np.nan
    tnr: float = np.nan
    precision: float = np.nan
    positive_predictive_value: float = np.nan
    ppv: float = np.nan
    negative_predictive_value: float = np.nan
    npv: float = np.nan
    false_positive_rate: float = np.nan
    fpr: float = np.nan
    false_omission_rate: float = np.nan
    f1: float = np.nan
    roc_auc: float = np.nan

    balanced_accuracy: float = np.nan
    informedness: float = np.nan
    markedness: float = np.nan
    mcc: float = np.nan

    brier: float = np.nan

    def __post_init__(self):

        assert self.y_true.ndim == 1
        assert self.y_pred.ndim == 1
        assert len(self.y_true) == len(self.y_pred)

        confusion = confusion_matrix(self.y_true, self.y_pred)
        tn, fp, fn, tp = confusion.ravel()
        self.true_positive = tp
        self.true_negative = tn
        self.false_positive = fp
        self.false_negative = fn

        self.population = tp + tn + fp + fn
        self.prevalence = (tp + fn) / self.population

        self.accuracy = accuracy_score(self.y_true, self.y_pred)
        self.recall = recall_score(self.y_true, self.y_pred)
        self.sensitivity = self.recall
        self.true_positive_rate = self.recall
        self.tpr = self.true_positive_rate 
        self.specificity = (tn / (tn + fp))
        self.true_negative_rate = self.specificity 
        self.tnr = self.true_negative_rate  
        self.precision = precision_score(self.y_true, self.y_pred)
        self.positive_predictive_value = self.precision 
        self.ppv = self.positive_predictive_value  
        self.negative_predictive_value = (tn / (tn + fn))
        self.npv = self.negative_predictive_value   
        self.false_positive_rate = (fp / (fp + tn))
        self.fpr = self.false_positive_rate  
        self.false_omission_rate = (fn / (fn + tn))
        self.f1 = f1_score(self.y_true, self.y_pred)

        self.balanced_accuracy = balanced_accuracy_score(
            self.y_true, self.y_pred)
        self.informedness = self.recall + self.specificity - 1
        self.markedness = self.ppv + self.npv - 1
        self.mcc = matthews_corrcoef(self.y_true, self.y_pred)

        if self.y_pred_proba.size > 0:
            brier = brier_score_loss(self.y_true, self.y_pred_proba)
            assert isinstance(brier, float)
            self.brier = brier

            roc_auc = roc_auc_score(self.y_true, self.y_pred_proba)
            assert isinstance(roc_auc, float)
            self.roc_auc = roc_auc 


    def round_floats(self, digits=2):
        for field in fields(self):
            if isinstance(getattr(self, field.name), float):
                setattr(
                    self, field.name, round(getattr(self, field.name), digits))

    def convert_int_to_float(self):
        """
        Make serialization more convenient by converting Numpy integers to 
            strings
        """
        for field in fields(self):
            # 'np.int' is deprecated, and using Python 'int' here doesn't work
            if isinstance(getattr(self, field.name), np.int64):
                setattr(self, field.name, float(getattr(self, field.name)))

    def convert_arrays_to_strings(self):
        """
        Make serialization more convenient by converting Numpy arrays to strings
        """
        for field in fields(self):
            if isinstance(getattr(self, field.name), np.ndarray):
                setattr(
                    self, field.name, str((getattr(self, field.name)).tolist()))


def get_binary_classification_metrics(
    y_true: np.ndarray, y_pred_proba: np.ndarray, 
    threshold_metric_func: Callable) -> BinaryClassificationMetrics:
    """
    Determine best-performing binary classification threshold according to
    'threshold_metric_func', use the threshold to calculate binary predictions,
    then calculate and return binary classification metrics
    """

    best_threshold_metric_test = (
        get_binary_classification_threshold_and_best_metric(
            y_true, y_pred_proba, threshold_metric_func))

    y_pred = y_pred_proba > best_threshold_metric_test.threshold

    classification_metrics = BinaryClassificationMetrics(
        y_true=y_true, y_pred=y_pred, y_pred_proba=y_pred_proba,
        classification_threshold=best_threshold_metric_test.threshold)

    return classification_metrics 


def plot_confusion_matrix(
    y_true: np.ndarray, y_pred_proba: np.ndarray, threshold: float, 
    output_filepath: Path):
    """
    Plot confusion matrix with positive cases along the top and left of the 2x2 
        grid
    """

    y_pred = y_pred_proba > threshold
    confusion = confusion_matrix(y_true, y_pred)[::-1, ::-1].transpose()

    plot = ConfusionMatrixDisplay(confusion, display_labels=[1, 0])
    fig, ax = plt.subplots(figsize=(4, 3))
    plot.plot(ax=ax)
    ax.set_ylabel('Predicted label', fontsize=14)
    ax.set_xlabel('True label', fontsize=14)
    plt.tight_layout()

    plt.savefig(output_filepath)
    plt.clf()
    plt.close()


def plot_roc_curve(
    y_true: np.ndarray, y_pred_proba: np.ndarray, tpr: float, fpr: float,
    prevalence: float, output_filepath: Path, plot_balance: bool=True):
    """
    Plot Receiver Operating Characteristic (ROC) Curve
    """

    slope = (1 - prevalence) / prevalence
    intercept = tpr - (slope * fpr)
    fprs = np.linspace(0.01, 0.99, 99)
    tprs = slope * fprs + intercept

    # roc = roc_curve(y_true, y_pred_proba_test)
    plot = (
        RocCurveDisplay(tpr=tpr, fpr=fpr)
        .from_predictions(y_true, y_pred_proba))
    plt.scatter(fpr, tpr, marker='o')

    if plot_balance:
        plot.ax_.plot([0, 1], [0, 1], color='black', linestyle='dotted')
    else:
        plot.ax_.plot(fprs, tprs, color='black', linestyle='dotted')

    plt.savefig(output_filepath)
    plt.clf()
    plt.close()


def plot_precision_recall_curve(
    y_true: np.ndarray, y_pred_proba: np.ndarray, 
    recall: float, precision: float, 
    informedness: float, markedness: float, 
    prevalence: float, output_filepath: Path):
    """
    Plot Receiver Operating Characteristic (ROC) Curve
    """

    plot = (
        PrecisionRecallDisplay(
            precision=precision, recall=recall)
        .from_predictions(y_true=y_true, y_pred=y_pred_proba))
    plt.scatter(recall, precision, marker='o')
    # plt.scatter(informedness, markedness, marker='x')
    plt.axhline(prevalence, color='black', linestyle='dotted')

    plt.savefig(output_filepath)
    plt.clf()
    plt.close()


def plot_hierarchical_cluster_dendrogram(
    shap_sample: np.ndarray, output_filepath: Path) -> np.ndarray:
    """
    Plot a dendrogram of hierarchical clustering of SHAP values
    """

    condensed_distance_matrix = pdist(shap_sample)
    cluster_linkage_matrix = (
        sp.cluster.hierarchy.complete(condensed_distance_matrix))

    plt.figure(figsize=(16, 8))
    plt.ylabel('Distance')
    plt.xlabel('Index')
    plt.tight_layout()
    sp.cluster.hierarchy.dendrogram(cluster_linkage_matrix, leaf_rotation=90)
    plt.savefig(output_filepath)
    plt.clf()
    plt.close()

    return cluster_linkage_matrix


def plot_shap_clusters_by_color(
    shap_sample: np.ndarray, cluster_linkage_matrix: np.ndarray, max_d: float,
    perplexities: list[int], random_state: int, output_filepath: Path):
    """
    Plot t-SNE clusters of SHAP values at different perplexities
    """

    # acceptable color schemes for clusters
    color_choices = [
        'viridis', 'brg', 'rainbow_r', 'tab10_r', 'tab20_r', 'turbo', 'turbo_r', 
        'Dark2_r']
    cmap=color_choices[1]

    tsne_colnames = ['TSNE_1', 'TSNE_2']

    for perplexity in perplexities:
        tsne = TSNE(
            n_components=2, perplexity=perplexity, random_state=random_state)
        tsne_embedding = tsne.fit_transform(shap_sample)
        cluster_labels = fcluster(
            cluster_linkage_matrix, max_d, criterion='distance')

        plt.scatter(
            tsne_embedding[:, 0], tsne_embedding[:, 1], 
            c=cluster_labels, cmap=cmap, alpha=0.3)

        plt.ylabel(tsne_colnames[0])
        plt.xlabel(tsne_colnames[1])
        title = f'{len(np.unique(cluster_labels))} clusters by color'
        plt.title(title)
        plt.tight_layout()

        output_filename = (
            output_filepath.stem + f'_{perplexity}' + output_filepath.suffix)
        output_filepath_i = output_filepath.parent / output_filename
        plt.savefig(output_filepath_i)

        plt.clf()
        plt.close()


def plot_shap_clusters_by_marker_1(
    shap_sample: np.ndarray, cluster_linkage_matrix: np.ndarray, max_d: float,
    perplexities: list[int], shap_feature: np.ndarray, feature_name: str,
    random_state: int, output_filepath: Path):
    """
    Plot t-SNE clusters of SHAP values at different perplexities
    """

    cmap='CMRmap_r'

    markers = ['o', '^', 'X', '+', 's', 'v', 'p', 'P', '*', 'D', 'd', 'H', 'h']

    tsne_colnames = ['TSNE_1', 'TSNE_2']

    for perplexity in perplexities:
        tsne = TSNE(
            n_components=2, perplexity=perplexity, random_state=random_state)
        tsne_embedding = tsne.fit_transform(shap_sample)
        cluster_labels = fcluster(
            cluster_linkage_matrix, max_d, criterion='distance')

        values = np.unique(cluster_labels)

        for i, value in enumerate(values):

            value_idxs = np.where(cluster_labels == value)[0]

            plt.scatter(
                tsne_embedding[value_idxs, 0], tsne_embedding[value_idxs, 1], 
                marker=markers[i], s=96, 
                c=shap_feature[value_idxs], cmap=cmap, 
                alpha=0.5)

        plt.ylabel(tsne_colnames[0])
        plt.xlabel(tsne_colnames[1])
        title = f'{len(values)} clusters by marker, colored by {feature_name}'
        plt.title(title)
        plt.tight_layout()

        output_filename = (
            output_filepath.stem + f'_mark1_{perplexity}_{feature_name}' + 
            output_filepath.suffix)
        output_filepath_i = output_filepath.parent / output_filename
        plt.savefig(output_filepath_i)

        plt.clf()
        plt.close()


def plot_shap_clusters_by_marker_2(
    shap_sample: np.ndarray, cluster_linkage_matrix: np.ndarray, max_d: float,
    perplexities: list[int], shap_feature: np.ndarray, feature_name: str,
    random_state: int, output_filepath: Path):
    """
    Plot t-SNE clusters of SHAP values at different perplexities
    """

    cmap='CMRmap_r'

    markers = ['o', '^', 'X', '+', 's', 'v', 'p', 'P', '*', 'D', 'd', 'H', 'h']

    tsne_colnames = ['TSNE_1', 'TSNE_2']

    for perplexity in perplexities:
        tsne = TSNE(
            n_components=2, perplexity=perplexity, random_state=random_state)
        tsne_embedding = tsne.fit_transform(shap_sample)
        cluster_labels = fcluster(
            cluster_linkage_matrix, max_d, criterion='distance')

        values = np.unique(cluster_labels)

        fig, ax = plt.subplots(2, 2, sharex=True, sharey=True)

        grid = [[0, 0], [0, 1], [1, 0], [1, 1]]

        for i in range(4):
            try:
                value = values[i]
            except:
                break
            value_idxs = np.where(cluster_labels == value)[0]
            ax[grid[i][0], grid[i][1]].scatter(
                tsne_embedding[value_idxs, 0], tsne_embedding[value_idxs, 1], 
                marker=markers[i], s=96, 
                c=shap_feature[value_idxs], cmap=cmap, 
                alpha=0.5)

        title = f'{len(values)} clusters by marker, colored by {feature_name}'
        plt.suptitle(title)
        plt.tight_layout()

        output_filename = (
            output_filepath.stem + f'_mark2_{perplexity}_{feature_name}' + 
            output_filepath.suffix)
        output_filepath_i = output_filepath.parent / output_filename
        plt.savefig(output_filepath_i)

        plt.clf()
        plt.close()


def plot_feature_importances_bar(df: pl.DataFrame, output_filepath: Path):
    """
    Plot feature importances on a horizontal bar chart
    """

    plt.grid(axis='x', zorder=0)
    plt.barh(df[:, 0], df[:, 1], zorder=2) 
    plt.xlabel('Importances')
    plt.tight_layout(pad=3.0)

    plt.title(f'Top {len(df)} Features')

    plt.savefig(output_filepath)

    plt.clf()
    plt.close()


def process_model_results(
    model_idx: int, output_path: Path, 
    data_input_filename: str, data_input_filepath: Path, 
    train_data_input_filepath: Path, test_data_input_filepath: Path,
    model_input_filepath: Path):


    ##################################################
    # LOAD ARTIFACTS
    ##################################################

    df = pl.read_parquet(data_input_filepath)
    # train_df = pl.read_parquet(train_data_input_filepath)
    test_df = pl.read_parquet(test_data_input_filepath)
    model = joblib.load(model_input_filepath)


    ##################################################
    # 
    ##################################################

    response_colname = df.columns[-1]
    y_pred_proba = model.predict_proba(
        test_df.drop(response_colname))[:, 1]

    optimize_metric = balanced_accuracy_score
    # optimize_metric = matthews_corrcoef
    y_true = test_df[response_colname].to_numpy()
    metrics = get_binary_classification_metrics(
        y_true, y_pred_proba, optimize_metric)

    output_filename = 'metrics.json'
    output_filepath = output_path / output_filename
    metrics.convert_int_to_float()
    metrics.convert_arrays_to_strings()
    metrics_dict = asdict(metrics)
    with open(output_filepath, 'w') as json_file:
        json.dump(metrics_dict, json_file, indent=4)

    # after serializing 'metrics', re-load its original values
    metrics = get_binary_classification_metrics(
        y_true, y_pred_proba, optimize_metric)
    
    output_filename = 'confusion.png'
    output_filepath = output_path / output_filename
    plot_confusion_matrix(
        y_true, y_pred_proba, metrics.classification_threshold, 
        output_filepath)

    output_filename = 'roc.png'
    output_filepath = output_path / output_filename
    plot_roc_curve(
        y_true, y_pred_proba, metrics.tpr, metrics.fpr, metrics.prevalence, 
        output_filepath, True)

    output_filename = 'precision_recall.png'
    output_filepath = output_path / output_filename
    plot_precision_recall_curve(
        y_true, y_pred_proba, 
        metrics.recall, metrics.precision, 
        metrics.informedness, metrics.markedness, 
        metrics.prevalence, output_filepath)


    ##################################################
    # 
    ##################################################

    y_colname = data_input_filename.split('_df')[0]
    df_x = df.drop(y_colname)
    df_x_pd = df_x.to_pandas()
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(df_x_pd)

    output_filename = 'df_x.parquet'
    output_filepath = output_path / output_filename
    df_x.write_parquet(output_filepath)

    output_filename = 'shap_values.npy'
    output_filepath = output_path / output_filename
    np.savetxt(output_filepath, shap_values, delimiter=',')


    ##################################################
    # FORCE PLOT
    ##################################################

    n = 200
    np.random.seed(513686)
    row_idxs = np.random.randint(0, df_x.shape[0], n)
    df_x_sample = df_x[row_idxs, :]
    shap_sample = shap_values[row_idxs, :]

    plot = shap.force_plot(
        explainer.expected_value, shap_sample, df_x_sample.to_pandas(), 
        show=True)
    plot_html = plot.html()

    output_filename = 'force_plot.html'
    output_filepath = output_path / output_filename
    write_list_to_text_file([plot_html], output_filepath)


    ##################################################
    # FEATURE IMPORTANCE 
    ##################################################

    # shap.summary_plot(shap_values, df_x_pd, plot_type='bar')
    # shap.summary_plot(shap_values, df_x_pd, max_display=7)

    feature_importances = np.abs(shap_values).mean(axis=0)
    feature_importances_df = (
        pl.DataFrame({
            'feature': df_x.columns, 
            'importance': feature_importances})
        ).sort('importance', descending=True)

    top_n_features = 10
    top_feature_df = feature_importances_df[:top_n_features, :].reverse()
    top_feature_df = top_feature_df.with_columns(
        pl.col('feature')
        .str.replace_all('_', ' ')
        .str.to_titlecase())

    output_filename = 'feature_importances.png'
    output_filepath = output_path / output_filename
    plot_feature_importances_bar(top_feature_df, output_filepath)

    output_filename = 'feature_importances.parquet'
    output_filepath = output_path / output_filename
    feature_importances_df.write_parquet(output_filepath)

    output_filename = 'feature_importances.csv'
    output_filepath = output_path / output_filename
    feature_importances_df.write_csv(output_filepath)


    ##################################################
    # SUPERVISED CLUSTERING
    ##################################################

    sample_n = 200
    np.random.seed(500765)
    row_idxs = np.random.randint(0, shap_values.shape[0], sample_n)
    shap_sample = shap_values[row_idxs, :]

    output_filename = 'dendrogram.png'
    output_filepath = output_path / output_filename
    cluster_linkage_matrix = (
        plot_hierarchical_cluster_dendrogram(shap_sample, output_filepath))

    # distance value 'max_d' determined from inspection of dendrogram
    if model_idx == 25:
        max_d = 2.5
    elif model_idx == 50:
        max_d = 3.2
    elif model_idx == 75:
        max_d = 3.2
    else:
        raise ValueError(
            f'Please inspect dendrogram for model {model_idx} and set "max_d"')

    perplexities = [30, 45, 60, 90]
    random_state = 636250
    output_filename = 'supervised_clusters.png'
    output_filepath = output_path / output_filename

    plot_shap_clusters_by_color(
        shap_sample, cluster_linkage_matrix, max_d, perplexities, random_state, 
        output_filepath)

    # plot the top 'feature_n' most important features
    feature_n = 7
    feature_srs = feature_importances_df.head(feature_n)['feature']
    for feature_name in feature_srs:

        col_idx = df_x.columns.index(feature_name)
        shap_feature = shap_sample[:, col_idx]
        # random_state = 636250
        plot_shap_clusters_by_marker_1(
            shap_sample, cluster_linkage_matrix, max_d, perplexities, 
            shap_feature, feature_name, random_state, output_filepath)

        plot_shap_clusters_by_marker_2(
            shap_sample, cluster_linkage_matrix, max_d, perplexities, 
            shap_feature, feature_name, random_state, output_filepath)


def main():

    # from sklearn.metrics import get_scorer_names
    # get_scorer_names()


    ##################################################
    # SET CONFIGURATION
    ##################################################

    pl.Config.set_tbl_rows(10)
    pl.Config.set_tbl_cols(16)

    # input_path = Path.cwd() / 'input'

    model_idxs = [25, 50, 75]

    for idx in model_idxs:

        print(f'Processing model {idx} results')

        model_input_path = Path.cwd() / 'output' / 'model'

        output_path = Path.cwd() / 'output' / f'model_eval_{idx}'
        output_path.mkdir(exist_ok=True, parents=True)

        data_input_filename = f'y{idx}_df.parquet'
        data_input_filepath = model_input_path / data_input_filename

        train_data_input_filename = f'y{idx}_train_df.parquet'
        train_data_input_filepath = model_input_path / train_data_input_filename

        test_data_input_filename = f'y{idx}_test_df.parquet'
        test_data_input_filepath = model_input_path / test_data_input_filename

        model_input_filename = f'y{idx}_model.pickle'
        model_input_filepath = model_input_path / model_input_filename

        process_model_results(
            idx, output_path, data_input_filename, data_input_filepath, 
            train_data_input_filepath, test_data_input_filepath,
            model_input_filepath)


if __name__ == '__main__':
    main()
