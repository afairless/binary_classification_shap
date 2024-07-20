 
import polars as pl

from src.s04_model_train import (
    one_hot_encode_str_columns,
    )


def test_one_hot_encode_str_columns_01():
    """
    Test no one-hot encoding because input lacks any columns of strings
    """

    df = pl.DataFrame({
        'a': [1, 2, 3, 4, 5],
        'b': [1, 1, 2, 2, 2],
        'c': [9, 8, 8, 7, 7]})

    result = one_hot_encode_str_columns(df)

    correct_result = pl.DataFrame({
        'a': [1, 2, 3, 4, 5],
        'b': [1, 1, 2, 2, 2],
        'c': [9, 8, 8, 7, 7]})

    assert result.equals(correct_result)


def test_one_hot_encode_str_columns_02():
    """
    Test one-hot encoding one column of strings
    """

    df = pl.DataFrame({
        'a': [1, 2, 3, 4, 5],
        'b': ['1', '1', '2', '2', '2'],
        'c': [9, 8, 8, 7, 7]})

    result = one_hot_encode_str_columns(df)

    correct_result = pl.DataFrame({
        'b_1': [1, 1, 0, 0, 0],
        'b_2': [0, 0, 1, 1, 1],
        'a': [1, 2, 3, 4, 5],
        'c': [9, 8, 8, 7, 7]})

    assert result.equals(correct_result)


def test_one_hot_encode_str_columns_03():
    """
    Test one-hot encoding two columns of strings
    """

    df = pl.DataFrame({
        'a': [1, 2, 3, 4, 5],
        'b': ['1', '1', '2', '2', '2'],
        'c': ['9', '8', '8', '7', '7']})

    result = one_hot_encode_str_columns(df)

    correct_result = pl.DataFrame({
        'b_1': [1, 1, 0, 0, 0],
        'b_2': [0, 0, 1, 1, 1],
        'c_7': [0, 0, 0, 1, 1],
        'c_8': [0, 1, 1, 0, 0],
        'c_9': [1, 0, 0, 0, 0],
        'a': [1, 2, 3, 4, 5]})

    assert result.equals(correct_result)


def test_one_hot_encode_str_columns_04():
    """
    Test one-hot encoding all columns of strings
    """

    df = pl.DataFrame({
        'a': ['1', '2', '3', '4', '5'],
        'b': ['1', '1', '2', '2', '2'],
        'c': ['9', '8', '8', '7', '7']})

    result = one_hot_encode_str_columns(df)

    correct_result = pl.DataFrame({
        'a_1': [1, 0, 0, 0, 0],
        'a_2': [0, 1, 0, 0, 0],
        'a_3': [0, 0, 1, 0, 0],
        'a_4': [0, 0, 0, 1, 0],
        'a_5': [0, 0, 0, 0, 1],
        'b_1': [1, 1, 0, 0, 0],
        'b_2': [0, 0, 1, 1, 1],
        'c_7': [0, 0, 0, 1, 1],
        'c_8': [0, 1, 1, 0, 0],
        'c_9': [1, 0, 0, 0, 0]})

    assert result.equals(correct_result)


