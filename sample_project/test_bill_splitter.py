"""Tests for the bill splitter. One of them fails."""

import pytest

from bill_splitter import split_bill


def test_splits_evenly_without_tip():
    assert split_bill(100, 4, tip_percent=0) == 25


def test_rejects_zero_people():
    with pytest.raises(ValueError):
        split_bill(100, 0, tip_percent=0)


def test_single_person_no_tip():
    assert split_bill(80, 1, tip_percent=0) == 80


def test_tip_is_a_percentage():
    assert split_bill(250, 1, tip_percent=10) == 275
