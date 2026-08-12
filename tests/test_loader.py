import pytest

from src.arm.loader import load_transactions


def test_load_transactions(tmp_path):
    csv_file = tmp_path / "transactions.csv"

    csv_file.write_text(
        "transaction_id,item\n"
        "1,Bread\n"
        "1,Milk\n"
        "1,Butter\n"
        "2,Bread\n"
        "2,Milk\n"
        "3,Bread\n"
        "3,Butter\n",
        encoding="utf-8",
    )

    result = load_transactions(csv_file)

    assert result == [
        frozenset({"Bread", "Milk", "Butter"}),
        frozenset({"Bread", "Milk"}),
        frozenset({"Bread", "Butter"}),
    ]


def test_load_empty_csv(tmp_path):
    csv_file = tmp_path / "empty.csv"

    csv_file.write_text(
        "transaction_id,item\n",
        encoding="utf-8",
    )

    assert load_transactions(csv_file) == []


def test_invalid_header(tmp_path):
    csv_file = tmp_path / "invalid.csv"

    csv_file.write_text(
        "id,product\n"
        "1,Bread\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError):
        load_transactions(csv_file)


def test_skip_empty_values(tmp_path):
    csv_file = tmp_path / "transactions.csv"

    csv_file.write_text(
        "transaction_id,item\n"
        "1,Bread\n"
        ",Milk\n"
        "2,\n"
        "2,Butter\n",
        encoding="utf-8",
    )

    assert load_transactions(csv_file) == [
        frozenset({"Bread"}),
        frozenset({"Butter"}),
    ]