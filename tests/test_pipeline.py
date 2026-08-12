from src.arm.pipeline import mine_dataset
from src.arm.transaction import create_itemset


def test_mine_dataset(tmp_path):
    csv_file = tmp_path / "transactions.csv"

    csv_file.write_text(
        "transaction_id,item\n"
        "1,Bread\n"
        "1,Milk\n"
        "1,Butter\n"
        "2,Bread\n"
        "2,Milk\n"
        "3,Bread\n"
        "3,Butter\n"
        "4,Bread\n"
        "4,Milk\n"
        "4,Butter\n",
        encoding="utf-8",
    )

    frequent_itemsets, rules = mine_dataset(
        csv_file,
        min_support=0.50,
        min_confidence=0.70,
    )

    assert create_itemset(["Bread"]) in frequent_itemsets
    assert create_itemset(["Milk"]) in frequent_itemsets
    assert create_itemset(["Butter"]) in frequent_itemsets

    assert create_itemset(["Bread", "Milk"]) in frequent_itemsets
    assert create_itemset(["Bread", "Butter"]) in frequent_itemsets

    assert len(rules) > 0