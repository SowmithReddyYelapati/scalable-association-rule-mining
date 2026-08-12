from src.arm.transaction import create_itemset


def test_create_itemset():
    itemset = create_itemset(["Milk", "Bread"])

    assert itemset == frozenset({"Milk", "Bread"})


def test_duplicate_items_are_removed():
    itemset = create_itemset(["Milk", "Bread", "Milk"])

    assert itemset == frozenset({"Milk", "Bread"})


def test_item_order_does_not_matter():
    first = create_itemset(["Milk", "Bread"])
    second = create_itemset(["Bread", "Milk"])

    assert first == second


def test_empty_items_are_ignored():
    itemset = create_itemset(["Milk", "", "  ", "Bread"])

    assert itemset == frozenset({"Milk", "Bread"})