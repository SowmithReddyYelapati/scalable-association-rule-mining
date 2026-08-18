# Scalable Association Rule Mining

A scalable implementation of **Apriori Association Rule Mining** with optimized support counting, pruning, benchmarking, visualization, and evaluation on the real-world UCI Online Retail dataset.

## Project Objective

The objective of this project is to implement and evaluate an Apriori-based association rule mining system and study its performance as dataset size and minimum support vary.

The project currently focuses on **Apriori as the baseline algorithm**.

## Features

- Apriori frequent itemset mining
- Candidate generation and pruning
- Optimized support counting
- Association rule generation
- Support, confidence, and lift calculation
- Synthetic scalability benchmarking
- Benchmark visualization
- Real-world Online Retail dataset integration
- Dataset preprocessing and transaction construction
- Cached transaction loading
- Real-world rule reporting
- Automated test suite

## Dataset

The project uses the **Online Retail dataset**.

Raw dataset:

- Records: **541,909**
- Invoices: **25,900**
- Products: **4,070**
- Countries: **38**

After preprocessing:

- Valid records: **530,693**
- Transactions: **20,136**
- Products used for mining: **3,925**

### Data preprocessing

The following records are removed:

1. Cancelled invoices
2. Transactions with non-positive quantities
3. Records with missing product descriptions

`CustomerID` is not required because the mining unit is the invoice/transaction.

## Transaction Representation

Each invoice is converted into a transaction represented as a set of product `StockCode` values.

Example:

```text
Invoice 536365

{
    85123A,
    71053,
    84406B,
    84029G,
    84029E,
    22752,
    21730
}