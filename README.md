# Hybrid Apriori Algorithm

An improved version of the Apriori algorithm for association rule mining, designed to be more flexible and efficient than the classic version.

## What is this?

This project implements a **Hybrid Apriori** algorithm that extracts association rules from transaction data. Unlike the classic Apriori, our version adapts to your data automatically and handles rare items better.

## Key Improvements

### 1. Adaptive Starting Threshold
Instead of guessing a support value, the algorithm calculates it automatically:
- Computes the average support of all single items
- Sets the starting threshold to **85% of this average**
- Minimum guaranteed value: 15%

### 2. Per-Item Support (MIS)
Each item has its own support threshold:
- Frequent items → higher threshold
- Rare items → lower threshold
- Formula: `MIS = max(item_support × 0.55, 0.03)`

### 3. Decreasing Threshold
The support threshold decreases as itemsets get longer:
- k=1 → 21.25%
- k=2 → 17.25%
- k=3 → 13.25%
- k=4 → 9.25%
- k=5 → 5.25%

This allows long, rare patterns to survive.

## Results

On our test dataset (701 transactions, 40 items):
- **142** frequent itemsets found
- **788** association rules generated
- **0.05 seconds** execution time

### Example Rules Found:
"Beaucoup d'apps" → "Comportement intensif"
Support: 39% | Confidence: 100% | Lift: 2.55


## How to Use

### Requirements
```bash
pip install pandas matplotlib psutil