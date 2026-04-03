# Hybrid Apriori Algorithm

An improved version of the Apriori algorithm that combines **three powerful techniques**:
- **Adaptive Minimum Support** (automatically calculated from the data)
- **Per-Item Minimum Support (MIS)** – handles rare and common items differently
- **Degressive Minimum Support** – decreases threshold for longer itemsets

### Features
- Better at discovering interesting rare patterns
- More intelligent threshold selection compared to classic Apriori
- Applied on a **User Behavior Dataset** (smartphone usage patterns)
- Generates strong association rules with high confidence and lift

### Technologies
- Python 3
- Pandas
- Matplotlib (for visualization)
- Jupyter Notebook

### Project Structure

Hybrid-Apriori/
├── Apriori_Hybrid.ipynb
├── user_behavior_dataset_cleaned.csv
├── visualisations_hybrid_regles.png
└── README.md

