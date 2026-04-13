import pandas as pd
from itertools import combinations
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np
from collections import Counter

# Chargement des données 
def charger_donnees(fichier):
    df = pd.read_csv(fichier, header=None, dtype=str)
    transactions = [
        sorted([str(item).strip() for item in row if pd.notna(item) and str(item).strip() != ''])
        for _, row in df.iterrows()
    ]
    return [t for t in transactions if t]

# Calcul du support
def calculer_support(itemset, transactions):
    n = len(transactions)
    if n == 0:
        return 0.0
    return sum(1 for t in transactions if frozenset(itemset).issubset(t)) / n

# Génération des candidats 
def generer_candidats(frequents_prec):
    candidats = []
    prec_set = {frozenset(item) for item in frequents_prec}
    
    for i in range(len(frequents_prec)):
        for j in range(i + 1, len(frequents_prec)):
            L1 = frequents_prec[i]
            L2 = frequents_prec[j]
            if L1[:-1] == L2[:-1]:
                candidat = sorted(L1 + [L2[-1]])
                if all(frozenset(sub) in prec_set for sub in combinations(candidat, len(candidat)-1)):
                    candidats.append(candidat)
    return candidats

#  Apriori Dégressif Adaptatif 
def apriori_degressif_adaptatif(transactions, minsup_initial=0.25, min_confiance=0.7):
    items_uniques = sorted({item for t in transactions for item in t})
    supports = {}
    epsilon = 1e-5
    
    print(f"Transactions  : {len(transactions)}")
    print(f"Items uniques : {len(items_uniques)}\n")
    
    # Niveau 1
    k = 1
    minsup_k = minsup_initial
    frequents_prec = []
    
    for item in items_uniques:
        sup = calculer_support([item], transactions)
        if sup >= minsup_k:
            fs = frozenset([item])
            supports[fs] = sup
            frequents_prec.append([item])
    
    print(f"k={k} | minsup={minsup_k:.4f} | {len(frequents_prec)} itemsets fréquents")
    
    # Niveaux suivants
    while True:
        k += 1
        nb_prec = len(frequents_prec)
        if nb_prec == 0:
            break
            
        reduction = 0.035 + (0.055 * (k - 1) / (nb_prec + 15))
        minsup_k = max(minsup_initial - (k - 1) * reduction, epsilon)
        
        candidats = generer_candidats(frequents_prec)
        if not candidats:
            break
            
        frequents_k = []
        for c in candidats:
            sup = calculer_support(c, transactions)
            if sup >= minsup_k:
                fs = frozenset(c)
                supports[fs] = sup
                frequents_k.append(sorted(c))
        
        print(f"k={k} | minsup={minsup_k:.4f} | {len(frequents_k)} itemsets fréquents")
        
        if not frequents_k:
            break
        frequents_prec = frequents_k
    
    return supports

# Extraction des règles
def extraire_regles(supports, min_confiance=0.7):
    regles = []
    for itemset, sup in supports.items():
        if len(itemset) < 2:
            continue
        for taille in range(1, len(itemset)):
            for ant in combinations(sorted(itemset), taille):
                ant_fs = frozenset(ant)
                cons_fs = itemset - ant_fs
                sup_ant = supports.get(ant_fs)
                sup_cons = supports.get(cons_fs)
                if sup_ant and sup_cons:
                    conf = sup / sup_ant
                    lift = conf / sup_cons if sup_cons > 0 else 0
                    if conf >= min_confiance:
                        regles.append((ant_fs, cons_fs, sup, conf, lift))
    return sorted(regles, key=lambda x: x[3], reverse=True)

# Exécution
FICHIER = "dataset/user_behavior_dataset_cleaned.csv"

transactions = charger_donnees(FICHIER)

supports = apriori_degressif_adaptatif(transactions, minsup_initial=0.25, min_confiance=0.7)

regles = extraire_regles(supports, 0.7)

print(f"\nItemsets fréquents : {len(supports)}")
print(f"Règles générées    : {len(regles)}\n")

# Affichage des 15 premières règles 
print("--- RÈGLES D'ASSOCIATION (Top 15) ---\n")

for i, (ant, cons, sup, conf, lift) in enumerate(regles[:15], 1):
    print(f"Règle {i:2d}: {set(ant)} --> {set(cons)} | "
          f"Support={sup:.3f}  Confiance={conf:.3f}  Lift={lift:.2f}")

print(f"\n... et {len(regles) - 15} autres règles non affichées.")

# Visualisation
print("\nCréation des visualisations...")

supports_vals   = [r[2] for r in regles]
confiances_vals = [r[3] for r in regles]
lifts_vals      = [r[4] for r in regles]

fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle("Visualisation des règles d'association — Apriori Dégressif Adaptatif", 
             fontsize=14, fontweight="bold")

# 1. Scatter Support vs Confiance
sc = axes[0, 0].scatter(supports_vals, confiances_vals, c=lifts_vals, 
                        cmap="RdYlGn", s=40, alpha=0.7, edgecolors="none")
plt.colorbar(sc, ax=axes[0, 0], label="Lift")
axes[0, 0].set_xlabel("Support")
axes[0, 0].set_ylabel("Confiance")
axes[0, 0].set_title("Support vs Confiance (couleur = Lift)")
axes[0, 0].axhline(y=0.7, color="grey", linestyle="--", linewidth=0.8, alpha=0.6)

# 2. Histogrammes
axes[0, 1].hist(supports_vals,   bins=20, alpha=0.6, label="Support",   color="steelblue")
axes[0, 1].hist(confiances_vals, bins=20, alpha=0.6, label="Confiance", color="darkorange")
axes[0, 1].hist(lifts_vals,      bins=20, alpha=0.5, label="Lift",      color="seagreen")
axes[0, 1].set_xlabel("Valeur")
axes[0, 1].set_ylabel("Fréquence")
axes[0, 1].set_title("Distribution Support / Confiance / Lift")
axes[0, 1].legend()

# 3. Top 15 items
ax3 = axes[1, 0]
all_items = [item for ant, cons, *_ in regles for item in list(ant) + list(cons)]
item_counts = Counter(all_items).most_common(15)
items_noms, items_freq = zip(*item_counts)
colors_bar = cm.Blues(np.linspace(0.4, 0.9, len(items_noms)))
ax3.barh(items_noms[::-1], items_freq[::-1], color=colors_bar[::-1])
ax3.set_xlabel("Occurrences dans les règles")
ax3.set_title("Top 15 items dans les règles d'association")
ax3.tick_params(axis="y", labelsize=9)

# 4. Matrice Lift moyen
ax4 = axes[1, 1]
top_items = [i for i, _ in Counter(all_items).most_common(8)]
matrix = np.zeros((len(top_items), len(top_items)))
counts = np.zeros((len(top_items), len(top_items)), dtype=int)

for ant, cons, _, _, lift in regles:
    for a in ant:
        for c in cons:
            if a in top_items and c in top_items:
                i = top_items.index(a)
                j = top_items.index(c)
                matrix[i][j] += lift
                counts[i][j] += 1

with np.errstate(invalid='ignore'):
    avg_lift = np.where(counts > 0, matrix / counts, np.nan)

im = ax4.imshow(avg_lift, cmap="YlOrRd", aspect="auto")
plt.colorbar(im, ax=ax4, label="Lift moyen")
ax4.set_xticks(range(len(top_items)))
ax4.set_yticks(range(len(top_items)))
ax4.set_xticklabels(top_items, rotation=45, ha="right", fontsize=8)
ax4.set_yticklabels(top_items, fontsize=8)
ax4.set_title("Lift moyen : antécédent → conséquent (top 8 items)")
ax4.set_xlabel("Conséquent")
ax4.set_ylabel("Antécédent")

plt.tight_layout()
plt.savefig("visualisations_regles_degressif.png", dpi=150, bbox_inches="tight")
plt.show()

print("\nFigure sauvegardée : visualisations_regles_degressif.png")