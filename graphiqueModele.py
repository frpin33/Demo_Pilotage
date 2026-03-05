
import matplotlib.pyplot as plt 
import os                       
import numpy as np              
import pandas as pd

if __name__ == "__main__" :

    logfile = './logs/SEG_02272026T0912.log'

    f1List = []
    f1List = np.arange(0, 0.7, 0.01).tolist()

    df = pd.read_csv(logfile, sep=',', header=0)
    

    fig, ax1 = plt.subplots(figsize=(14, 7))

    epochs = df['epoch'].tolist()

    # 1. Configuration de l'axe X avec ROTATION
    ax1.set_xticks(epochs)
    ax1.set_xticklabels(epochs, rotation=45, ha='right', fontsize=9) # Rotation à 45°
    ax1.set_xlabel('Époques')

    # 2. Grilles verticales fortes pour alignement précis
    ax1.grid(True, which='major', axis='x', linestyle='-', alpha=0.5, color='gray')
    ax1.grid(True, which='major', axis='y', linestyle='--', alpha=0.3)

    # 3. Tracé avec marqueurs pour viser le point exact
    ax1.set_ylabel('Perte (Loss)', color='tab:red')
    ax1.plot(epochs, df['loss'].tolist(), color='tab:blue', label='Train Loss', marker='o', markersize=4, linewidth=1.5)
    ax1.plot(epochs, df['val_loss'].tolist(), color='tab:red', label='Validation Loss', marker='o', markersize=4, linewidth=1.5)
    ax1.tick_params(axis='y', labelcolor='tab:red')

    # 4. Deuxième axe Y (Dice / F1)
    ax2 = ax1.twinx() 
    ax2.set_ylabel('Dice / F1 Score', color='tab:green')
    ax2.plot(epochs, df['val_DiceCoef'].tolist(), color='tab:green', label='Val DiceCoef', linestyle='--', marker='s', markersize=4, linewidth=1.5)
    ax2.plot(epochs, f1List, color='tab:purple', label='F1 Score', marker='D', markersize=4, linewidth=1.5)
    ax2.tick_params(axis='y', labelcolor='tab:green')

    # 5. Titre et Légendes
    plt.title('Évolution de l\'entraînement : Loss vs Dice Coefficient', pad=20)

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper center', bbox_to_anchor=(0.5, -0.18), ncol=4)

    # 6. Sauvegarde haute précision
    plt.tight_layout()
    plt.show()

