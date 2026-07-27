import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_curve,
    auc,
)

class ModelEvaluator:
    """Evalúa modelos de clasificación binaria: reports, matrices de confusión y curvas ROC."""

    SMOTE_ORDER = ["SMOTETK", "SMOTEENN", "NM-1", "IHT"]
    MODEL_ORDER = ["XGB", "LGBM"]

    def __init__(self, predicciones: dict, seed: int):
        """
        Parameters
        ----------
        predicciones : dict
            Clave = f"seed{semilla}_{smote}_{modelo}"
            Valor = {'y_test': array, 'y_pred': array, 'y_proba': array}
        seed : int
            Semilla a evaluar (42 o 123).
        """
        self.predicciones = predicciones
        self.seed = seed

    def _filter_by_seed(self):
        """Filtra predicciones solo para la semilla seleccionada."""
        prefix = f"seed{self.seed}_"
        return {k: v for k, v in self.predicciones.items() if k.startswith(prefix)}

    def classification_reports(self):
        """Imprime el classification report de cada combinación (solo test)."""
        filtered = self._filter_by_seed()
        for smote in self.SMOTE_ORDER:
            for model in self.MODEL_ORDER:
                key = f"seed{self.seed}_{smote}_{model}"
                if key not in filtered:
                    continue
                data = filtered[key]
                print(f"\n{'='*60}")
                print(f"  {smote} | {model}  (TEST) — Seed {self.seed}")
                print(f"{'='*60}")
                print(classification_report(
                    data["y_test"], data["y_pred"],
                    target_names=["Clase 0", "Clase 1"],
                    digits=4,
                ))

    def plot_confusion_matrices(self):
        """Grid 2×4 de matrices de confusión (test)."""
        filtered = self._filter_by_seed()
        fig, axes = plt.subplots(2, 4, figsize=(20, 8))
        fig.suptitle(f"Matrices de Confusión (TEST) — Seed {self.seed}", fontsize=16, y=1.02)

        for col_idx, smote in enumerate(self.SMOTE_ORDER):
            for row_idx, model in enumerate(self.MODEL_ORDER):
                ax = axes[row_idx, col_idx]
                key = f"seed{self.seed}_{smote}_{model}"
                if key not in filtered:
                    ax.set_visible(False)
                    continue

                data = filtered[key]
                cm = confusion_matrix(data["y_test"], data["y_pred"])

                sns.heatmap(
                    cm, annot=True, fmt="d", cmap="Blues",
                    xticklabels=["0", "1"], yticklabels=["0", "1"],
                    ax=ax, cbar=False,
                )
                ax.set_title(f"{smote}\n{model}", fontsize=11)
                ax.set_ylabel("Real" if row_idx == 0 else "")
                ax.set_xlabel("Predicho" if row_idx == 1 else "")

        plt.tight_layout()
        plt.show()

    def plot_roc_curves(self):
        """Grid 2×4 de curvas ROC (test)."""
        filtered = self._filter_by_seed()
        fig, axes = plt.subplots(2, 4, figsize=(20, 8))
        fig.suptitle(f"Curvas ROC (TEST) — Seed {self.seed}", fontsize=16, y=1.02)

        for col_idx, smote in enumerate(self.SMOTE_ORDER):
            for row_idx, model in enumerate(self.MODEL_ORDER):
                ax = axes[row_idx, col_idx]
                key = f"seed{self.seed}_{smote}_{model}"
                if key not in filtered:
                    ax.set_visible(False)
                    continue

                data = filtered[key]
                y_test = data["y_test"]
                y_proba = data["y_proba"]

                fpr, tpr, _ = roc_curve(y_test, y_proba)
                roc_auc_val = auc(fpr, tpr)

                ax.plot(fpr, tpr, color="darkorange", lw=2,
                        label=f"AUC = {roc_auc_val:.4f}")
                ax.plot([0, 1], [0, 1], color="navy", lw=1, linestyle="--")
                ax.set_xlim([0.0, 1.0])
                ax.set_ylim([0.0, 1.05])
                ax.set_xlabel("FPR" if row_idx == 1 else "")
                ax.set_ylabel("TPR" if row_idx == 0 else "")
                ax.set_title(f"{smote}\n{model}", fontsize=11)
                ax.legend(loc="lower right", fontsize=9)

        plt.tight_layout()
        plt.show()

    def plot_all(self):
        """Ejecuta los 3 reportes: classification reports, confusion matrices y ROC curves."""
        self.classification_reports()
        self.plot_confusion_matrices()
        self.plot_roc_curves()
