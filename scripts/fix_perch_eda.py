import json
import os

def rewrite_perch_eda():
    with open('notebooks/01_perch_baseline_eda.ipynb', 'r') as f:
        nb = json.load(f)

    # Find cells up to section 5
    new_cells = []
    for cell in nb['cells']:
        if len(cell['source']) > 0 and '## 5. Batch' in cell['source'][0]:
            break
        new_cells.append(cell)

    # Add new cells for loading embeddings
    new_cells.extend([
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "## 5. Batch Analysis (Full Dataset)\n",
                "\n",
                "Instead of computing embeddings on the fly, we will load the pre-computed embeddings for the entire training set."
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "processed_dir = '../data/processed'\n",
                "embeddings_path = os.path.join(processed_dir, 'perch_v1_embeddings.npz')\n",
                "metadata_path = os.path.join(processed_dir, 'train_with_perch_v1.csv')\n",
                "\n",
                "data = np.load(embeddings_path)\n",
                "embeddings = data['embeddings']\n",
                "indices = data['indices']\n",
                "\n",
                "train_df_perch = pd.read_csv(metadata_path)\n",
                "labels = train_df_perch['primary_label'].values\n",
                "\n",
                "print(f\"Loaded {len(embeddings)} embeddings.\")"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "## 6. Embedding Projection (PCA & t-SNE)\n",
                "\n",
                "Visualizing the high-dimensional embeddings in 2D."
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "from sklearn.manifold import TSNE\n",
                "from sklearn.decomposition import PCA\n",
                "import matplotlib.pyplot as plt\n",
                "import pandas as pd\n",
                "\n",
                "print(\"Running PCA...\")\n",
                "pca = PCA(n_components=50)\n",
                "pca_res = pca.fit_transform(embeddings)\n",
                "\n",
                "print(\"Running t-SNE...\")\n",
                "tsne = TSNE(n_components=2, random_state=42, init='pca', learning_rate='auto')\n",
                "tsne_res = tsne.fit_transform(pca_res)\n",
                "\n",
                "plt.figure(figsize=(12, 8))\n",
                "top_species = pd.Series(labels).value_counts().head(10).index\n",
                "for species in top_species:\n",
                "    mask = labels == species\n",
                "    plt.scatter(tsne_res[mask, 0], tsne_res[mask, 1], label=species, alpha=0.6, s=10)\n",
                "\n",
                "plt.title(\"t-SNE Projection of Perch Embeddings (Top 10 Species, Full Dataset)\")\n",
                "plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')\n",
                "plt.show()\n"
            ]
        }
    ])

    nb['cells'] = new_cells
    with open('notebooks/01_perch_baseline_eda.ipynb', 'w') as f:
        json.dump(nb, f, indent=1)

if __name__ == '__main__':
    rewrite_perch_eda()
