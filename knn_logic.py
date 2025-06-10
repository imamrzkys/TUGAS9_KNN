# Logic for preprocessing, training, and KNN prediction
import pandas as pd
import numpy as np
import joblib
import os
from sklearn.neighbors import NearestNeighbors
from sklearn.decomposition import PCA
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

import sys


def train_and_save_knn_model(preprocessed_path, scaled_path, model_path):
    grouped = pd.read_csv(preprocessed_path)
    grouped_scaled = pd.read_csv(scaled_path)
    features = ['Total_Amount', 'Total_Quantity', 'Total_Transaksi', 'Rata_rata_Harga']
    knn = NearestNeighbors(n_neighbors=6, metric='euclidean')
    knn.fit(grouped_scaled[features])
    # Tidak perlu simpan scaler karena sudah scaled
    joblib.dump({'knn': knn, 'grouped': grouped, 'grouped_scaled': grouped_scaled}, model_path)


# Fungsi utama untuk Flask

def generate_training_plots():
    import matplotlib.pyplot as plt
    from sklearn.neighbors import NearestNeighbors
    from sklearn.metrics import silhouette_score
    import numpy as np
    import os
    preprocessed_path = os.path.join('data', 'OnlineRetail_preprocessed.csv')
    scaled_path = os.path.join('data', 'OnlineRetail_scaled.csv')
    grouped = pd.read_csv(preprocessed_path)
    grouped_scaled = pd.read_csv(scaled_path)
    features = ['Total_Amount', 'Total_Quantity', 'Total_Transaksi', 'Rata_rata_Harga']
    X = grouped_scaled[features].values

    # 1. Elbow Method Plot
    distortions = []
    K_range = range(2, 13)
    print("\n=== Elbow Method: K vs Distortion ===")
    print("+-----+--------------+")
    print("|  K  |  Distortion |")
    print("+-----+--------------+")
    for k in K_range:
        knn = NearestNeighbors(n_neighbors=k)
        knn.fit(X)
        dists, _ = knn.kneighbors(X)
        distortion = np.mean(dists[:, 1:])
        distortions.append(distortion)
        print(f"| {k:3d} | {distortion:10.4f} |")
    print("+-----+--------------+")
    plt.figure(figsize=(7,4))
    plt.plot(list(K_range), distortions, marker='o', color='#4F8EF7')
    plt.title('Elbow Method: Nilai K vs Distortion', fontsize=13)
    plt.xlabel('Nilai K Tetangga')
    plt.ylabel('Rata-rata Jarak ke Tetangga')
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.savefig('static/elbow_method.png', dpi=160)
    plt.close()

    # 2. Histogram distribusi fitur utama
    import seaborn as sns
    colors = sns.color_palette('Set2', 4)
    print("\n=== Statistik Distribusi Fitur ===")
    print("+----------------------+--------------+--------------+")
    print("|      Fitur           |    Mean      |     Std      |")
    print("+----------------------+--------------+--------------+")
    plt.figure(figsize=(11,6))
    for i, f in enumerate(features):
        plt.subplot(2,2,i+1)
        data = grouped[f]
        c = colors[i]
        # Log scale for large, skewed features
        if f in ["Total_Amount", "Total_Quantity"] and (data > 0).any():
            plt.hist(data[data > 0], bins=20, color=c, alpha=0.8, edgecolor='white', log=False)
            plt.xscale('log')
            plt.xlabel(f"{f} (log scale)")
        else:
            plt.hist(data, bins=20, color=c, alpha=0.8, edgecolor='white')
            plt.xlabel(f)
        mean, std = data.mean(), data.std()
        print(f"| {f:20s} | {mean:10.2f} | {std:10.2f} |")
        plt.title(f"{f}\n(mean={mean:.1f}, std={std:.1f})", fontsize=11)
        plt.ylabel('Jumlah Customer')
        plt.grid(True, linestyle='--', alpha=0.3)
    print("+----------------------+--------------+--------------+")
    plt.tight_layout()
    plt.savefig('static/histogram_features.png', dpi=160)
    plt.close()

    # 3. Silhouette Score vs K
    from sklearn.cluster import KMeans
    sil_scores = []
    K_range2 = range(2, 9)
    print("\n=== Silhouette Score vs K (KMeans) ===")
    print("+-----+--------------------+")
    print("|  K  | Silhouette Score   |")
    print("+-----+--------------------+")
    for k in K_range2:
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = kmeans.fit_predict(X)
        sil = silhouette_score(X, labels)
        sil_scores.append(sil)
        print(f"| {k:3d} | {sil:16.4f} |")
    print("+-----+--------------------+")
    plt.figure(figsize=(7,4))
    plt.plot(list(K_range2), sil_scores, marker='o', color='#F7B32B')
    plt.title('Silhouette Score vs Jumlah Cluster (KMeans)', fontsize=13)
    plt.xlabel('Jumlah Cluster (K)')
    plt.ylabel('Silhouette Score')
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.savefig('static/silhouette_score.png', dpi=160)
    plt.close()


def process_and_predict(customer_id, k_value=5):
    preprocessed_path = os.path.join('data', 'OnlineRetail_preprocessed.csv')
    scaled_path = os.path.join('data', 'OnlineRetail_scaled.csv')
    model_path = os.path.join('model', 'knn_model.pkl')
    if not os.path.exists(model_path):
        train_and_save_knn_model(preprocessed_path, scaled_path, model_path)
    model = joblib.load(model_path)
    knn = model['knn']
    grouped = model['grouped']
    grouped_scaled = model['grouped_scaled']
    features = ['Total_Amount', 'Total_Quantity', 'Total_Transaksi', 'Rata_rata_Harga']
    # Cari index customer
    customer_id = float(customer_id)
    idx = grouped[grouped['CustomerID'] == customer_id].index
    if len(idx) == 0:
        raise Exception(f'Customer ID {customer_id} tidak ditemukan di data!')
    idx = idx[0]
    X = grouped_scaled[features]
    X_pred = pd.DataFrame([X.iloc[idx].values], columns=features)
    distances, indices = knn.kneighbors(X_pred, n_neighbors=k_value+1)
    neighbors = []
    for i, dist in zip(indices[0][1:], distances[0][1:]):  # skip self
        neighbors.append({
            'CustomerID': int(grouped.iloc[i]['CustomerID']),
            'Distance': float(dist)
        })
    # Visualisasi PCA SELURUH DATA (overview)
    # (print hasil analisis ke terminal dipindah ke bawah, setelah variabel path di-set)

    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X)
    plt.figure(figsize=(8,6))
    colors_overview = plt.cm.rainbow(np.linspace(0,1,len(X_pca)))
    plt.scatter(X_pca[:,0], X_pca[:,1], color=colors_overview, alpha=0.7, s=55, edgecolor='white', linewidth=1.3, label=None, zorder=1)
    plt.title('PCA Seluruh Customer', fontsize=17, fontweight='bold')
    plt.xlabel('PCA 1', fontsize=13)
    plt.ylabel('PCA 2', fontsize=13)
    plt.grid(True, linestyle='--', alpha=0.4)
    plt.tight_layout()
    overview_path = f'static/pca_overview.png'
    plt.savefig(overview_path, dpi=180)
    plt.close()

    # Visualisasi PCA dengan highlight neighbors & customer utama
    plt.figure(figsize=(8,6))
    bg_colors = plt.cm.spring(np.linspace(0.2,0.9,len(X_pca)))
    plt.scatter(X_pca[:,0], X_pca[:,1], color=bg_colors, alpha=0.45, s=60, label=None, edgecolor='white', linewidth=1.2, zorder=1)
    plt.scatter(X_pca[idx,0], X_pca[idx,1], color='#FF4F5A', s=260, marker='*', label='Customer Utama', edgecolor='white', linewidth=3.2, zorder=15)
    neighbor_colors = plt.cm.autumn(np.linspace(0.15,0.85,len(indices[0][1:])))
    for n, (i, col) in enumerate(zip(indices[0][1:], neighbor_colors)):
        plt.scatter(X_pca[i,0], X_pca[i,1], color=col, s=150, marker='o', edgecolor='white', linewidth=2.2, label='Tetangga' if n==0 else "", zorder=10)
    plt.legend(fontsize=13, loc='upper right', frameon=True)
    plt.title('PCA: Customer & Neighbors', fontsize=18, fontweight='bold')
    plt.xlabel('PCA 1', fontsize=14)
    plt.ylabel('PCA 2', fontsize=14)
    plt.grid(True, linestyle='--', alpha=0.4)
    plt.tight_layout()
    highlight_path = f'static/pca_highlight_{int(customer_id)}.png'
    plt.savefig(highlight_path, dpi=180)
    plt.close()

    # Bar chart: Jarak ke neighbors
    neighbor_labels = [str(int(grouped.iloc[i]['CustomerID'])) for i in indices[0][1:]]
    neighbor_distances = [float(dist) for dist in distances[0][1:]]
    plt.figure(figsize=(7,4))
    colors = plt.cm.viridis(np.linspace(0, 1, len(neighbor_labels)))
    bars = plt.bar(neighbor_labels, neighbor_distances, color=colors, edgecolor='black')
    plt.xlabel('Customer ID Tetangga')
    plt.ylabel('Jarak (Euclidean)')
    plt.title('Jarak ke 5 Tetangga Terdekat')
    plt.tight_layout()
    for bar in bars:
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height(), f'{bar.get_height():.3f}', ha='center', va='bottom', fontsize=10)
    bar_path = f'static/bar_neighbors_{int(customer_id)}.png'
    plt.savefig(bar_path)
    plt.close()

    # === Cetak hasil analisis ke terminal ===
    print(f"\n[Analisis CustomerID: {int(customer_id)}]")
    print("Customer mirip:", [n['CustomerID'] for n in neighbors])
    print("Rank\tCustomerID\tDistance")
    for idx, n in enumerate(neighbors, 1):
        print(f"{idx}\t{n['CustomerID']}\t{n['Distance']:.4f}")
    print(f"Visualisasi PCA: static/{os.path.basename(highlight_path)}")
    print(f"PCA Overview: static/{os.path.basename(overview_path)}")
    print(f"Bar Chart Neighbors: static/{os.path.basename(bar_path)}")
    # === Akhir cetak ===
    return neighbors, os.path.basename(highlight_path), os.path.basename(overview_path), os.path.basename(bar_path)



if __name__ == "__main__":
    print("=== Analisis KNN Pola Pelanggan ===")
    # 1. Cetak statistik training (Elbow, Statistik Fitur, Silhouette)
    generate_training_plots()
    # 2. Analisis KNN untuk 10 customer pertama
    preprocessed_path = os.path.join('data', 'OnlineRetail_preprocessed.csv')
    grouped = pd.read_csv(preprocessed_path)
    all_customer_ids = grouped['CustomerID'].unique()
    print(f"Total pelanggan: {len(all_customer_ids)}")
    print("\nMenampilkan hasil KNN untuk 10 pelanggan pertama:")
    for n, customer_id in enumerate(all_customer_ids[:10]):
        try:
            process_and_predict(customer_id)
        except Exception as e:
            print(f"Error pada CustomerID {customer_id}: {e}")
    print("\nSelesai. Untuk seluruh customer, edit jumlah pada all_customer_ids[:10] jika ingin lebih banyak.")
