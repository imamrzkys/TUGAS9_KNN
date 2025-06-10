# KNN Customer Pattern Flask App

Aplikasi web berbasis Flask untuk mendeteksi pola pelanggan menggunakan algoritma K-Nearest Neighbors (KNN) dari data transaksi OnlineRetail.csv.

## Fitur
- Upload data pelanggan
- Analisis kemiripan pola belanja (KNN)
- Visualisasi PCA pelanggan mirip
- UI Bootstrap, siap deploy Railway

## Struktur Folder
```
knn_customer_pattern/
├── static/
│   └── style.css
├── templates/
│   ├── index.html
│   └── result.html
├── model/
│   └── knn_model.pkl
├── data/
│   └── OnlineRetail.csv
├── app.py
├── knn_logic.py
├── requirements.txt
├── runtime.txt
├── Procfile
└── README.md
```

## Cara Menjalankan
1. Install dependencies: `pip install -r requirements.txt`
2. Jalankan: `flask run`
3. Deploy ke Railway sesuai petunjuk.
