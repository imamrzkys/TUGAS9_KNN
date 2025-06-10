# preprocessing.py
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

def preprocess_data(df):
    """
    Bersihkan dan ekstrak fitur pelanggan dari data mentah transaksi.
    Return: grouped (fitur asli), grouped_scaled (fitur ternormalisasi), scaler
    """
    # Hapus data tanpa CustomerID
    df = df.dropna(subset=['CustomerID'])
    df = df[df['Quantity'] > 0]
    df = df[df['UnitPrice'] > 0]
    df['Total_Amount'] = df['Quantity'] * df['UnitPrice']
    grouped = df.groupby('CustomerID').agg({
        'Total_Amount': 'sum',
        'Quantity': 'sum',
        'InvoiceNo': 'nunique',
        'UnitPrice': 'mean'
    }).reset_index()
    grouped.rename(columns={
        'Quantity': 'Total_Quantity',
        'InvoiceNo': 'Total_Transaksi',
        'UnitPrice': 'Rata_rata_Harga'
    }, inplace=True)
    # Normalisasi
    features = ['Total_Amount', 'Total_Quantity', 'Total_Transaksi', 'Rata_rata_Harga']
    scaler = StandardScaler()
    grouped_scaled = grouped.copy()
    grouped_scaled[features] = scaler.fit_transform(grouped[features])
    return grouped, grouped_scaled, scaler
