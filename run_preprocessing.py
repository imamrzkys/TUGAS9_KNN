# Jalankan preprocessing data mentah dan simpan hasilnya ke folder data/
import pandas as pd
from preprocessing import preprocess_data

# Path file mentah dan hasil
input_path = 'data/OnlineRetail.csv'
output_grouped = 'data/OnlineRetail_preprocessed.csv'
output_scaled = 'data/OnlineRetail_scaled.csv'

def main():
    df = pd.read_csv(input_path, encoding='ISO-8859-1')
    grouped, grouped_scaled, scaler = preprocess_data(df)
    grouped.to_csv(output_grouped, index=False)
    grouped_scaled.to_csv(output_scaled, index=False)
    print(f'Hasil preprocessing disimpan di: {output_grouped} dan {output_scaled}')

if __name__ == '__main__':
    main()
