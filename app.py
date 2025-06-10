 # Flask app main file

from flask import Flask, render_template, request, redirect, url_for, flash
import os
from knn_logic import process_and_predict, generate_training_plots

app = Flask(__name__)
app.secret_key = 'your_secret_key'  # Ganti dengan secret key aman
# Generate training plots saat startup (sekali saja)
import contextlib
with contextlib.redirect_stdout(open(os.devnull, 'w')):
    generate_training_plots()


UPLOAD_FOLDER = 'data'
ALLOWED_EXTENSIONS = {'csv'}
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Home page: upload form & result
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

import pandas as pd

@app.route('/', methods=['GET', 'POST'])
def index():
    # Ambil seluruh CustomerID unik dari data preprocessed
    df = pd.read_csv(os.path.join('data', 'OnlineRetail_preprocessed.csv'))
    customer_ids = df['CustomerID'].astype(int).unique().tolist()
    customer_ids = sorted(customer_ids)
    if request.method == 'POST':
        customer_id = request.form['customer_id']
        k_value = int(request.form.get('k_value', 5))
        try:
            neighbors, pca_plot, pca_overview, bar_path = process_and_predict(customer_id, k_value)
            sample_data = df.head(20).to_dict(orient='records')
            return render_template(
                'index.html',
                customer_id=customer_id,
                k_value=k_value,
                neighbors=neighbors,
                sample_data=sample_data,
                elbow_path='elbow_method.png',
                hist_path='histogram_features.png',
                silhouette_path='silhouette_score.png',
                pca_plot=pca_plot,
                pca_overview=pca_overview,
                bar_path=bar_path,
                customer_ids=customer_ids
            )
        except Exception as e:
            flash(str(e))
            sample_data = df.head(20).to_dict(orient='records')
            return render_template(
                'index.html',
                customer_ids=customer_ids,
                sample_data=sample_data,
                elbow_path='elbow_method.png',
                hist_path='histogram_features.png',
                silhouette_path='silhouette_score.png'
            )
    else:
        # GET: halaman awal, kirim sample_data & path grafik supaya template tidak error
        sample_data = df.head(20).to_dict(orient='records')
        return render_template(
            'index.html',
            customer_ids=customer_ids,
            sample_data=sample_data,
            elbow_path='elbow_method.png',
            hist_path='histogram_features.png',
            silhouette_path='silhouette_score.png'
        )

if __name__ == '__main__':
    app.run(debug=True)
