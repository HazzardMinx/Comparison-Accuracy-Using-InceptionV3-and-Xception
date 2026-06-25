# Deploy menggunakan flask - Perbandingan InceptionV3 vs Xception
import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

from flask import Flask, render_template, request
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import (
    preprocess_input as mobilenet_preprocess,
    decode_predictions
)
from tensorflow.keras.applications.xception import preprocess_input as xception_preprocess
from tensorflow.keras.applications.inception_v3 import preprocess_input as inception_preprocess
import matplotlib
matplotlib.use('Agg')  # Supaya bisa render tanpa display
import matplotlib.pyplot as plt
import uuid

app = Flask(__name__)

# ─── Load semua model ───────────────────────────────────────────────────────
model_xception   = load_model('4 Class Xception SGD 32-20.h5')
model_inception  = load_model('4 Class InceptionV3 SGD 32-20.h5')
mobilenet        = MobileNetV2(weights='imagenet')

# ─── Label kelas ────────────────────────────────────────────────────────────
LABEL_DECODE = ['Sepatu Boat', 'Sendal Clog', 'Sepatu Brogue', 'Sepatu Sneaker']

SHOE_LABELS = [
    'running_shoe', 'loafer', 'sandal', 'clog',
    'boot', 'sneaker', 'ballet_shoe', 'cowboy_boot',
    'wooden_shoe', 'flip_flop', 'rugby_boot', 'platform_shoe',
    'moccasin', 'oxford_shoe', 'shoe_shop',
]

# ─── Fungsi bantu ────────────────────────────────────────────────────────────
def is_shoe(image_path):
    """Filter awal: cek apakah gambar adalah sepatu menggunakan MobileNet."""
    img = load_img(image_path, target_size=(224, 224))
    img_array = np.array(img, dtype='float32')
    img_array = mobilenet_preprocess(img_array)
    img_array = img_array.reshape(1, 224, 224, 3)

    preds = mobilenet.predict(img_array, verbose=0)
    top_preds = decode_predictions(preds, top=5)[0]

    for _, label, _ in top_preds:
        if label in SHOE_LABELS:
            return True
    return False


def predict_model(model, preprocess_fn, image_path, target_size=(299, 299)):
    """Jalankan prediksi untuk satu model, kembalikan (nama_kelas, confidence, semua_prob)."""
    img = load_img(image_path, color_mode='rgb', target_size=target_size)
    img = np.array(img, dtype='float32')
    img = preprocess_fn(img)
    img = img.reshape(1, target_size[0], target_size[1], 3)

    pred = model.predict(img, verbose=0)
    idx  = np.argmax(pred)
    return LABEL_DECODE[idx], float(pred[0][idx]), pred[0].tolist()


def buat_grafik(probs_xception, probs_inception, save_dir):
    """Buat bar chart perbandingan confidence kedua model, simpan ke file PNG."""
    x = np.arange(len(LABEL_DECODE))
    width = 0.35

    fig, ax = plt.subplots(figsize=(9, 5))
    bars1 = ax.bar(x - width/2, [p * 100 for p in probs_xception],
                   width, label='Xception', color='#4C72B0', alpha=0.85)
    bars2 = ax.bar(x + width/2, [p * 100 for p in probs_inception],
                   width, label='InceptionV3', color='#DD8452', alpha=0.85)

    ax.set_xlabel('Kelas Sepatu', fontsize=12)
    ax.set_ylabel('Confidence (%)', fontsize=12)
    ax.set_title('Perbandingan Confidence: Xception vs InceptionV3', fontsize=13, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(LABEL_DECODE, rotation=15, ha='right')
    ax.set_ylim(0, 110)
    ax.legend()

    # Tambah label angka di atas bar
    for bar in bars1:
        ax.annotate(f'{bar.get_height():.1f}%',
                    xy=(bar.get_x() + bar.get_width() / 2, bar.get_height()),
                    xytext=(0, 3), textcoords='offset points', ha='center', fontsize=8)
    for bar in bars2:
        ax.annotate(f'{bar.get_height():.1f}%',
                    xy=(bar.get_x() + bar.get_width() / 2, bar.get_height()),
                    xytext=(0, 3), textcoords='offset points', ha='center', fontsize=8)

    plt.tight_layout()

    chart_filename = f'chart_{uuid.uuid4().hex[:8]}.png'
    chart_path = os.path.join(save_dir, chart_filename)
    plt.savefig(chart_path, dpi=120)
    plt.close()
    return chart_filename


# ─── Routes ─────────────────────────────────────────────────────────────────
@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'GET':
        return render_template('index.html')

    # Terima file gambar
    imagefile = request.files['imagefile']
    filename  = os.path.basename(imagefile.filename)
    save_dir  = os.path.join(app.static_folder, 'images')
    os.makedirs(save_dir, exist_ok=True)
    image_path = os.path.join(save_dir, filename)
    imagefile.stream.seek(0)
    imagefile.save(image_path)

    # Filter: bukan sepatu?
    if not is_shoe(image_path):
        return render_template(
            'predict.html',
            error="Gambar tidak terdeteksi sebagai sepatu.",
            filename=filename
        )

    # Prediksi kedua model
    nama_xception,  conf_xception,  probs_xception  = predict_model(
        model_xception,  xception_preprocess, image_path, target_size=(299, 299))
    nama_inception, conf_inception, probs_inception = predict_model(
        model_inception, inception_preprocess, image_path, target_size=(299, 299))

    # Tentukan pemenang
    if conf_xception >= conf_inception:
        pemenang = f"Xception lebih yakin → {nama_xception} ({conf_xception*100:.2f}%)"
    else:
        pemenang = f"InceptionV3 lebih yakin → {nama_inception} ({conf_inception*100:.2f}%)"

    # Buat grafik
    chart_filename = buat_grafik(probs_xception, probs_inception, save_dir)

    # Data tabel perbandingan
    tabel = []
    for i, kelas in enumerate(LABEL_DECODE):
        tabel.append({
            'kelas'        : kelas,
            'xception'     : f"{probs_xception[i]*100:.2f}%",
            'inception'    : f"{probs_inception[i]*100:.2f}%",
            'xception_raw' : round(probs_xception[i], 4),   # untuk progress bar
            'inception_raw': round(probs_inception[i], 4),  # untuk progress bar
        })

    return render_template(
        'predict.html',
        filename=filename,
        chart_filename=chart_filename,
        tabel=tabel,
        hasil_xception =f"{nama_xception} ({conf_xception*100:.2f}%)",
        hasil_inception=f"{nama_inception} ({conf_inception*100:.2f}%)",
        pemenang=pemenang,
    )


if __name__ == '__main__':
    app.run(debug=True)
