# Tugas Besar PCD

1. Syahra Rizky Ramadhani Nasution (1301220066)
2. Fadil Rafliansyah (1301220154)
3. Puguh Aiman Ariyanto (1301223038)

## Cara training model

jalankan file `Tubes_PCD_Training.ipynb` dan download file model yang sudah ditrain (`knn_model.pkl` dan `label_encoder.pkl`) dan simpan dalam folder ini.

## Cara menjalankan aplikasi

install library dari `requirements.txt`

```bash
pip install -r requirements.txt
```

jalankan aplikasi menggunakan streamlit

```bash
streamlit run app.py
```

buka browser dan akses http://localhost:8501, kemudian upload gambar produk dengan background putih yang ingin diprediksi atau bisa mengambil gambar dari folder `test`.
