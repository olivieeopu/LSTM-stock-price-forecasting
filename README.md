# Prediksi Harga Saham dengan LSTM

Eksperimen time series forecasting untuk memprediksi harga penutupan saham IBM dan Facebook menggunakan lima hari perdagangan terakhir. Proyek membandingkan baseline LSTM dengan dua konfigurasi modifikasi.

**Tech stack:** Python, pandas, NumPy, TensorFlow/Keras, Scikit-learn, Matplotlib, dan Seaborn.

## Latar Belakang

Data harga saham memiliki urutan waktu yang perlu dipertahankan dalam proses pemodelan. LSTM merupakan salah satu arsitektur neural network yang dirancang untuk mempelajari hubungan dalam data sekuensial.

Proyek ini mengevaluasi pengaruh perubahan konfigurasi LSTM terhadap error prediksi harga penutupan saham. Eksperimen mencakup perubahan jumlah layer dan unit, activation function, optimizer, serta penggunaan Dropout.

## Tujuan

- Mengeksplorasi data historis saham IBM dan Facebook.
- Membentuk sampel supervised learning menggunakan sliding window.
- Mengembangkan baseline LSTM dan dua konfigurasi modifikasi.
- Membandingkan performa model menggunakan RMSE, MAE, dan MAPE.
- Mengidentifikasi konfigurasi dengan error terendah dalam eksperimen.

## Dataset

Dataset terdiri dari data historis dua saham:

| Dataset | Jumlah Observasi Awal |
|---|---:|
| IBM | 14.663 |
| Facebook (FB) | 1.980 |

Kolom `Date` digunakan untuk pengurutan dan pembagian data, sedangkan `Close` menjadi input harga sekaligus target prediksi.

Prediksi ditujukan untuk **hari perdagangan berikutnya**, bukan hari kalender berikutnya. Akhir pekan dan hari libur bursa tidak ditambahkan sebagai observasi buatan.

## Alur Analisis

1. Exploratory Data Analysis (EDA).
2. Pemeriksaan missing values dan duplikasi.
3. Pengurutan serta pembagian data secara kronologis.
4. Normalisasi harga menggunakan MinMaxScaler.
5. Pembentukan sliding window.
6. Pelatihan baseline LSTM.
7. Eksperimen Modifikasi 1 dan Modifikasi 2.
8. Evaluasi serta visualisasi actual vs predicted.

### Data Preprocessing

Data diurutkan berdasarkan tanggal dan dibagi secara kronologis agar data uji berasal dari periode setelah data latih.

MinMaxScaler di-fit pada bagian data latih, kemudian digunakan untuk mentransformasi data uji. Bagian akhir data latih digunakan sebagai validation set.

### Sliding Window

Konfigurasi input dan target:

- **Window size:** 5 hari perdagangan.
- **Forecast horizon:** 1 hari perdagangan.
- **Input feature:** harga penutupan.
- **Target:** harga penutupan hari perdagangan berikutnya.

```text
Input                        Target
[t-4, t-3, t-2, t-1, t]  →  [t+1]
```

Window yang saling tumpang tindih digunakan untuk membentuk sampel prediksi satu langkah ke depan.

## Arsitektur Model

### Baseline LSTM

```text
LSTM (50 units, ReLU)
Dense (1)
```

- **Optimizer:** SGD
- **Loss:** Mean Squared Error (MSE)

### Modifikasi 1

```text
LSTM (64 units, tanh, return_sequences=True)
LSTM (32 units, tanh)
Dense (16 units, ReLU)
Dense (1)
```

- **Optimizer:** Adam
- **Loss:** MSE

Konfigurasi ini menggunakan dua layer LSTM dan satu hidden Dense layer. Activation function dan optimizer juga berubah dibandingkan baseline.

### Modifikasi 2

```text
LSTM (64 units, tanh, return_sequences=True)
Dropout (0.1)
LSTM (32 units, tanh)
Dense (16 units, ReLU)
Dense (1)
```

- **Optimizer:** Adam
- **Loss:** MSE

Modifikasi 2 menambahkan Dropout sebesar 0,1 pada konfigurasi Modifikasi 1 untuk mengevaluasi pengaruh regularisasi.

## Evaluasi Model

Prediksi dikembalikan ke skala harga asli sebelum dievaluasi menggunakan:

- **RMSE:** memberikan penalti lebih besar pada error yang besar.
- **MAE:** rata-rata selisih absolut antara prediksi dan harga aktual.
- **MAPE:** rata-rata error absolut relatif terhadap harga aktual, dalam persen.

RMSE dan MAE dinyatakan dalam **USD**. Nilai yang lebih rendah menunjukkan error yang lebih kecil.

## Hasil Eksperimen

Tabel berikut merangkum hasil evaluasi data uji yang tersimpan dalam notebook.

| Dataset | Konfigurasi | RMSE (USD) | MAE (USD) | MAPE |
|---|---|---:|---:|---:|
| IBM | Baseline | 3,58 | 2,33 | 1,80% |
| IBM | **Modifikasi 1** | **2,71** | **1,76** | **1,35%** |
| IBM | Modifikasi 2 | 5,38 | 3,95 | 3,01% |
| Facebook | Baseline | 8,28 | 7,25 | 3,75% |
| Facebook | **Modifikasi 1** | **5,01** | **4,06** | **2,16%** |
| Facebook | Modifikasi 2 | 12,52 | 10,84 | 5,73% |

## Temuan Utama

### Modifikasi 1 menghasilkan error terendah

Pada kedua dataset, Modifikasi 1 memperoleh RMSE, MAE, dan MAPE terendah di antara konfigurasi yang diuji.

- **IBM:** RMSE turun dari 3,58 menjadi 2,71.
- **Facebook:** RMSE turun dari 8,28 menjadi 5,01.

### Penambahan Dropout belum meningkatkan performa

Modifikasi 2 menghasilkan error data uji lebih tinggi daripada baseline maupun Modifikasi 1.

Hasil ini menunjukkan bahwa penambahan regularisasi tidak otomatis meningkatkan performa forecasting. Manfaatnya perlu dievaluasi berdasarkan konfigurasi dan dataset yang digunakan.

### Pengaruh setiap perubahan belum dapat dipisahkan

Modifikasi 1 mengubah beberapa komponen sekaligus, termasuk jumlah layer, jumlah unit, activation function, dan optimizer.

Karena itu, peningkatan hasil belum dapat dikaitkan hanya dengan satu perubahan tertentu. Eksperimen terpisah diperlukan untuk mengukur kontribusi masing-masing komponen.

## Kesimpulan

Modifikasi 1 menjadi konfigurasi terbaik dalam eksperimen ini berdasarkan ketiga metrik evaluasi pada masing-masing dataset.

Perbandingan dengan Modifikasi 2 menunjukkan pentingnya mengevaluasi perubahan arsitektur melalui hasil pengujian, termasuk ketika perubahan tersebut bertujuan mengurangi overfitting.

## Batasan dan Pengembangan Selanjutnya

- Model hanya menggunakan riwayat harga penutupan, tanpa fitur volume, berita, atau indikator eksternal.
- Evaluasi berfokus pada prediksi satu hari perdagangan ke depan, bukan prediksi beberapa hari sekaligus.
- Eksperimen belum menyertakan naive baseline, seperti menggunakan harga hari ini sebagai prediksi harga besok.
- Hasil yang dilaporkan belum menunjukkan kestabilan performa melalui pengulangan beberapa random seed.
- Pengembangan berikutnya dapat mencakup walk-forward evaluation dan eksperimen perubahan konfigurasi satu per satu.
- Evaluasi error harga belum mencakup performa strategi trading.

## Penulis

Fransciska Olivia Putri Warae  
Mahasiswa Data Science, BINUS University
