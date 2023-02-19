# K-Nearest Neighbors


## Cara Kerja KNN
Algoritma KNN merupakan salah satu algoritma supervised learning yang memproses data berdasarkan jarak (distance based). Dalam pengukuran jarak antar titik data, KNN dapat menggunakan perhitungan jarak Euclidean, Manhattan dll.

KNN dapat digunakan dalam kasus klasifikasi ataupun regresi. Dalam klasifikasi, KNN akan menghitung jarak setiap titik data training terhadap data testing kemudian mengurutkannya dari jarak terkecil ke terbesar. Huruf 'K' dalam KNN mewakili seberapa banyak titik data yang dipertimbangkan dalam pengambilan keputusan. Label output terbanyak akan digunakan untuk men-judge label data testing (majority vote). Dalam kasus regresi dilakukan hal serupa namun penentuan output dilakukan dengan perhitungan rata-rata.

## Deskripsi Modul KNN

Dalam proyek ini, modul KNN dalam file `knn.py` dikonstruksi dalam pemrograman berorientasi objek (object-oriented programming). Berikut adalah deskripsi fungsi-fungsi yang dimuat dalam class KNN.

1. `__init__`

berisi instance attribute dari class KNN

2. `euclidean`

fungsi untuk menghitung jarak antartitik dengan fungsi Euclidean

$$
d(p, q)=\sqrt{(p_1-q_1)^2+(p_2-q_2)^2+...+(p_n-q_n)^2}
$$

3. `manhattan`

fungsi untuk menghitung jarak antartitik dengan fungsi Manhattan

$$
d(p,q)=|p_1-q_1|+|p_2-q_2|+...+|p_n+q_n|
$$

4. `standardizer`

fungsi untuk menormalisasi nilai setiap fitur dalam dataset

$$
x_{stdvalue}=\frac{x-\mu}{\sigma}
$$

di mana $x$ adalah titik data dalam dataset, $\mu$ rata-rata, dan $\sigma$ standar deviasi

5. `calculate_distance`

menghitung jarak menggunakan `euclidean` atau `manhattan` bergantung pada input dari pengguna

6. `majority_vote`

menentukan indeks dari titik data terdekat sebanyak k kemudian mengembalikan nilai kelas terbanyak di antara titik-titik data tersebut.

7. `neighbor_index`

menentukan indeks k-titik data terdekat dari data test

8. `calculate_knn`

menentukan kelas/output dan indeks k-titik-titik data terdekat dari 1 titik data test

9. `transform`

serupa dengan `calculate_knn` tetapi dapat digunakan untuk lebih dari 1 titik data test

## Cara Menggunakan Modul

Jika library numpy belum tersedia dalam local/global environment direktori kerja, lakukan instalasi library numpy terlebih dahulu dengan perintah berikut.

```
python -m pip install numpy
```
Setelah modul terinstall, modul `knn.py` dapat digunakan langsung dalam script ataupun jupyter notebook.