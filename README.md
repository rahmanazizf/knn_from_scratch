# K-Nearest Neighbors

Algoritma KNN merupakan salah satu algoritma supervised learning yang memproses data berdasarkan jarak (distance based). Dalam pengukuran jarak antar titik data, KNN dapat menggunakan perhitungan jarak Euclidean, Manhattan dll.

# Cara Kerja KNN

KNN dapat digunakan dalam kasus klasifikasi ataupun regresi. Dalam klasifikasi, KNN akan menghitung jarak setiap titik data training terhadap data testing kemudian mengurutkannya dari jarak terkecil ke terbesar. Huruf 'K' dalam KNN mewakili seberapa banyak titik data yang dipertimbangkan dalam pengambilan keputusan. Label output terbanyak akan digunakan untuk men-judge label data testing (majority vote). Dalam kasus regresi dilakukan hal serupa namun penentuan output dilakukan dengan perhitungan rata-rata.

# Deskripsi Modul KNN

Dalam proyek ini, modul KNN dalam file `knn.py` dikonstruksi dalam pemrograman berorientasi objek (object-oriented programming). Berikut adalah deskripsi fungsi-fungsi yang dimuat dalam class KNN.

1. `__init__`

berisi instance attribute dari class KNN

1. `euclidean`

fungsi untuk menghitung jarak antartitik dengan fungsi Euclidean

$$
d(p, q)=\sqrt{(p_1-q_1)^2+(p_2-q_2)^2+...+(p_n-q_n)^2}
$$

1. `manhattan`

fungsi untuk menghitung jarak antartitik dengan fungsi Manhattan

$$
d(p,q)=|p_1-q_1|+|p_2-q_2|+...+|p_n+q_n|
$$

1. `standardizer`

fungsi untuk menormalisasi nilai setiap fitur dalam dataset

$$
x_{stdvalue}=\frac{x-\mu}{\sigma}
$$

di mana $x$ adalah titik data dalam dataset, $\mu$ rata-rata, dan $\sigma$ standar deviasi

1. `calculate_distance`

menghitung jarak menggunakan `euclidean` atau `manhattan` bergantung pada input dari pengguna

1. `majority_vote`

menentukan indeks dari titik data terdekat sebanyak k kemudian mengembalikan nilai kelas terbanyak di antara titik-titik data tersebut.

1.