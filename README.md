# Otonom Araç Görüntü İşleme Projesi

Bu proje, otonom araçlar için temel görüntü işleme özelliklerini içerir. Şerit tespiti ve nesne tanıma (yaya, araç, trafik işaretleri vb.) özelliklerini barındırır.

## Özellikler

- Şerit Tespiti (Canny Edge Detection ve Hough Transform kullanarak)
- Nesne Tespiti (YOLO v8 kullanarak)
- Gerçek Zamanlı Görüntü İşleme

## Kurulum

1. Gerekli kütüphaneleri yükleyin:
```bash
pip install -r requirements.txt
```

2. YOLO modelini indirin:
```bash
# YOLO modeli ilk çalıştırmada otomatik olarak indirilecektir
```

## Kullanım

Programı çalıştırmak için:
```bash
python main.py
```

- Program başladığında kamera görüntüsü açılacaktır
- Sol tarafta şerit tespiti, sağ tarafta nesne tespiti gösterilecektir
- Çıkmak için 'q' tuşuna basın

## Geliştirme

Proje şu ana modülleri içerir:

- `lane_detection.py`: Şerit tespiti için gerekli fonksiyonları içerir
- `object_detection.py`: YOLO tabanlı nesne tespiti için gerekli fonksiyonları içerir
- `main.py`: Ana uygulama dosyası

## Notlar

- Şerit tespiti için Canny Edge Detection ve Hough Transform kullanılmıştır
- Nesne tespiti için YOLO v8 modeli kullanılmıştır
- Performans iyileştirmeleri için parametreler ayarlanabilir 