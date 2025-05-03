import cv2
from lane_detection import LaneDetector
from object_detection import ObjectDetector
import time
import numpy as np
import tkinter as tk
from tkinter import filedialog

def resize_image(image, target_width=640, target_height=480):
    """
    Görüntüyü hedef boyuta yeniden boyutlandırır
    """
    return cv2.resize(image, (target_width, target_height))

def draw_info_panel(image, fps, detected_objects):
    """
    Bilgi panelini görüntüye ekler
    """
    # Yarı saydam siyah panel
    overlay = image.copy()
    cv2.rectangle(overlay, (10, 10), (300, 100), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.7, image, 0.3, 0, image)
    
    # FPS bilgisi
    cv2.putText(image, f"FPS: {fps:.1f}", (20, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    
    # Tespit edilen nesneler
    y_pos = 60
    for obj, count in detected_objects.items():
        text = f"{obj}: {count}"
        cv2.putText(image, text, (20, y_pos),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        y_pos += 30

def select_video_file():
    """
    Video dosyası seçmek için dosya seçici açar
    """
    root = tk.Tk()
    root.withdraw()  # Ana pencereyi gizle
    file_path = filedialog.askopenfilename(
        title="Video dosyası seçin",
        filetypes=[
            ("Video dosyaları", "*.mp4 *.avi *.mov *.mkv"),
            ("Tüm dosyalar", "*.*")
        ]
    )
    return file_path

def main():
    """
    Ana uygulama fonksiyonu
    """
    # Video dosyası seç
    video_path = select_video_file()
    if not video_path:
        print("Video dosyası seçilmedi!")
        return
    
    # Video başlat
    cap = cv2.VideoCapture(video_path)
    
    # Video bağlantısını kontrol et
    if not cap.isOpened():
        print("Video açılamadı! Lütfen geçerli bir video dosyası seçin.")
        return
    
    # Video özelliklerini al
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    
    print(f"Video özellikleri:")
    print(f"Çözünürlük: {frame_width}x{frame_height}")
    print(f"FPS: {fps}")
    
    # Şerit ve nesne tespiti için sınıfları başlat
    lane_detector = LaneDetector()
    object_detector = ObjectDetector()
    
    print("Video işleme başlıyor...")
    
    # FPS hesaplama için değişkenler
    prev_time = 0
    current_fps = 0
    
    # Önceki tespit edilen nesneleri saklamak için
    prev_objects = {}
    
    while True:
        # Videodan kare al
        ret, frame = cap.read()
        if not ret:
            print("Video bitti!")
            break
            
        try:
            # FPS hesapla
            current_time = time.time()
            current_fps = 1 / (current_time - prev_time) if prev_time > 0 else 0
            prev_time = current_time
            
            # Görüntüyü yeniden boyutlandır
            frame = resize_image(frame)
            
            # Şerit tespiti yap
            frame_with_lanes = lane_detector.detect_lanes(frame.copy())
            
            # Nesne tespiti yap
            frame_with_objects, current_objects = object_detector.detect_objects(frame.copy())
            
            # Eğer tespit edilen nesneler değiştiyse bilgi panelini güncelle
            if current_objects != prev_objects:
                prev_objects = current_objects.copy()
            
            # Bilgi panelini ekle
            draw_info_panel(frame_with_objects, current_fps, current_objects)
            
            # Görüntüleri yan yana göster
            try:
                combined_frame = cv2.hconcat([frame_with_lanes, frame_with_objects])
            except cv2.error as e:
                print(f"Görüntü birleştirme hatası: {str(e)}")
                continue
            
            # Görüntüyü göster
            cv2.imshow('Otonom Araç Görüntü İşleme', combined_frame)
            
            # Video hızını ayarla (orijinal FPS'e göre)
            delay = int(1000/fps)  # milisaniye cinsinden gecikme
            if cv2.waitKey(delay) & 0xFF == ord('q'):
                break
            
        except Exception as e:
            print(f"Görüntü işleme hatası: {str(e)}")
            continue
    
    # Kaynakları serbest bırak
    cap.release()
    cv2.destroyAllWindows()
    print("Program sonlandırıldı.")

if __name__ == "__main__":
    main() 
#test