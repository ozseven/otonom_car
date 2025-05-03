import cv2
import numpy as np
from ultralytics import YOLO
import torch

class TrafficSignDetector:
    def __init__(self, model_path='yolov8n.pt'):
        """
        Trafik işareti tespiti için YOLO modelini başlatır
        """
        # GPU kullanılabilirliğini kontrol et
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print(f"Model {self.device} üzerinde çalışıyor")
        
        # Modeli GPU'ya yükle
        self.model = YOLO(model_path)
        self.model.to(self.device)
        self.model.conf = 0.5  # Güven eşiği
        
        # Trafik işareti sınıfları ve açıklamaları
        self.sign_descriptions = {
            'stop': 'Dur',
            'yield': 'Yol Ver',
            'speed_limit': 'Hız Limiti',
            'no_entry': 'Girişi Olmayan Yol',
            'no_parking': 'Park Etmek Yasak',
            'no_u_turn': 'U Dönüşü Yasak',
            'one_way': 'Tek Yön',
            'pedestrian_crossing': 'Yaya Geçidi',
            'school_zone': 'Okul Bölgesi',
            'traffic_light': 'Trafik Işığı',
            'warning': 'Uyarı İşareti',
            'priority_road': 'Ana Yol',
            'roundabout': 'Dönel Kavşak',
            'parking': 'Park Yeri',
            'bus_stop': 'Otobüs Durağı'
        }
        
    def detect_signs(self, image):
        """
        Görüntüdeki trafik işaretlerini tespit eder
        
        Args:
            image: İşlenecek görüntü
            
        Returns:
            Tespit edilen işaretlerin işaretlendiği görüntü
        """
        # YOLO ile nesne tespiti yap
        results = self.model(image, verbose=False)
        
        # Tespit edilen işaretleri görüntüye çiz
        for result in results:
            boxes = result.boxes
            for box in boxes:
                # Kutu koordinatlarını al
                x1, y1, x2, y2 = box.xyxy[0]
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                
                # Sınıf adını ve güven skorunu al
                class_id = int(box.cls[0])
                conf = float(box.conf[0])
                class_name = result.names[class_id]
                
                # İşaret açıklamasını al
                description = self.sign_descriptions.get(class_name, class_name)
                
                # Kutuyu çiz
                cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
                
                # Etiket metnini oluştur
                label = f'{description}: {conf:.2f}'
                
                # Etiket arka planı için metin boyutunu hesapla
                (text_width, text_height), _ = cv2.getTextSize(
                    label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
                
                # Etiket arka planını çiz
                cv2.rectangle(image, (x1, y1 - text_height - 10),
                            (x1 + text_width, y1), (0, 255, 0), -1)
                
                # Etiketi çiz
                cv2.putText(image, label, (x1, y1 - 5),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)
        
        return image 