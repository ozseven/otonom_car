from ultralytics import YOLO
import cv2
import numpy as np
from collections import Counter
import torch

class ObjectDetector:
    # Varsayılan modeli yolov8l.pt olarak değiştir
    def __init__(self, model_path='yolov8n.pt'): 
        """
        YOLO modelini başlatır
        
        Args:
            model_path: YOLO model dosyasının yolu
        """
        # GPU kullanılabilirliğini kontrol et
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print(f"Kullanılan cihaz: {self.device}")
        
        # Modeli yükle ve GPU'ya taşı
        self.model = YOLO(model_path)
        self.model.to(self.device)
        
        # Model ayarları
        self.model.conf = 0.5  # Güven eşiği
        self.model.iou = 0.45  # IOU eşiği
        
    def detect_objects(self, image):
        """
        Görüntüdeki nesneleri tespit eder
        
        Args:
            image: İşlenecek görüntü
            
        Returns:
            Tespit edilen nesnelerin işaretlendiği görüntü ve nesne sayıları
        """
        # YOLO ile nesne tespiti yap
        results = self.model(image, verbose=False)
        
        # Tespit edilen nesneleri say
        detected_objects = Counter()
        
        # Tespit edilen nesneleri görüntüye çiz
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
                
                # Nesne sayısını güncelle
                detected_objects[class_name] += 1
                
                # Kutuyu çiz
                cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
                
                # Etiket metnini oluştur
                label = f'{class_name}: {conf:.2f}'
                
                # Etiket arka planı için metin boyutunu hesapla
                (text_width, text_height), _ = cv2.getTextSize(
                    label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
                
                # Etiket arka planını çiz
                cv2.rectangle(image, (x1, y1 - text_height - 10),
                            (x1 + text_width, y1), (0, 255, 0), -1)
                
                # Etiketi çiz
                cv2.putText(image, label, (x1, y1 - 5),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)
        
        return image, dict(detected_objects)
