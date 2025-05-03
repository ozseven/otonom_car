import cv2
from lane_detection import LaneDetector
from object_detection import ObjectDetector
import time
import numpy as np
import tkinter as tk
from tkinter import filedialog
import asyncio
import threading
from queue import Queue
import concurrent.futures

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
    root.withdraw()
    file_path = filedialog.askopenfilename(
        title="Video dosyası seçin",
        filetypes=[
            ("Video dosyaları", "*.mp4 *.avi *.mov *.mkv"),
            ("Tüm dosyalar", "*.*")
        ]
    )
    return file_path

class VideoProcessor:
    def __init__(self):
        self.frame_queue = Queue(maxsize=30)
        self.result_queue = Queue(maxsize=30)
        self.running = False
        self.lane_detector = LaneDetector()
        self.object_detector = ObjectDetector()
        print("Video işleyici başlatıldı")
        
    async def process_frame(self, frame):
        """
        Bir kareyi asenkron olarak işler
        """
        try:
            loop = asyncio.get_event_loop()
            
            # Şerit ve nesne tespitini paralel olarak yap
            with concurrent.futures.ThreadPoolExecutor() as executor:
                print("Şerit tespiti başlatılıyor...")
                lane_future = loop.run_in_executor(
                    executor, self.lane_detector.detect_lanes, frame.copy())
                
                print("Nesne tespiti başlatılıyor...")
                object_future = loop.run_in_executor(
                    executor, self.object_detector.detect_objects, frame.copy())
                
                # Sonuçları al
                print("Şerit tespiti sonuçları bekleniyor...")
                frame_with_lanes = await lane_future
                print("Şerit tespiti tamamlandı")
                
                print("Nesne tespiti sonuçları bekleniyor...")
                frame_with_objects, current_objects = await object_future
                print("Nesne tespiti tamamlandı")
                
            return frame_with_lanes, frame_with_objects, current_objects
        except Exception as e:
            print(f"Kare işleme hatası: {str(e)}")
            print(f"Hata detayı: {type(e).__name__}")
            import traceback
            print(f"Hata izi: {traceback.format_exc()}")
            return None, None, {}

    async def process_video(self, cap):
        """
        Video işleme döngüsü
        """
        print("Video işleme başlatılıyor...")
        prev_time = 0
        current_fps = 0
        frame_count = 0
        
        while self.running:
            ret, frame = cap.read()
            if not ret:
                print("Video bitti veya okunamadı!")
                break
                
            frame_count += 1
            if frame_count == 1:
                print(f"Video boyutu: {frame.shape}")
                
            current_time = time.time()
            current_fps = 1 / (current_time - prev_time) if prev_time > 0 else 0
            prev_time = current_time
            
            frame = resize_image(frame)
            self.frame_queue.put((frame, current_fps))
            
            # Kuyruk doluysa eski kareleri at
            if self.frame_queue.full():
                try:
                    self.frame_queue.get_nowait()
                except:
                    pass
                    
            await asyncio.sleep(0.001)  # CPU kullanımını azalt

    async def process_frames(self):
        """
        Kare işleme döngüsü
        """
        print("Kare işleme başlatılıyor...")
        while self.running:
            if not self.frame_queue.empty():
                frame, fps = self.frame_queue.get()
                result = await self.process_frame(frame)
                if result[0] is not None:  # Eğer işleme başarılıysa
                    self.result_queue.put((result, fps))
                
                # Kuyruk doluysa eski sonuçları at
                if self.result_queue.full():
                    try:
                        self.result_queue.get_nowait()
                    except:
                        pass
                        
            await asyncio.sleep(0.001)

async def main():
    """
    Ana uygulama fonksiyonu
    """
    print("Program başlatılıyor...")
    video_path = select_video_file()
    if not video_path:
        print("Video dosyası seçilmedi!")
        return
    
    print(f"Seçilen video: {video_path}")
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("Video açılamadı! Lütfen geçerli bir video dosyası seçin.")
        return
    
    # Video özelliklerini al ve yazdır
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    print(f"Video özellikleri:")
    print(f"Çözünürlük: {frame_width}x{frame_height}")
    print(f"FPS: {fps}")
    print(f"Toplam kare sayısı: {total_frames}")
    
    processor = VideoProcessor()
    processor.running = True
    
    # Video işleme ve kare işleme görevlerini başlat
    video_task = asyncio.create_task(processor.process_video(cap))
    frame_task = asyncio.create_task(processor.process_frames())
    
    prev_objects = {}
    print("Video işleme başlıyor...")
    
    try:
        while True:
            if not processor.result_queue.empty():
                (frame_with_lanes, frame_with_objects, current_objects), fps = processor.result_queue.get()
                
                if current_objects != prev_objects:
                    prev_objects = current_objects.copy()
                
                draw_info_panel(frame_with_objects, fps, current_objects)
                
                try:
                    combined_frame = cv2.hconcat([frame_with_lanes, frame_with_objects])
                    cv2.imshow('Otonom Araç Görüntü İşleme', combined_frame)
                except cv2.error as e:
                    print(f"Görüntü birleştirme hatası: {str(e)}")
                    continue
                
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    print("Kullanıcı tarafından durduruldu")
                    break
                    
            await asyncio.sleep(0.001)
            
    except Exception as e:
        print(f"Hata oluştu: {str(e)}")
    finally:
        print("Program sonlandırılıyor...")
        processor.running = False
        await asyncio.gather(video_task, frame_task, return_exceptions=True)
        cap.release()
        cv2.destroyAllWindows()
        print("Program sonlandırıldı.")

if __name__ == "__main__":
    asyncio.run(main()) 
#test