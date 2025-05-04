import cv2
from lane_detection import LaneDetector
from object_detection import ObjectDetector
import time
import numpy as np
import tkinter as tk
from tkinter import filedialog, messagebox # Import messagebox
import asyncio
import threading
import os # Add os import
from queue import Queue
import concurrent.futures

# FPS sınırı
MAX_FPS = 60
FRAME_TIME = 1.0 / MAX_FPS

def resize_image(image, target_width=640, target_height=480):
    """
    Görüntüyü hedef boyuta yeniden boyutlandırır
    """
    
    resized = cv2.resize(image, (target_width, target_height))
    return resized

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

def select_input_path():
    """
    Video dosyası veya resim içeren klasör seçmek için dosya/klasör seçici açar
    """
    root = tk.Tk()
    root.withdraw()
    
    # Ask user to choose file or directory
    choice = messagebox.askquestion("Giriş Seçimi", "Video dosyası mı yoksa resim klasörü mü seçmek istersiniz?", icon='question', detail="Video için 'Evet', Klasör için 'Hayır' seçin.")
    
    path = None # Initialize path
    if choice == 'yes': # User chose video file
        path = filedialog.askopenfilename(
            title="Video dosyası seçin",
            filetypes=[
                ("Video dosyaları", "*.mp4 *.avi *.mov *.mkv"),
                ("Tüm dosyalar", "*.*")
            ]
        )
    elif choice == 'no': # User chose image directory
        path = filedialog.askdirectory(
            title="Resim klasörü seçin"
        )
    # If user closes the messagebox, choice might be something else or path remains None
        
    return path

class VideoProcessor:
    def __init__(self):
        self.frame_queue = Queue(maxsize=30)
        self.result_queue = Queue(maxsize=30)
        self.running = False
        self.lane_detector = LaneDetector()
        self.object_detector = ObjectDetector()
        # self.traffic_sign_detector = TrafficSignDetector() # Kaldırıldı
        self.rotate_flag = 0  # 0: no rotation, 1: 90 deg, 2: 180 deg, 3: 270 deg
        print("Video işleyici başlatıldı (Trafik İşareti Tanıma Devre Dışı)")
        
    async def process_frame(self, frame):
        """
        Bir kareyi asenkron olarak işler
        """
        try:
            loop = asyncio.get_event_loop()
            
            # Şerit, nesne ve trafik işareti tespitini paralel olarak yap
            with concurrent.futures.ThreadPoolExecutor() as executor:
                print("Şerit tespiti başlatılıyor...")
                lane_future = loop.run_in_executor(
                    executor, self.lane_detector.detect_lanes, frame.copy())
                
                print("Nesne tespiti başlatılıyor...")
                object_future = loop.run_in_executor(
                    executor, self.object_detector.detect_objects, frame.copy())
                
                # print("Trafik işareti tespiti başlatılıyor...") # Kaldırıldı
                # traffic_sign_future = loop.run_in_executor(
                #     executor, self.traffic_sign_detector.detect_signs, frame.copy()) # Kaldırıldı
                
                # Sonuçları al
                print("Şerit tespiti sonuçları bekleniyor...")
                frame_with_lanes = await lane_future
                print("Şerit tespiti tamamlandı")
                
                print("Nesne tespiti sonuçları bekleniyor...")
                frame_with_objects, current_objects = await object_future
                print("Nesne tespiti tamamlandı")
                
                # print("Trafik işareti tespiti sonuçları bekleniyor...") # Kaldırıldı
                # frame_with_traffic_signs = await traffic_sign_future # Kaldırıldı
                # print("Trafik işareti tespiti tamamlandı") # Kaldırıldı
                
            # Sadece şerit ve nesne sonuçlarını döndür
            return frame_with_lanes, frame_with_objects, current_objects 
        except Exception as e:
            print(f"Kare işleme hatası: {str(e)}")
            print(f"Hata detayı: {type(e).__name__}")
            import traceback
            print(f"Hata izi: {traceback.format_exc()}")
            # Dönüş değerini güncelle
            return None, None, {} 

    async def process_video(self, input_source, is_video):
        """
        Video veya resim dizisi işleme döngüsü
        """
        print(f"Giriş işleme başlatılıyor... ({'Video' if is_video else 'Resim Dizisi'})")
        prev_time = 0
        current_fps = 0
        frame_count = 0
        image_index = 0 # For image sequence processing
        
        while self.running:
            try:
                frame = None
                ret = False
                
                if is_video:
                    # Read frame from video capture
                    ret, frame = input_source.read()
                    if not ret:
                        print("Video bitti veya okunamadı!")
                        break
                else:
                    # Read frame from image list
                    if image_index < len(input_source):
                        image_path = input_source[image_index]
                        frame = cv2.imread(image_path)
                        if frame is not None:
                            ret = True
                            print(f"İşleniyor: {os.path.basename(image_path)} ({image_index + 1}/{len(input_source)})")
                            image_index += 1
                        else:
                            print(f"Uyarı: Resim okunamadı veya bozuk: {image_path}")
                            image_index += 1 # Skip corrupted image
                            continue # Skip this iteration
                    else:
                        print("Resim dizisi bitti!")
                        self.running = False # Stop processing after last image
                        break # End of image sequence

                if not ret or frame is None:
                     # Should not happen based on logic above, but as a safeguard
                    print("Kare alınamadı.")
                    break

                frame_count += 1
                if frame_count == 1 and is_video: # Print dimensions only once for video
                    print(f"Video boyutu: {frame.shape}")
                elif frame_count == 1 and not is_video:
                     print(f"İlk resim boyutu: {frame.shape}")
                    
                current_time = time.time()
                elapsed = current_time - prev_time
                
                # FPS sınırlaması
                if elapsed < FRAME_TIME:
                    await asyncio.sleep(FRAME_TIME - elapsed)
                
                current_fps = 1 / (time.time() - prev_time) if prev_time > 0 else 0
                prev_time = time.time()

                # Rotate frame based on the processor's flag BEFORE resizing
                if self.rotate_flag == 1:
                    frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
                elif self.rotate_flag == 2:
                    frame = cv2.rotate(frame, cv2.ROTATE_180)
                elif self.rotate_flag == 3:
                    frame = cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE)
                
                frame = resize_image(frame)
                self.frame_queue.put((frame, current_fps))

                # Add 5-second delay only if processing images
                if not is_video:
                    print("Sonraki resim için 5 saniye bekleniyor...")
                    await asyncio.sleep(5) 
                
                # Kuyruk doluysa eski kareleri at
                if self.frame_queue.full():
                    try:
                        self.frame_queue.get_nowait()
                    except:
                        pass
                        
            except Exception as e:
                print(f"Video işleme hatası: {str(e)}")
                continue

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

    input_path = None
    print(input_path)
    for i in range(10):
        print(f"Giriş yolu seçimi denemesi: {i+1}")
        input_path = select_input_path() # Use the new selection function
        if input_path:
            print(f"Seçilen yol: {input_path}")
            break
        else:
             # User might have cancelled the selection dialog
             print("Seçim iptal edildi veya geçersiz.")
             # Optional: Ask again or exit? For now, let's retry.
             print(f"Dosya veya klasör seçilmedi, {i+1}. deneme. 0.5 saniye sonra tekrar deneniyor...")
             await asyncio.sleep(0.5)


    if not input_path:
        print("10 denemeden sonra hala dosya veya klasör seçilmedi. Program sonlandırılıyor.")
        return

    input_source = None
    is_video = False
    cap = None # Initialize cap to None

    if os.path.isfile(input_path):
        print(f"Seçilen video dosyası: {input_path}")
        cap = cv2.VideoCapture(input_path)
        if not cap.isOpened():
            print("Video açılamadı! Lütfen geçerli bir video dosyası seçin.")
            return
        input_source = cap
        is_video = True
        
        # Video özelliklerini al ve yazdır
        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        print(f"Video özellikleri:")
        print(f"Çözünürlük: {frame_width}x{frame_height}")
        print(f"FPS: {fps}")
        print(f"Toplam kare sayısı: {total_frames}")

    elif os.path.isdir(input_path):
        print(f"Seçilen resim klasörü: {input_path}")
        # Get sorted list of image files
        image_extensions = ('.png', '.jpg', '.jpeg', '.bmp', '.tiff', '.webp') # Added webp
        try:
            all_files = os.listdir(input_path)
            image_files = sorted([
                os.path.join(input_path, f) for f in all_files
                if os.path.isfile(os.path.join(input_path, f)) and f.lower().endswith(image_extensions)
            ])
        except Exception as e:
             print(f"Klasör okunurken hata oluştu: {e}")
             return

        if not image_files:
            print(f"Seçilen klasörde desteklenen formatta ({', '.join(image_extensions)}) resim bulunamadı.")
            return
            
        input_source = image_files # Store the list of image paths
        is_video = False
        print(f"Bulunan resim sayısı: {len(image_files)}")
        # Optional: Get dimensions from the first image
        try:
            first_img = cv2.imread(image_files[0])
            if first_img is not None:
                frame_height, frame_width = first_img.shape[:2]
                print(f"İlk resim çözünürlüğü: {frame_width}x{frame_height}")
            else:
                 print("Uyarı: İlk resim okunamadı, boyutlar alınamadı.")
        except Exception as e:
            print(f"İlk resim okunurken hata: {e}")
            
        # fps = MAX_FPS # Use default FPS for images - Handled by FRAME_TIME logic
        # total_frames = len(image_files)

    else:
        # This case might occur if the path selected is neither file nor directory
        # or if the selection was cancelled and path remained None or empty string
        print(f"Geçersiz veya seçilmemiş giriş yolu: '{input_path}'")
        return
    
    processor = VideoProcessor()
    processor.running = True
    
    # rotate_flag is now managed within the processor instance
    
    try:
        # Video/Resim işleme ve kare işleme görevlerini başlat
        # Pass the determined input_source (VideoCapture or list of image paths) and type flag
        video_task = asyncio.create_task(processor.process_video(input_source, is_video)) 
        frame_task = asyncio.create_task(processor.process_frames())
        
        prev_objects = {}
        print("Video işleme başlıyor...")
        
        while True:
            try:
                if not processor.result_queue.empty():
                    # Sonuçları trafik işareti olmadan al
                    (frame_with_lanes, frame_with_objects, current_objects), fps = processor.result_queue.get() 
                    
                    if current_objects != prev_objects:
                        prev_objects = current_objects.copy()
                    
                    draw_info_panel(frame_with_objects, fps, current_objects)
                    
                    try:
                        # Sadece şerit ve nesne tanıma sonuçlarını birleştir
                        # Rotation is now handled before processing in process_video
                        combined_frame = cv2.hconcat([frame_with_lanes, frame_with_objects]) 
                        cv2.imshow('Otonom Araç Görüntü İşleme', combined_frame)
                    except cv2.error as e:
                        print(f"Görüntü birleştirme hatası: {str(e)}")
                        # Birleştirme başarısız olursa, örneğin nesne tanıma sonucunu göster
                        cv2.imshow('Otonom Araç Görüntü İşleme', frame_with_objects) 

                    key = cv2.waitKey(1) & 0xFF
                    if key == ord('q'):
                        break
                    elif key == ord('r'):
                        # Update the processor's rotate flag
                        processor.rotate_flag = (processor.rotate_flag + 1) % 4
                        print(f"Görüntü döndürme: {processor.rotate_flag * 90} derece")

                await asyncio.sleep(0.001)
                
            except Exception as e:
                print(f"Ana döngü hatası: {str(e)}")
                continue
                
    except Exception as e:
        print(f"Program hatası: {str(e)}")
    finally:
        processor.running = False
        if is_video and cap is not None: # Release video capture if it was used
            cap.release()
        cv2.destroyAllWindows()
        print("Program sonlandırıldı.")

if __name__ == "__main__":
    asyncio.run(main()) 
#test
