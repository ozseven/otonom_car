import cv2

class VideoProcessor:
    def process_video(self, video_path):
        """Video işleme ana fonksiyonu"""
        try:
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                raise Exception("Video açılamadı")

            # Video özelliklerini al
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fps = int(cap.get(cv2.CAP_PROP_FPS))

            # Video kaydedici
            output_path = video_path.rsplit('.', 1)[0] + '_processed.mp4'
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

            frame_count = 0
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break

                # Videoyu saat yönünde 90 derece döndür
                #frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
                # Frame'i işle
                processed_frame = self.process_frame(frame)
                
                # İşlenmiş frame'i kaydet
                out.write(processed_frame)
                
                frame_count += 1
                progress = (frame_count / total_frames) * 100
                print(f"\rİşleniyor: %{progress:.1f}", end="")

            cap.release()
            out.release()
            print("\nVideo işleme tamamlandı!")
            return output_path

        except Exception as e:
            print(f"Video işleme hatası: {str(e)}")
            return None 