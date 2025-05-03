import cv2
import numpy as np

class LaneDetector:
    def __init__(self):
        """
        Şerit tespiti için gerekli parametreleri başlatır
        """
        # Canny kenar tespiti için parametreler
        self.low_threshold = 50
        self.high_threshold = 150
        
        # Hough dönüşümü için parametreler
        self.rho = 1
        self.theta = np.pi/180
        self.threshold = 50
        self.min_line_length = 100
        self.max_line_gap = 50
        
        # Şerit filtreleme parametreleri
        self.min_slope = 0.3  # Minimum eğim
        self.max_slope = 2.0  # Maximum eğim
        self.roi_vertices = None  # İlgi alanı köşe noktaları
        
        # Şerit tipleri için parametreler
        self.dashed_line_gap = 50  # Kesikli çizgi boşluğu
        self.solid_line_length = 100  # Düz çizgi minimum uzunluğu

    def set_roi(self, image_shape):
        """
        İlgi alanını (ROI) belirler
        """
        height, width = image_shape[:2]
        # Alt yarısı ve orta kısmı seç
        self.roi_vertices = np.array([
            [(0, height), (width/2 - width/4, height/2),
             (width/2 + width/4, height/2), (width, height)]
        ], dtype=np.int32)

    def classify_lane_type(self, line):
        """
        Şerit tipini belirler (kesikli/düz)
        """
        x1, y1, x2, y2 = line[0]
        length = np.sqrt((x2 - x1)**2 + (y2 - y1)**2)
        
        if length < self.solid_line_length:
            return "Kesikli Şerit"
        else:
            return "Düz Şerit"

    def filter_lines(self, lines):
        """
        Tespit edilen çizgileri filtreler ve şerit tiplerini belirler
        """
        if lines is None:
            return []
            
        filtered_lines = []
        for line in lines:
            x1, y1, x2, y2 = line[0]
            
            # Yatay çizgileri filtrele
            if abs(x2 - x1) < 20:  # Dikey çizgileri kabul et
                continue
                
            # Eğimi hesapla
            if x2 - x1 == 0:
                continue
            slope = abs((y2 - y1) / (x2 - x1))
            
            # Eğim aralığında olan çizgileri kabul et
            if self.min_slope <= slope <= self.max_slope:
                # Şerit tipini belirle
                lane_type = self.classify_lane_type(line)
                filtered_lines.append((line, lane_type))
                
        return filtered_lines

    def preprocess_image(self, image):
        """
        Görüntüyü ön işleme adımlarından geçirir
        
        Args:
            image: İşlenecek görüntü
            
        Returns:
            İşlenmiş görüntü
        """
        # Görüntüyü küçült
        image = cv2.resize(image, (640, 480))
        
        # Görüntüyü gri tonlamaya çevir
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Gürültüyü azalt
        blur = cv2.GaussianBlur(gray, (5, 5), 0)
        
        # İlgi alanını belirle
        if self.roi_vertices is None:
            self.set_roi(blur.shape)
            
        # Maske oluştur
        mask = np.zeros_like(blur)
        cv2.fillPoly(mask, self.roi_vertices, 255)
        
        # İlgi alanını uygula
        masked_image = cv2.bitwise_and(blur, mask)
        
        return masked_image

    def detect_edges(self, image):
        """
        Canny kenar tespiti uygular
        
        Args:
            image: İşlenecek görüntü
            
        Returns:
            Kenar tespiti yapılmış görüntü
        """
        edges = cv2.Canny(image, self.low_threshold, self.high_threshold)
        return edges

    def detect_lanes(self, image):
        """
        Şeritleri tespit eder
        
        Args:
            image: İşlenecek görüntü
            
        Returns:
            Şeritlerin çizildiği görüntü
        """
        # Görüntüyü ön işle
        processed = self.preprocess_image(image)
        
        # Kenar tespiti yap
        edges = self.detect_edges(processed)
        
        # Hough dönüşümü ile çizgileri tespit et
        lines = cv2.HoughLinesP(
            edges,
            self.rho,
            self.theta,
            self.threshold,
            minLineLength=self.min_line_length,
            maxLineGap=self.max_line_gap
        )
        
        # Çizgileri filtrele ve şerit tiplerini belirle
        filtered_lines = self.filter_lines(lines)
        
        # Tespit edilen şeritleri görüntüye çiz
        if filtered_lines:
            for line, lane_type in filtered_lines:
                x1, y1, x2, y2 = line[0]
                
                # Şerit tipine göre renk belirle
                color = (0, 255, 0) if lane_type == "Düz Şerit" else (0, 0, 255)
                
                # Şeridi çiz
                cv2.line(image, (x1, y1), (x2, y2), color, 2)
                
                # Şerit tipini yaz
                cv2.putText(image, lane_type, (x1, y1 - 10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        
        return image 