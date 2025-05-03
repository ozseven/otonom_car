import cv2
import numpy as np
from scipy.signal import find_peaks

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
        self.threshold = 30
        self.min_line_length = 40
        self.max_line_gap = 50
        
        # Şerit filtreleme parametreleri
        self.min_slope = 0.2
        self.max_slope = 2.0
        self.roi_vertices = None
        
        # Şerit takibi için parametreler
        self.prev_lines = []
        self.max_prev_lines = 10
        self.line_persistence = 5
        self.line_counter = {}
        
        # Polinom regresyon parametreleri
        self.poly_degree = 2  # Polinom derecesi
        self.left_fit = None  # Sol şerit polinom katsayıları
        self.right_fit = None  # Sağ şerit polinom katsayıları
        self.left_points = []  # Sol şerit noktaları
        self.right_points = []  # Sağ şerit noktaları
        self.smooth_factor = 0.8  # Polinom yumuşatma faktörü

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

    def calculate_slope(self, x1, y1, x2, y2):
        """
        İki nokta arasındaki eğimi hesaplar
        """
        if x2 - x1 == 0:
            return float('inf')
        return (y2 - y1) / (x2 - x1)

    def separate_lanes(self, lines):
        """
        Tespit edilen çizgileri sol ve sağ şerit olarak ayırır
        """
        left_lines = []
        right_lines = []
        
        if lines is None:
            return left_lines, right_lines
            
        for line in lines:
            x1, y1, x2, y2 = line[0]
            slope = self.calculate_slope(x1, y1, x2, y2)
            
            # Çizginin orta noktasını hesapla
            mid_x = (x1 + x2) / 2
            
            # Çizgiyi sol veya sağ şeride ata
            if slope < 0 and mid_x < 320:  # Sol şerit
                left_lines.append(line)
            elif slope > 0 and mid_x > 320:  # Sağ şerit
                right_lines.append(line)
                
        return left_lines, right_lines

    def fit_polynomial(self, lines, side='left'):
        """
        Şerit noktalarına polinom uydurur
        """
        if not lines:
            return None
            
        # Tüm noktaları topla
        points = []
        for line in lines:
            x1, y1, x2, y2 = line[0]
            points.extend([(x1, y1), (x2, y2)])
            
        points = np.array(points)
        if len(points) < 2:
            return None
            
        # x ve y koordinatlarını ayır
        x = points[:, 0]
        y = points[:, 1]
        
        # Polinom uydur
        try:
            fit = np.polyfit(y, x, self.poly_degree)
            
            # Önceki polinom ile yumuşat
            if side == 'left' and self.left_fit is not None:
                fit = self.smooth_factor * self.left_fit + (1 - self.smooth_factor) * fit
            elif side == 'right' and self.right_fit is not None:
                fit = self.smooth_factor * self.right_fit + (1 - self.smooth_factor) * fit
                
            return fit
        except:
            return None

    def generate_lane_points(self, fit, y_start, y_end):
        """
        Polinom katsayılarından şerit noktalarını oluşturur
        """
        if fit is None:
            return []
            
        # y değerlerini oluştur
        y = np.linspace(y_start, y_end, 50)
        
        # x değerlerini hesapla
        x = np.polyval(fit, y)
        
        # Noktaları birleştir
        points = np.column_stack((x, y))
        return points.astype(np.int32)

    def filter_lines(self, lines):
        """
        Tespit edilen çizgileri filtreler
        """
        if lines is None:
            return self.prev_lines
            
        filtered_lines = []
        for line in lines:
            x1, y1, x2, y2 = line[0]
            slope = self.calculate_slope(x1, y1, x2, y2)
            
            if abs(slope) >= self.min_slope and abs(slope) <= self.max_slope:
                filtered_lines.append(line)
        
        # Sol ve sağ şeritleri ayır
        left_lines, right_lines = self.separate_lanes(filtered_lines)
        
        # Polinom uydur
        self.left_fit = self.fit_polynomial(left_lines, 'left')
        self.right_fit = self.fit_polynomial(right_lines, 'right')
        
        # Şerit noktalarını oluştur
        height = 480  # Görüntü yüksekliği
        self.left_points = self.generate_lane_points(self.left_fit, height, height//2)
        self.right_points = self.generate_lane_points(self.right_fit, height, height//2)
        
        return filtered_lines

    def preprocess_image(self, image):
        """
        Görüntüyü ön işleme adımlarından geçirir
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
        """
        edges = cv2.Canny(image, self.low_threshold, self.high_threshold)
        return edges

    def draw_lane_lines(self, image):
        """
        Şeritleri görüntüye çizer
        """
        # Sol şeridi çiz
        if len(self.left_points) > 1:
            cv2.polylines(image, [self.left_points], False, (255, 255, 255), 3)
            
        # Sağ şeridi çiz
        if len(self.right_points) > 1:
            cv2.polylines(image, [self.right_points], False, (255, 255, 255), 3)
            
        return image

    def detect_lanes(self, image):
        """
        Şeritleri tespit eder
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
        
        # Çizgileri filtrele
        self.filter_lines(lines)
        
        # Şeritleri çiz
        image = self.draw_lane_lines(image)
        
        return image 