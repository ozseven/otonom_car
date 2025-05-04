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
        self.threshold = 25
        self.min_line_length = 25
        self.max_line_gap = 80
        
        # Şerit filtreleme parametreleri
        self.min_slope = 0.05
        self.max_slope = 3.0
        self.roi_vertices = None
        
        # Perspektif düzeltme parametreleri
        self.perspective_matrix = None
        self.inverse_perspective_matrix = None
        self.lane_width_ratio = 0.1  # Görüntü genişliğine göre şerit genişliği oranı
        self.expected_lane_width = 3.5  # Metre cinsinden beklenen şerit genişliği
        
        # Şerit takibi için parametreler
        self.prev_lines = []
        self.max_prev_lines = 10
        self.line_persistence = 5
        self.line_counter = {}
        
        # Polinom regresyon parametreleri
        self.poly_degree = 2
        self.left_fit = None
        self.right_fit = None
        self.left_points = []
        self.right_points = []
        self.smooth_factor = 0.85
        
        # Perspektif düzeltme noktaları
        self.src_points = None
        self.dst_points = None
        
        # Görüntü boyutları
        self.image_width = None
        self.image_height = None
        self.scale_factor = 1.0

    def calculate_scale_factor(self, width, height):
        """
        Görüntü boyutlarına göre ölçeklendirme faktörünü hesaplar
        """
        # Referans boyutlar (1280x720)
        ref_width = 1920
        ref_height = 1080
        
        # Genişlik ve yükseklik için ayrı ölçeklendirme faktörleri
        width_scale = width / ref_width
        height_scale = height / ref_height
        
        # En küçük ölçeklendirme faktörünü kullan
        return min(width_scale, height_scale)

    def scale_parameters(self):
        """
        Görüntü boyutlarına göre parametreleri ölçeklendirir
        """
        if self.image_width is None or self.image_height is None:
            return
            
        # Ölçeklendirme faktörünü hesapla
        self.scale_factor = self.calculate_scale_factor(self.image_width, self.image_height)
        
        # Hough dönüşümü parametrelerini ölçeklendir
        self.min_line_length = int(25 * self.scale_factor)
        self.max_line_gap = int(80 * self.scale_factor)
        
        # Şerit genişliğini ölçeklendir
        self.lane_width_pixels = int(self.image_width * self.lane_width_ratio)

    def initialize_perspective_transform(self, image_shape):
        """
        Perspektif dönüşüm matrisini başlatır
        """
        height, width = image_shape[:2]
        self.image_width = width
        self.image_height = height
        
        # Parametreleri ölçeklendir
        self.scale_parameters()
        
        # Kaynak noktalar (orijinal görüntüdeki trapez)
        # Çok daha dar ve merkezi bir görüş açısı için noktaları ayarla
        self.src_points = np.float32([
            [width * 0.35, height],  # Sol alt - daha sağa
            [width * 0.48, height * 0.5],  # Sol üst - daha yukarı ve sağa
            [width * 0.52, height * 0.5],  # Sağ üst - daha yukarı ve sola
            [width * 0.65, height]  # Sağ alt - daha sola
        ])
        
        # Hedef noktalar (düzleştirilmiş görüntüdeki dikdörtgen)
        # Çok daha dar ve merkezi bir çıktı için noktaları ayarla
        self.dst_points = np.float32([
            [width * 0.35, height],  # Sol alt
            [width * 0.35, 0],  # Sol üst
            [width * 0.65, 0],  # Sağ üst
            [width * 0.65, height]  # Sağ alt
        ])
        
        # Perspektif dönüşüm matrislerini hesapla
        self.perspective_matrix = cv2.getPerspectiveTransform(self.src_points, self.dst_points)
        self.inverse_perspective_matrix = cv2.getPerspectiveTransform(self.dst_points, self.src_points)

    def apply_perspective_transform(self, image):
        """
        Perspektif düzeltme uygular
        """
        if self.perspective_matrix is None:
            self.initialize_perspective_transform(image.shape)
        
        return cv2.warpPerspective(image, self.perspective_matrix, (image.shape[1], image.shape[0]))

    def inverse_perspective_transform(self, image):
        """
        Ters perspektif düzeltme uygular
        """
        if self.inverse_perspective_matrix is None:
            self.initialize_perspective_transform(image.shape)
        
        return cv2.warpPerspective(image, self.inverse_perspective_matrix, (image.shape[1], image.shape[0]))

    def calculate_lane_width(self, left_fit, right_fit, y):
        """
        Belirli bir y koordinatında şerit genişliğini hesaplar
        """
        if left_fit is None or right_fit is None:
            return None
            
        left_x = np.polyval(left_fit, y)
        right_x = np.polyval(right_fit, y)
        return right_x - left_x

    def adjust_lane_width(self, left_fit, right_fit, y_points):
        """
        Şerit genişliğini ayarlar
        """
        if left_fit is None or right_fit is None:
            return left_fit, right_fit
            
        # Ortalama şerit genişliğini hesapla
        widths = []
        for y in y_points:
            width = self.calculate_lane_width(left_fit, right_fit, y)
            if width is not None:
                widths.append(width)
        
        if not widths:
            return left_fit, right_fit
            
        avg_width = np.mean(widths)
        
        # Şerit genişliğini ayarla
        tolerance = int(self.lane_width_pixels * 0.1)  # %10 tolerans
        if abs(avg_width - self.lane_width_pixels) > tolerance:
            # Sağ şeridi sola veya sağa kaydır
            adjustment = (self.lane_width_pixels - avg_width) / 2
            right_fit[0] = right_fit[0] - adjustment
            
        return left_fit, right_fit

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
            if slope < 0 and mid_x < 640:  # Sol şerit
                left_lines.append(line)
            elif slope > 0 and mid_x > 640:  # Sağ şerit
                right_lines.append(line)
                
        return left_lines, right_lines

    def fit_polynomial(self, lines, side='left'):
        """
        Şerit noktalarına polinom uydurur
        """
        if not lines:
            return None
            
        points = []
        for line in lines:
            x1, y1, x2, y2 = line[0]
            points.extend([(x1, y1), (x2, y2)])
            
        points = np.array(points)
        if len(points) < 2:
            return None
            
        x = points[:, 0]
        y = points[:, 1]
        
        try:
            fit = np.polyfit(y, x, self.poly_degree)
            
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
            
        # Nokta sayısını görüntü yüksekliğine göre ayarla
        num_points = int(self.image_height * 0.1)  # Her 10 pikselde bir nokta
        y = np.linspace(y_start, y_end, num_points)
        x = np.polyval(fit, y)
        
        points = np.column_stack((x, y))
        return points.astype(np.int32)

    def filter_lines(self, lines):
        """
        Tespit edilen çizgileri filtreler ve şerit genişliğini ayarlar
        """
        if lines is None:
            return self.prev_lines
            
        filtered_lines = []
        for line in lines:
            x1, y1, x2, y2 = line[0]
            slope = self.calculate_slope(x1, y1, x2, y2)
            
            if abs(slope) >= self.min_slope and abs(slope) <= self.max_slope:
                filtered_lines.append(line)
        
        left_lines, right_lines = self.separate_lanes(filtered_lines)
        
        # Polinom uydur
        self.left_fit = self.fit_polynomial(left_lines, 'left')
        self.right_fit = self.fit_polynomial(right_lines, 'right')
        
        # Şerit genişliğini ayarla
        y_points = np.linspace(self.image_height, 0, int(self.image_height * 0.1))
        self.left_fit, self.right_fit = self.adjust_lane_width(self.left_fit, self.right_fit, y_points)
        
        # Şerit noktalarını oluştur
        self.left_points = self.generate_lane_points(self.left_fit, self.image_height, 0)
        self.right_points = self.generate_lane_points(self.right_fit, self.image_height, 0)
        
        return filtered_lines

    def preprocess_image(self, image):
        """
        Görüntüyü ön işleme adımlarından geçirir
        """
        # Görüntü boyutlarını güncelle
        height, width = image.shape[:2]
        if width != self.image_width or height != self.image_height:
            self.image_width = width
            self.image_height = height
            self.scale_parameters()
        
        # Görüntüyü gri tonlamaya çevir
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Gürültüyü azalt
        blur = cv2.GaussianBlur(gray, (5, 5), 0)
        
        # Kontrastı artır
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        enhanced = clahe.apply(blur)
        
        # Perspektif düzeltme uygula
        warped = self.apply_perspective_transform(enhanced)
        
        return warped

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
        # Perspektif düzeltilmiş görüntüde şeritleri çiz
        lane_image = np.zeros_like(image)
        
        if len(self.left_points) > 1 and len(self.right_points) > 1:
            # Sol şeridi çiz
            cv2.polylines(lane_image, [self.left_points], False, (255, 255, 255), 
                         int(3 * self.scale_factor))
            # Sağ şeridi çiz
            cv2.polylines(lane_image, [self.right_points], False, (255, 255, 255), 
                         int(3 * self.scale_factor))
            
            # Şeritler arasını doldur
            points = np.vstack((self.left_points, self.right_points[::-1]))
            cv2.fillPoly(lane_image, [points], (0, 255, 0))
        
        # Ters perspektif dönüşümü uygula
        lane_image = self.inverse_perspective_transform(lane_image)
        
        # Orijinal görüntü ile birleştir
        result = cv2.addWeighted(image, 1, lane_image, 0.3, 0)
        
        return result

    def detect_lanes(self, image):
        """
        Şeritleri tespit eder
        """
        processed = self.preprocess_image(image)
        edges = self.detect_edges(processed)
        
        lines = cv2.HoughLinesP(
            edges,
            self.rho,
            self.theta,
            self.threshold,
            minLineLength=self.min_line_length,
            maxLineGap=self.max_line_gap
        )
        
        self.filter_lines(lines)
        result = self.draw_lane_lines(image)
        
        return result 