import cv2
import numpy as np
import matplotlib.pyplot as plt
import os

gray = None          # ภาพขาวดำ
blur = None          # ภาพหลังเบลอด้วย Gaussian
G = None             # Gradient magnitude จาก Sobel
theta = None         # ทิศทางของ gradient
nms = None           # ผลจาก Non-Maximum Suppression
dt = None            # ผลจาก Double Threshold
weak = None          # ค่า intensity ของขอบแบบ weak
strong = None        # ค่า intensity ของขอบแบบ strong
final = None         # ภาพขอบสุดท้ายหลัง hysteresis


def grayscale_image(image_path):
    """
    ฟังก์ชันนี้ใช้สำหรับ:
    - โหลดภาพจากพาธที่ระบุ
    - แปลงเป็นภาพขาวดำ (Grayscale)
    - บันทึกผลลัพธ์ใน ../result/grayscale_pic.jpg
    """
    global gray

    # โหลดภาพ
    img = cv2.imread(image_path)

    # แปลงเป็นขาวดำ
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # สร้างโฟลเดอร์ result ถ้ายังไม่มี
    os.makedirs('../result', exist_ok=True)

    # บันทึกภาพ
    cv2.imwrite('../result/grayscale_pic.jpg', gray)

    # แสดงภาพ
    plt.imshow(gray, cmap='gray')
    plt.title('Grayscale Image')
    plt.axis('off')
    plt.show()


# --------------------------------------------------------------
def gaussian_kernel(size=21, sigma=5):
    """
    ฟังก์ชันนี้ใช้สำหรับ:
    - สร้างฟิลเตอร์ Gaussian สำหรับการเบลอภาพ
    - size คือขนาด kernel (ต้องเป็นเลขคี่)
    - sigma คือค่าการกระจาย (ยิ่งมากยิ่งเบลอ)
    """
    ax = np.linspace(-(size - 1) / 2., (size - 1) / 2., size)
    xx, yy = np.meshgrid(ax, ax)
    kernel = np.exp(-(xx**2 + yy**2) / (2. * sigma**2))
    return kernel / np.sum(kernel)


# --------------------------------------------------------------
def apply_gaussian_blur():
    """
    ฟังก์ชันนี้ใช้สำหรับ:
    - เบลอภาพด้วย Gaussian Filter เพื่อลดสัญญาณรบกวน (Noise)
    - บันทึกผลลัพธ์ใน ../result/blur_pic.jpg
    """
    global blur

    # โหลดภาพ grayscale ที่ได้จากขั้นตอนก่อนหน้า
    img = cv2.imread('../result/grayscale_pic.jpg')
    if img is None:
        print("❌ not found grayscale image")
        return

    # แปลงเป็น grayscale อีกครั้งเพื่อความแน่ใจ
    gray_local = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # สร้าง Gaussian Kernel
    kernel = gaussian_kernel()

    # ทำการเบลอภาพ
    blur = cv2.filter2D(gray_local, -1, kernel) # ทำการ convolution

    # บันทึกภาพ
    cv2.imwrite('../result/blur_pic.jpg', blur)

    # แสดงผล
    plt.imshow(blur, cmap='gray')
    plt.title('Gaussian Blur')
    plt.axis('off')
    plt.show()


# --------------------------------------------------------------
def sobel_edge_detection():
    """
    ฟังก์ชันนี้ใช้สำหรับ:
    - ตรวจจับขอบภาพด้วย Sobel Operator
    - คำนวณทั้ง Gradient Magnitude (G) และ Gradient Direction (theta)
    - บันทึกผลลัพธ์ใน ../result/sobel_edge_pic.jpg
    """
    global G, theta

    # โหลดภาพที่เบลอแล้ว
    img = cv2.imread('../result/blur_pic.jpg', cv2.IMREAD_GRAYSCALE)
    if img is None:
        print("❌ not found blur image")
        return

    # Sobel Kernels
    Kx = np.array([[-1,0,1],
                   [-2,0,2],
                   [-1,0,1]])
    Ky = np.array([[1,2,1],
                   [0,0,0],
                   [-1,-2,-1]])

    # คำนวณ Gradient
    Gx = cv2.filter2D(img, cv2.CV_64F, Kx) # ทำการ convolution
    plt.imshow(Gx, cmap='gray')
    plt.title('Sobel Gradient X-axis')
    plt.axis('off')
    plt.show()
    Gy = cv2.filter2D(img, cv2.CV_64F, Ky) # ทำการ convolution
    plt.imshow(Gy, cmap='gray')
    plt.title('Gradient Y-axis')
    plt.axis('off')
    plt.show()

    # ขนาดของ Gradient (Magnitude)
    G = np.hypot(Gx, Gy)
    G = G / G.max() * 255
    G = G.astype(np.uint8)

    # ทิศทางของ Gradient (Direction)
    theta = np.arctan2(Gy, Gx)

    # บันทึกผล
    cv2.imwrite('../result/sobel_edge_pic.jpg', G)

    # แสดงภาพ
    plt.imshow(G, cmap='gray')
    plt.title('Sobel Gradient Magnitude')
    plt.axis('off')
    plt.show()


# --------------------------------------------------------------
def non_maximum_suppression():
    """
    ฟังก์ชันนี้ใช้สำหรับ:
    - ทำให้เส้นขอบบางลง โดยเก็บเฉพาะพิกเซลที่มีค่า gradient มากที่สุดในทิศทางเดียวกัน
    - บันทึกผลใน ../result/nms_pic.jpg
    """
    global nms

    M, N = G.shape
    Z = np.zeros((M, N), dtype=np.uint8)

    # แปลงมุมจาก radian → degree
    angle = theta * 180.0 / np.pi
    angle[angle < 0] += 180

    for i in range(1, M-1):
        for j in range(1, N-1):
            q = 255
            r = 255

            # ตรวจทิศทางของ gradient แล้วเทียบค่าข้างเคียง
            if (0 <= angle[i,j] < 22.5) or (157.5 <= angle[i,j] <= 180):
                q = G[i, j+1]
                r = G[i, j-1]
            elif (22.5 <= angle[i,j] < 67.5):
                q = G[i+1, j-1]
                r = G[i-1, j+1]
            elif (67.5 <= angle[i,j] < 112.5):
                q = G[i+1, j]
                r = G[i-1, j]
            elif (112.5 <= angle[i,j] < 157.5):
                q = G[i-1, j-1]
                r = G[i+1, j+1]

            # เก็บเฉพาะค่าสูงสุด
            if (G[i,j] >= q) and (G[i,j] >= r):
                Z[i,j] = G[i,j]
            else:
                Z[i,j] = 0

    nms = Z
    cv2.imwrite('../result/nms_pic.jpg', nms)

    plt.imshow(nms, cmap='gray')
    plt.title('After Non-Maximum Suppression')
    plt.axis('off')
    plt.show()


# --------------------------------------------------------------
def double_threshold(low_ratio=0.05, high_ratio=0.15):
    """
    ฟังก์ชันนี้ใช้สำหรับ:
    - แยกพิกเซลขอบเป็น 3 ระดับ:
        1. strong (ขอบชัด)
        2. weak (ขอบบาง)
        3. non-edge (ไม่ใช่ขอบ)
    - บันทึกผลใน ../result/double_threshold_pic.jpg
    """
    global dt, weak, strong

    high_threshold = nms.max() * high_ratio
    low_threshold = high_threshold * low_ratio

    M, N = nms.shape
    dt = np.zeros((M, N), dtype=np.uint8)

    weak = np.uint8(75)
    strong = np.uint8(255)

    strong_i, strong_j = np.where(nms >= high_threshold)
    weak_i, weak_j = np.where((nms <= high_threshold) & (nms >= low_threshold))

    dt[strong_i, strong_j] = strong
    dt[weak_i, weak_j] = weak

    cv2.imwrite('../result/double_threshold_pic.jpg', dt)

    plt.imshow(dt, cmap='gray')
    plt.title('Double Threshold')
    plt.axis('off')
    plt.show()


# --------------------------------------------------------------
def hysteresis():
    """
    ฟังก์ชันนี้ใช้สำหรับ:
    - ทำ Edge Tracking by Hysteresis
    - ถ้าขอบ weak อยู่ติดกับ strong → ถือว่าเป็นขอบจริง
    - ถ้าไม่ติด → ลบทิ้ง
    - บันทึกผลสุดท้ายใน ../result/final_canny_pic.jpg
    """
    global final

    M, N = dt.shape
    img = dt.copy()

    for i in range(1, M-1):
        for j in range(1, N-1):
            if img[i, j] == weak:
                if np.any(img[i-1:i+2, j-1:j+2] == strong):
                    img[i, j] = strong
                else:
                    img[i, j] = 0

    final = img
    cv2.imwrite('../result/final_canny_pic.jpg', final)

    plt.imshow(final, cmap='gray')
    plt.title('Final Canny Edge Detection')
    plt.axis('off')
    plt.show()


# --------------------------------------------------------------
if __name__ == "__main__":
    # 🔹 ขั้นตอนการทำงานทั้งหมด
    grayscale_image('../images/cat.jpg')
    apply_gaussian_blur()
    sobel_edge_detection()
    non_maximum_suppression()
    double_threshold()
    hysteresis()

    print("\n=== ตัวอย่างค่าภายใน G (Gradient Magnitude) ===")
    print(G[:5, :5])  # แสดง 5×5 พื้นที่บนซ้าย

    print("\n=== ตัวอย่างค่าภายใน θ (Gradient Direction, หน่วย radian) ===")
    print(theta[:5, :5])  # แสดง 5×5 พื้นที่บนซ้าย

    print("Canny Edge Detection Completed! Results saved in '../result/'")
