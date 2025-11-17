import datetime
from datetime import datetime
import argparse
from pathlib import Path
import cv2
import os
import numpy as np
from skimage.metrics import peak_signal_noise_ratio, structural_similarity

def write_file(out_path, img):
    if img.dtype == np.float32 or img.dtype == np.float64:
        img_disp = np.clip(img, 0, 1) if img.max() <= 1.0 else (img / img.max())
        img_disp = (img_disp * 255).astype(np.uint8)
    else:
        img_disp = img
    cv2.imwrite(str(out_path), img_disp)

def to_grayscale(img_bgr):
    if img_bgr.ndim == 2:
        img_gray = img_bgr
        print("Gambar sudah grayscale — tidak dikonversi.")
    else:
        # cek apakah ketiga kanal identik (hasil dari membaca grayscale dengan IMREAD_COLOR)
        if img_bgr.shape[2] == 3:
            b, g, r = cv2.split(img_bgr)
            if np.array_equal(b, g) and np.array_equal(b, r):
                img_gray = b
                print("Gambar secara efektif grayscale (3 kanal identik) — tidak dikonversi.")
            else:
                img_gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
                print("Gambar berwarna — dikonversi ke grayscale.")
        else:
            # fallback untuk jumlah kanal tak terduga (mis. 4)
            try:
                img_gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
                print("Gambar dengan kanal tak terduga — dikonversi ke grayscale.")
            except Exception:
                img_gray = img_bgr[..., 0]
                print("Fallback: menggunakan kanal pertama sebagai grayscale.")

    return img_gray


def find_optimal_kernel(img):
    best_kernel = None
    best_score = -1

    for k in [3, 5, 7, 9]:
        denoised = cv2.blur(img, (k, k))
        psnr = peak_signal_noise_ratio(img, denoised)
        ssim = structural_similarity(img, denoised)
        score = (psnr + (ssim * 100)) / 2  # gabungan
        print(f"Kernel {k}x{k}: PSNR={psnr:.2f}, SSIM={ssim:.3f}")
        if score > best_score:
            best_score = score
            best_kernel = k

    return best_kernel

def normalize_and_uint8(image):
    """
    Menormalkan citra hasil perhitungan (float) ke rentang 0–255
    lalu mengubahnya menjadi tipe uint8.
    """
    abs_img = np.abs(image)
    norm_img = (abs_img / abs_img.max()) * 255
    return norm_img.astype(np.uint8)


# ---------------------------------------------------------
# 3. Operator ROBERTS
# ---------------------------------------------------------
def roberts_edge(gray_img):
    # Kernel Roberts 2x2
    kernel_gx = np.array([[1, 0],
                          [0, -1]], dtype=np.float32)
    kernel_gy = np.array([[0, 1],
                          [-1, 0]], dtype=np.float32)

    gx = cv2.filter2D(gray_img, cv2.CV_64F, kernel_gx)
    gy = cv2.filter2D(gray_img, cv2.CV_64F, kernel_gy)

    mag = np.sqrt(gx**2 + gy**2)
    return normalize_and_uint8(mag)


# ---------------------------------------------------------
# 4. Operator PREWITT
# ---------------------------------------------------------
def prewitt_edge(gray_img):
    # Kernel Prewitt 3x3
    kernel_gx = np.array([[-1, 0, 1],
                          [-1, 0, 1],
                          [-1, 0, 1]], dtype=np.float32)

    kernel_gy = np.array([[1, 1, 1],
                          [0, 0, 0],
                          [-1, -1, -1]], dtype=np.float32)

    gx = cv2.filter2D(gray_img, cv2.CV_64F, kernel_gx)
    gy = cv2.filter2D(gray_img, cv2.CV_64F, kernel_gy)

    mag = np.sqrt(gx**2 + gy**2)
    return normalize_and_uint8(mag)


# ---------------------------------------------------------
# 5. Operator SOBEL
# ---------------------------------------------------------
def sobel_edge(gray_img):
    gx = cv2.Sobel(gray_img, cv2.CV_64F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray_img, cv2.CV_64F, 0, 1, ksize=3)
    mag = np.sqrt(gx**2 + gy**2)
    return normalize_and_uint8(mag)


# ---------------------------------------------------------
# 6. Operator LAPLACIAN
# ---------------------------------------------------------
def laplacian_edge(gray_img):
    lap = cv2.Laplacian(gray_img, cv2.CV_64F, ksize=3)
    return normalize_and_uint8(lap)

def main():
    output = "hasil"
    print("Running Program, on :", datetime.now().strftime("%Y-%m-%d"))
    parser = argparse.ArgumentParser(description="Program untuk manipulasi citra digital Forum Diskusi 15 Mata Kuliah Pengolahan Citra Digital")
    parser.add_argument("--image", required=True, help="Lokasi Gambar")
    args = parser.parse_args()
    print("Lokasi Gambar:", args.image)
    Path(output).mkdir(parents=True, exist_ok=True)

    # Baca gambar
    img_bgr = cv2.imread(args.image, cv2.IMREAD_COLOR)
    if img_bgr is None:
        raise FileNotFoundError(f"Gambar tidak ditemukan: {args.image}")
    img_filename = Path(args.image).name
    print("Nama file gambar:", img_filename)
    write_file(os.path.join(output, f"01_original_image_{img_filename}"), img_bgr)
    
    # Convert ke grayscale kalau gambarnya berwarna
    processed_img = to_grayscale(img_bgr)
    write_file(os.path.join(output, f"02_grayscale_image_{img_filename}"), processed_img)

    #Noise reduction
    optimal_kernel = find_optimal_kernel(processed_img)
    kernel_size = (optimal_kernel, optimal_kernel)
    print(f"Size kernel yang optimal: {kernel_size}")
    img_denoised = cv2.blur(processed_img, kernel_size) #menggunakan metode mean filter
    write_file(os.path.join(output, f"03_denoised_image_{img_filename}"), img_denoised)

    # Deteksi tepi dengan berbagai operator
    edges_roberts = roberts_edge(img_denoised)
    write_file(os.path.join(output, f"04_edges_roberts_{img_filename}"), edges_roberts) 
    edges_prewitt = prewitt_edge(img_denoised)
    write_file(os.path.join(output, f"05_edges_prewitt_{img_filename}"), edges_prewitt)
    edges_sobel = sobel_edge(img_denoised)
    write_file(os.path.join(output, f"06_edges_sobel_{img_filename}"), edges_sobel)
    edges_laplacian = laplacian_edge(img_denoised)
    write_file(os.path.join(output, f"07_edges_laplacian_{img_filename}"), edges_laplacian)

    # Add labels to each image
    labels = ['Roberts', 'Prewitt', 'Sobel', 'Laplacian']
    images = [edges_roberts, edges_prewitt, edges_sobel, edges_laplacian]
    labeled_images = []

    for img, label in zip(images, labels):
        labeled_img = cv2.putText(img.copy(), label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
        labeled_images.append(labeled_img)

    # Combine labeled images into a grid 2x2 and save
    top_row = np.hstack((labeled_images[0], labeled_images[1]))
    bottom_row = np.hstack((labeled_images[2], labeled_images[3]))
    combined_edges = np.vstack((top_row, bottom_row))
    write_file(os.path.join(output, f"08_edges_combined_{img_filename}"), combined_edges)

if __name__ == "__main__":
    main()