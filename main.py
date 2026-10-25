
import cv2
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# =========================================================
# 1. PENGATURAN FOLDER
# =========================================================

BASE_DIR = Path(__file__).resolve().parent
INPUT_DIR = BASE_DIR / "citra"
WATERSHED_DIR = BASE_DIR / "hasil_watershed"
EDGE_DIR = BASE_DIR / "hasil_edge"
COMPARISON_DIR = BASE_DIR / "hasil_perbandingan"

for folder in [WATERSHED_DIR, EDGE_DIR, COMPARISON_DIR]:
    folder.mkdir(parents=True, exist_ok=True)

EXTENSIONS = {
    ".jpg", ".jpeg", ".png", ".bmp",
    ".tif", ".tiff", ".webp"
}


# =========================================================
# 2. WATERSHED SEGMENTATION
# =========================================================

def watershed_segmentation(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    # Thresholding otomatis menggunakan Otsu
    _, binary = cv2.threshold(
        blurred,
        0,
        255,
        cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
    )

    kernel = np.ones((3, 3), np.uint8)

    # Menghilangkan noise
    opening = cv2.morphologyEx(
        binary,
        cv2.MORPH_OPEN,
        kernel,
        iterations=2
    )

    # Area latar belakang
    sure_bg = cv2.dilate(
        opening, kernel, iterations=3
    )

    # Area inti objek
    distance = cv2.distanceTransform(
        opening, cv2.DIST_L2, 5
    )

    if distance.max() > 0:
        _, sure_fg = cv2.threshold(
            distance,
            0.4 * distance.max(),
            255,
            cv2.THRESH_BINARY
        )
    else:
        sure_fg = np.zeros_like(gray)

    sure_fg = np.uint8(sure_fg)

    # Area yang belum diketahui
    unknown = cv2.subtract(sure_bg, sure_fg)

    # Membuat marker
    _, markers = cv2.connectedComponents(sure_fg)
    markers = markers + 1
    markers[unknown == 255] = 0

    # Menjalankan algoritma Watershed
    markers = cv2.watershed(image.copy(), markers)

    # Batas segmentasi berwarna merah
    boundary = image.copy()
    boundary[markers == -1] = [0, 0, 255]

    # Mask hasil segmentasi
    mask = np.zeros(gray.shape, dtype=np.uint8)
    mask[markers > 1] = 255

    # Hasil objek dengan latar belakang hitam
    segmented = cv2.bitwise_and(
        image, image, mask=mask
    )

    return boundary, mask, segmented


# =========================================================
# 3. EDGE-BASED SEGMENTATION
# =========================================================

def edge_segmentation(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Mengurangi noise
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    # Deteksi tepi Canny
    edges = cv2.Canny(
        blurred,
        threshold1=50,
        threshold2=150
    )

    # Menebalkan garis tepi
    kernel = np.ones((3, 3), np.uint8)
    edges_thick = cv2.dilate(
        edges, kernel, iterations=1
    )

    # Overlay garis merah
    overlay = image.copy()
    overlay[edges_thick > 0] = [0, 0, 255]

    return edges, overlay


# =========================================================
# 4. FUNGSI MENYIMPAN GAMBAR
# =========================================================

def save_image(path, image):
    success = cv2.imwrite(str(path), image)

    if not success:
        raise RuntimeError(
            f"Gagal menyimpan gambar: {path}"
        )


# =========================================================
# 5. PROGRAM UTAMA
# =========================================================

def main():
    print("=" * 65)
    print("  PROGRAM SEGMENTASI CITRA")
    print("  WATERSHED DAN EDGE-BASED")
    print("=" * 65)

    if not INPUT_DIR.exists():
        print(f"Folder citra tidak ditemukan: {INPUT_DIR}")
        return

    # Membaca semua gambar dalam folder citra
    files = sorted(
        [
            p for p in INPUT_DIR.iterdir()
            if p.is_file()
            and p.suffix.lower() in EXTENSIONS
        ],
        key=lambda p: p.name.lower()
    )

    print("\nDAFTAR CITRA:")

    for file in files:
        print(f"- {file.name}")

    print(f"\nJumlah citra ditemukan: {len(files)}")

    if len(files) == 0:
        print("Masukkan gambar ke folder citra terlebih dahulu.")
        return

    # Menyimpan hasil untuk ditampilkan sekaligus
    display_results = []
    success_count = 0
    failed_count = 0

    # Memproses gambar satu per satu
    for number, path in enumerate(files, start=1):
        print("\n" + "-" * 65)
        print(f"Memproses citra {number}/{len(files)}")
        print(f"Nama file: {path.name}")

        try:
            image = cv2.imread(str(path))

            if image is None:
                raise ValueError("Gambar tidak dapat dibaca.")

            # -----------------------------------------
            # METODE 1: WATERSHED
            # -----------------------------------------

            boundary, mask, segmented = (
                watershed_segmentation(image)
            )

            save_image(
                WATERSHED_DIR / f"{path.stem}_batas.jpg",
                boundary
            )

            save_image(
                WATERSHED_DIR / f"{path.stem}_mask.png",
                mask
            )

            save_image(
                WATERSHED_DIR / f"{path.stem}_objek.jpg",
                segmented
            )

            # -----------------------------------------
            # METODE 2: EDGE-BASED
            # -----------------------------------------

            edges, overlay = edge_segmentation(image)

            save_image(
                EDGE_DIR / f"{path.stem}_canny.png",
                edges
            )

            save_image(
                EDGE_DIR / f"{path.stem}_overlay.jpg",
                overlay
            )

            # Konversi BGR menjadi RGB
            original_rgb = cv2.cvtColor(
                image, cv2.COLOR_BGR2RGB
            )

            boundary_rgb = cv2.cvtColor(
                boundary, cv2.COLOR_BGR2RGB
            )

            segmented_rgb = cv2.cvtColor(
                segmented, cv2.COLOR_BGR2RGB
            )

            overlay_rgb = cv2.cvtColor(
                overlay, cv2.COLOR_BGR2RGB
            )

            # Simpan grafik perbandingan setiap citra
            fig, axes = plt.subplots(
                2, 3, figsize=(15, 9)
            )

            fig.suptitle(
                f"Hasil Segmentasi: {path.name}",
                fontsize=14,
                fontweight="bold"
            )

            panels = [
                (original_rgb, "Citra Asli", None),
                (boundary_rgb, "Watershed - Batas", None),
                (mask, "Watershed - Mask", "gray"),
                (segmented_rgb, "Watershed - Objek", None),
                (edges, "Edge-Based - Canny", "gray"),
                (overlay_rgb, "Edge-Based - Overlay", None)
            ]

            for ax, (picture, title, color_map) in zip(
                axes.flat, panels
            ):
                if color_map:
                    ax.imshow(picture, cmap=color_map)
                else:
                    ax.imshow(picture)

                ax.set_title(title, fontsize=10)
                ax.axis("off")

            plt.tight_layout(rect=[0, 0, 1, 0.94])

            comparison_path = (
                COMPARISON_DIR
                / f"perbandingan_{path.stem}.png"
            )

            fig.savefig(
                comparison_path,
                dpi=130,
                bbox_inches="tight"
            )

            plt.close(fig)

            # Simpan untuk tampilan gabungan
            display_results.append({
                "name": path.name,
                "original": original_rgb,
                "watershed": boundary_rgb,
                "edges": edges,
                "overlay": overlay_rgb
            })

            success_count += 1
            print("Watershed: berhasil")
            print("Edge-Based: berhasil")
            print(f"Grafik disimpan: {comparison_path.name}")

        except Exception as error:
            failed_count += 1
            print(f"ERROR pada {path.name}: {error}")

    # =====================================================
    # 6. MENAMPILKAN SEMUA HASIL DALAM SATU JENDELA
    # =====================================================

    if display_results:
        print("\nMenampilkan hasil seluruh citra...")

        rows = len(display_results)

        fig, axes = plt.subplots(
            rows, 4,
            figsize=(16, 4 * rows),
            squeeze=False
        )

        fig.suptitle(
            "PERBANDINGAN 5 CITRA: WATERSHED VS EDGE-BASED",
            fontsize=17,
            fontweight="bold",
            y=0.995
        )

        for row, result in enumerate(display_results):
            panels = [
                (
                    result["original"],
                    "Citra Asli",
                    None
                ),
                (
                    result["watershed"],
                    "Watershed - Batas Merah",
                    None
                ),
                (
                    result["edges"],
                    "Edge-Based - Canny",
                    "gray"
                ),
                (
                    result["overlay"],
                    "Edge-Based - Overlay",
                    None
                )
            ]

            for col, (picture, title, color_map) in enumerate(
                panels
            ):
                ax = axes[row, col]

                if color_map:
                    ax.imshow(picture, cmap=color_map)
                else:
                    ax.imshow(picture)

                ax.set_title(
                    f"{result['name']}\n{title}",
                    fontsize=9
                )
                ax.axis("off")

        plt.tight_layout(rect=[0, 0, 1, 0.985])

        # Tampilkan seluruh citra sekaligus
        plt.show()

        plt.close(fig)

    # =====================================================
    # 7. RINGKASAN
    # =====================================================

    print("\n" + "=" * 65)
    print("PROSES SEGMENTASI SELESAI")
    print(f"Total citra ditemukan : {len(files)}")
    print(f"Berhasil diproses     : {success_count}")
    print(f"Gagal diproses        : {failed_count}")
    print(f"Hasil Watershed       : {WATERSHED_DIR}")
    print(f"Hasil Edge-Based      : {EDGE_DIR}")
    print(f"Grafik per citra      : {COMPARISON_DIR}")
    print("=" * 65)


if __name__ == "__main__":
    main()
