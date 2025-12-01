#!/bin/bash

# Exit on error
set -e

# Ensure the assets directory exists
mkdir -p assets

# Define source directory relative to project root
SRC_DIR="../LaTeX/v6/ThreadWeaver__Adaptive_Threading_for_Efficient_Parallel_Reasoning_in_Language_Models/assets"

# Function to convert pdf to png and svg
convert_img() {
    local src="$1"
    local dest_png="$2"
    # local dest_svg="${dest_png%.png}.svg"
    
    if [ ! -f "$src" ]; then
        echo "Error: Source file not found: $src"
        return 1
    fi

    echo "Converting $src to PNG..."
    # Use pdftocairo for higher quality rasterization
    # -r 600 sets DPI to 600 (high res, but safe for cairo)
    # -singlefile ensures we get one png named exactly as specified (without page numbers)
    pdftocairo -png -singlefile -r 600 "$src" "${dest_png%.png}"

    # echo "Converting $src to SVG..."
    # pdftocairo -svg "$src" "$dest_svg"
}

echo "Starting image conversion..."

# Main figures
convert_img "$SRC_DIR/main_figure.pdf" "assets/main_figure.png"
convert_img "$SRC_DIR/teaser_accuracy.pdf" "assets/teaser_accuracy.png"
convert_img "$SRC_DIR/teaser_latency.pdf" "assets/teaser_latency.png"

# Speedup charts
convert_img "$SRC_DIR/correct_only/aime_32_speedup_ar_vs_pr.pdf" "assets/aime_speedup.png"
convert_img "$SRC_DIR/correct_only/math_1_speedup_ar_vs_pr.pdf" "assets/math_speedup.png"
convert_img "$SRC_DIR/correct_only/amc_8_speedup_ar_vs_pr.pdf" "assets/amc_speedup.png"
convert_img "$SRC_DIR/correct_only/olympiad_bench_8_speedup_ar_vs_pr.pdf" "assets/olympiad_speedup.png"

# Methodology figures
convert_img "$SRC_DIR/decomposition_inference.pdf" "assets/decomposition_inference.png"
convert_img "$SRC_DIR/decomposition_training.pdf" "assets/decomposition_training.png"
convert_img "$SRC_DIR/decomposition_training_two_parallel.pdf" "assets/decomposition_training_two_parallel.png"

# Appendix figures (Unfiltered speedups)
convert_img "$SRC_DIR/all/aime_32_speedup_ar_vs_pr.pdf" "assets/aime_speedup_all.png"
convert_img "$SRC_DIR/all/math_1_speedup_ar_vs_pr.pdf" "assets/math_speedup_all.png"
convert_img "$SRC_DIR/all/amc_8_speedup_ar_vs_pr.pdf" "assets/amc_speedup_all.png"
convert_img "$SRC_DIR/all/olympiad_bench_8_speedup_ar_vs_pr.pdf" "assets/olympiad_speedup_all.png"

echo "All assets generated successfully!"
