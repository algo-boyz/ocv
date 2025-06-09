#!/usr/bin/env just --justfile
# -----------------------------------------------------------------------------
# This justfile is designed to be cross-platform for Linux and macOS.
# It automatically detects the OS and sets the correct variables for shared
# libraries and linker paths.
#
# It assumes a directory structure like:
# .
# ├── demo/
# ├── bindings/ocv.cpp
# ├── build/
# └── justfile
#
# OpenCV 4 must be installed on your system (e.g., `brew install opencv`).
# -----------------------------------------------------------------------------

# Set a strict shell for all recipes.
set shell := ["bash", "-euo", "pipefail", "-c"]

# --- Directories ---
# Define project structure for clarity and easy modification.
BUILD_DIR := "build"
CPP_SRC_DIR := "ocv"
ODIN_SRC_DIR := "demo"

# --- OS & Build Configuration ---
# Detect the operating system to set platform-specific variables.
os := os()

# Set variables based on the detected operating system.
# For Linux, we use .so and LD_LIBRARY_PATH.
# For macOS, we use .dylib and DYLD_LIBRARY_PATH.
shared_ext   := if os == "macos" { ".dylib" } else { ".so" }
lib_path_var := if os == "macos" { "DYLD_LIBRARY_PATH" } else { "LD_LIBRARY_PATH" }

# Define full shared library and object filenames using variables.
shared_lib_file := "ocv" + shared_ext
object_file     := "ocv.o"
static_lib_file := "ocv.a"
odin_exe_file   := "odincv"

# --- Aliases & Default ---
# Default recipe to run when 'just' is called without arguments.
default: run_odin

# Alias 'run' to 'run_odin' for convenience.
alias run := run_odin

# --- Recipes ---

# Clean up all generated files and recreate the build directory.
clean:
    @echo "🧹 Cleaning up generated files..."
    @rm -rf {{BUILD_DIR}}
    @mkdir -p {{BUILD_DIR}}

# Compile the C++ code into a position-independent object file.
# `pkg-config --cflags` provides the necessary include paths for OpenCV.
# Prerequisite: Ensure the build directory exists.
object: clean
    @echo "📦 Compiling C++ object file..."
    gcc -c -Wall -Werror -fpic `pkg-config --cflags opencv4` \
        -o {{BUILD_DIR}}/{{object_file}} {{CPP_SRC_DIR}}/bindings.cpp

# Create a static library (.a) from the object file.
static: object
    @echo "📚 Creating static library..."
    ar rcs {{BUILD_DIR}}/{{static_lib_file}} {{BUILD_DIR}}/{{object_file}}

# Create a shared library (.so or .dylib) from the object file.
# `pkg-config --libs` provides the necessary library linkage for OpenCV.
shared: object
    @echo "🔗 Creating shared library ({{shared_lib_file}})..."
    gcc -shared -o {{BUILD_DIR}}/{{shared_lib_file}} {{BUILD_DIR}}/{{object_file}} \
        `pkg-config --libs opencv4` -lstdc++

# Build the Odin example, linking against our libraries.
build_odin: shared static
    @echo "🏗️ Building Odin example..."
    odin build {{ODIN_SRC_DIR}} \
        -out:{{BUILD_DIR}}/{{odin_exe_file}} \
        -build-mode:exe \
        -extra-linker-flags:"-L{{BUILD_DIR}} `pkg-config --libs opencv4` -lstdc++"

# Run the Odin example, setting the correct library path.
run_odin: build_odin
    @echo "🚀 Running Odin example..."
    {{lib_path_var}}={{BUILD_DIR}} {{BUILD_DIR}}/{{odin_exe_file}}
