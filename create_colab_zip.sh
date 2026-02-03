#!/bin/bash
# Helper script to create a zip file for Google Colab upload

echo "=================================="
echo "Creating Colab Upload Package"
echo "=================================="

# Get the current directory name
CURRENT_DIR=$(pwd)
PROJECT_NAME=$(basename "$CURRENT_DIR")

echo "Project: $PROJECT_NAME"
echo "Location: $CURRENT_DIR"
echo ""

# Create a temporary directory with the required structure
TEMP_DIR="colab_upload"
rm -rf "$TEMP_DIR"
mkdir -p "$TEMP_DIR"

echo "Copying project files..."

# Copy all necessary folders
cp -r pipeline "$TEMP_DIR/" 2>/dev/null && echo "  ✓ pipeline/"
cp -r ingestion "$TEMP_DIR/" 2>/dev/null && echo "  ✓ ingestion/"
cp -r rag "$TEMP_DIR/" 2>/dev/null && echo "  ✓ rag/"
cp -r crystal "$TEMP_DIR/" 2>/dev/null && echo "  ✓ crystal/"
cp -r prediction "$TEMP_DIR/" 2>/dev/null && echo "  ✓ prediction/"
cp -r synthesis "$TEMP_DIR/" 2>/dev/null && echo "  ✓ synthesis/"
cp -r utils "$TEMP_DIR/" 2>/dev/null && echo "  ✓ utils/" || echo "  ⊘ utils/ (not found, optional)"

# Copy essential files
cp reaction.csv "$TEMP_DIR/" 2>/dev/null && echo "  ✓ reaction.csv"
cp requirements.txt "$TEMP_DIR/" 2>/dev/null && echo "  ✓ requirements.txt"
cp README.md "$TEMP_DIR/" 2>/dev/null && echo "  ✓ README.md"

echo ""
echo "Creating zip file..."

# Create the zip file
ZIP_NAME="colab_project.zip"
cd "$TEMP_DIR"
zip -r "../$ZIP_NAME" . -q
cd ..

# Clean up temp directory
rm -rf "$TEMP_DIR"

echo ""
echo "=================================="
echo "✓ Package created: $ZIP_NAME"
echo "=================================="
echo ""
echo "File size: $(du -h "$ZIP_NAME" | cut -f1)"
echo ""
echo "Next steps:"
echo "1. Open your Colab notebook"
echo "2. Run the upload cell (uncomment the upload code)"
echo "3. Select $ZIP_NAME when prompted"
echo "4. Files will be extracted automatically"
echo ""
echo "Or manual upload:"
echo "1. Click folder icon (📁) in Colab sidebar"
echo "2. Drag and drop these folders:"
echo "   - pipeline/"
echo "   - ingestion/"
echo "   - rag/"
echo "   - crystal/"
echo "   - prediction/"
echo "   - synthesis/"
echo "3. Upload reaction.csv"
echo ""
