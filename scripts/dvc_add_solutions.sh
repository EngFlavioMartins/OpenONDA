#!/bin/bash
# Script to add simulation solution directories to DVC tracking
# This helps manage large simulation data across computers using DVC

set -e  # Exit on error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo "======================================="
echo "DVC Solution Directory Tracker"
echo "======================================="
echo ""

# Check if DVC is installed
if ! command -v dvc &> /dev/null; then
    echo -e "${RED}ERROR: DVC is not installed or not in PATH${NC}"
    echo "Install it with: pip install dvc"
    exit 1
fi

# Check if we're in a git repository
if ! git rev-parse --git-dir > /dev/null 2>&1; then
    echo -e "${RED}ERROR: Not in a git repository${NC}"
    exit 1
fi

# Check if DVC is initialized
if [ ! -d ".dvc" ]; then
    echo -e "${RED}ERROR: DVC is not initialized${NC}"
    echo "Initialize it with: dvc init"
    exit 1
fi

# Function to add a directory to DVC
add_to_dvc() {
    local dir=$1
    local dvc_file="${dir}.dvc"
    
    # Check if directory exists
    if [ ! -d "$dir" ]; then
        echo -e "${YELLOW}⚠ Skipping $dir (not found)${NC}"
        return
    fi
    
    # Check if already tracked by DVC
    if [ -f "$dvc_file" ]; then
        echo -e "${YELLOW}⚠ Skipping $dir (already tracked by DVC)${NC}"
        return
    fi
    
    # Check if directory is tracked by git
    if git ls-files --error-unmatch "$dir" > /dev/null 2>&1; then
        echo -e "${YELLOW}⚠ $dir is tracked by git, removing from git first...${NC}"
        git rm --cached -r "$dir" > /dev/null 2>&1 || true
    fi
    
    echo -e "${GREEN}✓ Adding $dir to DVC tracking...${NC}"
    dvc add "$dir"
    
    # Add .dvc file and .gitignore to git
    git add "$dvc_file" "$(dirname "$dir")/.gitignore" > /dev/null 2>&1 || true
}

# Count directories
total=0
added=0

echo "Searching for solution and referenceFlow directories..."
echo ""

# Find all solution directories
while IFS= read -r dir; do
    ((total++))
    if add_to_dvc "$dir"; then
        ((added++))
    fi
done < <(find tutorials -type d -name "solution" 2>/dev/null)

# Find all referenceFlow directories
while IFS= read -r dir; do
    ((total++))
    if add_to_dvc "$dir"; then
        ((added++))
    fi
done < <(find tutorials -type d -name "referenceFlow" 2>/dev/null)

echo ""
echo "======================================="
echo -e "${GREEN}Summary:${NC}"
echo "  Total directories found: $total"
echo "  Added to DVC: $added"
echo "  Skipped: $((total - added))"
echo "======================================="
echo ""

# Check git status
if git diff --cached --quiet; then
    echo -e "${YELLOW}No changes to commit${NC}"
else
    echo -e "${GREEN}Changes staged for commit:${NC}"
    git diff --cached --name-only
    echo ""
    echo "To commit these changes, run:"
    echo "  git commit -m \"Track simulation data with DVC\""
    echo ""
    echo "After committing, configure a DVC remote and push data:"
    echo "  dvc remote add -d storage <remote-path>"
    echo "  dvc push"
fi
