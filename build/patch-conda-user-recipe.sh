#!/bin/bash
# Usage: ./build/patch-conda-user-recipe.sh ./build/conda-user-recipe/physo/meta.yaml

# Check if file path is provided
if [ -z "$1" ]; then
  echo "Error: Please provide the path to meta.yaml"
  echo "Usage: $0 /path/to/meta.yaml"
  exit 1
fi

YAML_FILE="$1"

# Backup original file (safety measure)
cp "$YAML_FILE" "${YAML_FILE}.bak"

# Apply modifications
sed -i.bak '
# Replace torch with pytorch
s/^    - torch/    - pytorch/g

# Add noarch after build:
/^build:/a \
  noarch: python

# Replace github placeholder
s/your-github-id-here/WassimTenachi/g
' "$YAML_FILE"

# Verify changes
echo "=== Updated $YAML_FILE ==="
echo "Changes:"
diff -u "${YAML_FILE}.bak" "$YAML_FILE" || true

echo "Done. Backup saved as ${YAML_FILE}.bak"