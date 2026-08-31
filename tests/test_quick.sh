#!/bin/bash
cd /home/lkm20/TCC
source .venv/bin/activate
rm -rf ./mobilenet_test_output

for model in v1 v2 v3small v3large v4conv; do
    echo ""
    echo "======================================"
    echo "  Testing $model (2 epochs)"
    echo "======================================"
    python train_timm_models.py \
        --model "$model" \
        --epochs 2 \
        --batch-size 48 \
        --lr-stage1 0.001 \
        --lr-stage2 0.0005 \
        --patience 20 \
        --output-dir ./mobilenet_test_output 2>&1 | \
        grep -E "(Classification Report|accuracy|akiec|bcc|bkl|df|mel|nv|vasc|^$)"
done

echo ""
echo "======================================"
echo "  Quick test complete!"
echo "======================================"
cat ./mobilenet_test_output/*/results.json 2>/dev/null
