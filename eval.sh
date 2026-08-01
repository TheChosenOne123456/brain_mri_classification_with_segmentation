# python eval_kfold.py \
#   --config output/runs-hierarchical/train_config.py \
#   --data-root output/data-hdbet \
#   --checkpoint-root output/runs-hierarchical \
#   --seq 1 \
#   --model FoundationModelHierarchical \
  # --fold 1

# python eval_kfold.py \
#   --config output/runs-cross-entropy/train_config.py \
#   --data-root output/data-hdbet \
#   --checkpoint-root output/runs-cross-entropy \
#   --seq 1 \
#   --model FoundationModel_ori \

# python eval_kfold.py \
#   --config output/runs-hierarchical/train_config.py \
#   --data-root output/data-hdbet \
#   --checkpoint-root output/runs-hierarchical \
#   --seq 1 \
#   --model FoundationModelHierarchical \
#   --fold 2

# python eval_kfold.py \
#   --config output/runs-hierarchical/train_config.py \
#   --data-root output/data-hdbet \
#   --checkpoint-root output/runs-hierarchical \
#   --seq 1 \
#   --model FoundationModelHierarchical \
#   --fold 3

# python eval_kfold.py \
#   --config output/runs-hierarchical/train_config.py \
#   --data-root output/data-hdbet \
#   --checkpoint-root output/runs-hierarchical \
#   --seq 1 \
#   --model FoundationModelHierarchical \
#   --fold 4

# python eval_kfold.py \
#   --config output/runs-hierarchical/train_config.py \
#   --data-root output/data-hdbet \
#   --checkpoint-root output/runs-hierarchical \
#   --seq 1 \
#   --model FoundationModelHierarchical \
#   --fold 5

# python eval_kfold.py \
#   --config output/runs-hierarchical-lesion-head/train_config.py \
#   --data-root output/data-hdbet \
#   --checkpoint-root output/runs-hierarchical-lesion-head \
#   --seq 3 \
#   --model FoundationModelLesionAwareHierarchical

python eval_kfold.py \
  --config output/runs-cross-entropy/train_config.py \
  --data-root output/data-hdbet \
  --checkpoint-root output/runs-cross-entropy \
  --seq 3 \
  --model FoundationModel