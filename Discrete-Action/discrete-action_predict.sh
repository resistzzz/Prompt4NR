# Activate environment
source activate recsys

# Path of framework
FW=$HOME/Prompt4NR
TEMPLATE=$FW/Discrete-Action

# Change directory to the model template
cd $TEMPLATE

# Copy datasets to temp folder to save read/write operations computational costs
export DATA_SET=/English_small
TMPDIR=$FW/DATA

# Check whether the GPU is available
# srun python -uc "import torch; print('GPU available?', torch.cuda.is_available())"

# Model to use
MODEL_NAME=bert-base-uncased
date=$(date '+%Y-%m-%d')
# date=2024-06-19

# Run python scripts that train model and predicts. Content of run_py3.sh
python3 -u predict.py --prompt_type original --cluster_data_avail True --data_path $TMPDIR$DATA_SET --model_name $MODEL_NAME --test_batch_size 100 --max_tokens 500 --model_file ./temp/$MODEL_NAME$DATA_SET/$date/BestModel.pt --log True --world_size 4
