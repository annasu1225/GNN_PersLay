cd $(dirname $0)
cd ..

source ~/.bashrc
source ./scripts/utils.sh

root_path="$(pwd)/../../../.."
export PYTHONPATH="$root_path/":$PYTHONPATH

compound_encoder_config="model_configs/geognn_l8.json"
dataset="knot"
cached_data_path="./cached_data/$dataset"


if [ ! -f "$cached_data_path.done" ]; then
    echo "Preprocessing data...";
    python finetune_class.py \
                    --task=data \
                    --num_workers=10 \
                    --dataset_name="knot" \
                    --data_path="/gpfs/gibbs/pi/gerstein/as4272/KnotFun/datasets_for_geo/knotprot.json" \
                    --cached_data_path=$cached_data_path \
                    --compound_encoder_config=$compound_encoder_config \
                    --model_config="model_configs/down_mlp2.json"
    touch $cached_data_path.done
    else
        echo "Found preprocessed data... Training"
        batch_size=16
        # model_config="model_configs/down_mlp2.json"
        # lr=1e-3
        # head_lr=1e-3
        # dropout_rate=0.2
        init_model="./pretrain_models-chemrl_gem/class.pdparams"
        dataset="knot"
        log_prefix="log/pretrain"

        # model_config_list="model_configs/down_mlp2.json model_configs/down_mlp3.json"
	    # lrs_list="1e-3,1e-3 4e-3,4e-3 1e-4,1e-3"
	    # drop_list="0.2 0.5"
        
        model_config_list="model_configs/down_mlp3.json"
	    lrs_list="1e-4,1e-3"
	    drop_list="0.2"

        for model_config in $model_config_list; do
            for lrs in $lrs_list; do
                IFS=, read -r -a array <<< "$lrs"
                lr=${array[0]}
                head_lr=${array[1]}
                for dropout_rate in $drop_list; do
                    log_dir="$log_prefix-$dataset"
                    log_file="$log_dir/lr${lr}_${head_lr}-drop${dropout_rate}.txt"
                    echo "Outputs redirected to $log_file"
                    mkdir -p $log_dir

                    python make_embeddings.py \
                    --batch_size=$batch_size \
                    --max_epoch=100 \
                    --dataset_name=$dataset \
                    --data_path=$data_path \
                    --cached_data_path=$cached_data_path \
                    --split_type=random \
                    --compound_encoder_config=$compound_encoder_config \
                    --model_config=$model_config \
                    --init_model=$init_model \
                    --model_dir=./finetune_models/$dataset \
                    --encoder_lr=$lr \
                    --head_lr=$head_lr \
                    --dropout_rate=$dropout_rate 2>&1 | grep -v "Warning::" # >> $log_file
                done;
            done;
        done;

fi
