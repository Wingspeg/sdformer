#!/bin/bash

# CCFormer Ablation Study on Weather
# 在Weather数据集上进行消融实验，测试各个组件的贡献

export CUDA_VISIBLE_DEVICES=0

model_name=SDFormer
data_name=custom
pred_len=96  # 使用96步预测进行消融实验

echo "=========================================="
echo "CCFormer Ablation Study on Weather"
echo "Prediction Length: ${pred_len}"
echo "=========================================="

# 基础配置
base_config="--task_name long_term_forecast \
  --is_training 1 \
  --root_path ./dataset/weather/ \
  --data_path weather.csv \
  --model ${model_name} \
  --data ${data_name} \
  --features M \
  --seq_len 512 \
  --label_len 48 \
  --pred_len ${pred_len} \
  --e_layers 1 \
  --d_layers 1 \
  --factor 3 \
  --enc_in 21 \
  --dec_in 21 \
  --c_out 21 \
  --d_model 512 \
  --d_ff 2048 \
  --n_heads 4 \
  --dropout 0.1 \
  --batch_size 32 \
  --learning_rate 0.0001 \
  --train_epochs 20 \
  --patience 7 \
  --loss Huber \
  --itr 1"

echo ""
echo "=========================================="
echo "1. Full Model (Baseline)"
echo "=========================================="
python -u run.py \
  --model_id weather_512_${pred_len}_full \
  --des 'Full' \
  ${base_config}

echo ""
echo "=========================================="
echo "2. w/o Independent Encoders (→ Shared Encoder)"
echo "=========================================="
python -u run.py \
  --model_id weather_512_${pred_len}_shared_encoder \
  --des 'SharedEncoder' \
  --use_shared_encoder 1 \
  ${base_config}

echo ""
echo "=========================================="
echo "3. w/o ATSR (→ Uniform Weights)"
echo "=========================================="
python -u run.py \
  --model_id weather_512_${pred_len}_uniform_atsr \
  --des 'UniformATSR' \
  --use_uniform_atsr 1 \
  ${base_config}

echo ""
echo "=========================================="
echo "4. w/o AGF (→ Equal-Weight Averaging)"
echo "=========================================="
python -u run.py \
  --model_id weather_512_${pred_len}_equal_fusion \
  --des 'EqualFusion' \
  --use_equal_fusion 1 \
  ${base_config}

echo ""
echo "=========================================="
echo "5. w/o Anomaly-Aware Norm (→ Standard RevIN)"
echo "=========================================="
python -u run.py \
  --model_id weather_512_${pred_len}_standard_norm \
  --des 'StandardNorm' \
  --use_standard_norm 1 \
  ${base_config}

echo ""
echo "=========================================="
echo "Ablation Study Completed!"
echo "=========================================="
