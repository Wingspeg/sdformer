export CUDA_VISIBLE_DEVICES=0

model_name=SDFormer

for pred_len in 96 192 336 720; do
  python -u run.py \
    --task_name long_term_forecast \
    --is_training 1 \
    --root_path ./dataset/ETT-small/ \
    --data_path ETTh2.csv \
    --model_id ETTh2_512_${pred_len}_nh8 \
    --model $model_name \
    --data ETTh2 \
    --features M \
    --seq_len 512 \
    --label_len 48 \
    --pred_len ${pred_len} \
    --e_layers 1 \
    --d_layers 1 \
    --factor 3 \
    --enc_in 7 \
    --dec_in 7 \
    --c_out 7 \
    --d_model 512 \
    --d_ff 2048 \
    --n_heads 8 \
    --dropout 0.1 \
    --batch_size 32 \
    --learning_rate 0.0001 \
    --train_epochs 20 \
    --patience 7 \
    --loss Huber \
    --des 'Exp' \
    --itr 1
done
