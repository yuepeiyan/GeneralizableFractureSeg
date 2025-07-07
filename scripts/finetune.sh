CUDA_VISIBLE_DEVICES=5 python main.py \
        configs/finrtune.yaml \
        --json_list=pelvic_5training.json \
        --data_path=/data/ypy/Data/Pelvic/CTPelvic_dataset6 \
        --run_name=50pelvic+5pelvic \
        --pretrain=/data/ypy/Results/Tibia/Pretrain/ViT/pretain_50cases_pelvic_new/ckpts/checkpoint_9999.pth.tar