CUDA_VISIBLE_DEVICES=7 python recommender_Pytorch.py ./data/u5.base ./data/u5.test \
--output_path ./test_net/ \
--model CFNet \
--factor 5 \
--lr 1e-3 \
--wd 1e-4 \
--epochs 10000 \
--log_step 100 \
--print_log True \
