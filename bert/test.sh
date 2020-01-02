python test.py --outdir ./output/ \
--test_path ../../fetch_data/all_data/dev.json \
--bert_pretrain ../../bert_base \
--checkpoint ./save_model/model.best.pt \
--name dev.jsonl


python test.py --outdir ./output/ \
--test_path ../../fetch_data/all_data/test.json \
--bert_pretrain ../../bert_base \
--checkpoint ./save_model/model.best.pt \
--name test.jsonl

python test.py --outdir ./output/ \
--test_path ../../fetch_data/all_data/train.json \
--bert_pretrain ../../bert_base \
--checkpoint ./save_model/model.best.pt \
--name train.jsonl