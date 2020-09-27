### run

1. download gpt-small model and put it into gpt2-small directory

   ```shell
   cd ./gpt2-small
   wget https://s3.amazonaws.com/models.huggingface.co/bert/gpt2-pytorch_model.bin
   ```



2. scp glove.6B.300d.model.bin from my server

   

3. pre-process raw data into cache for fast loading data during training and testing

   ```python
   python preprocess_emotion_empathetic.py
   ```



4. training and testing model

   ```python
   python train_eval_emotion_gpt.py \
   	--do-train \
   	--do-eval \
   	--seed 1024 \
   	--device-ids 0,1 \
   	--parallel
   ```

   if cuda memory is large enough, we don't have to use `--parallel`. And other hyper-parameters is written in `./config/config_emotion_gpt.py` 。

