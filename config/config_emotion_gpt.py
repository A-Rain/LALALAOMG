save_dir = "./save/"
data_dir = "./data/"
cache_dir = "./cache/"
model_path = "/home/yuhanliu/datasets/gpt2-small"
bert_path = "/home/yuhanliu/datasets/bert-base-en"
SPECIAL_tokens = {'pad_token': '[PAD]', 'eos_token': '[EOS]', 'additional_special_tokens': ['<eos0>', '<eos1>', '[cls1]', '[cls2]']}

# mutiple GPU setting
gpu0_bsz = 8
gpu_id = '2,3'
parallel = True

# topk emotion prediction, tau is used to control weight distribution
topK_emotion = 1
tau = 1


alpha = 0.1
beta = 0.5
smooth =5

dist_loss = 'mse'

# whether apply hinge loss to prob dist loss
use_hinge = True
hinge_phi = .5

normalize = False

# topk sample
topK_sample = 3

do_train = True
do_eval = True
use_gpu = True

max_sequence_length = 130
train_batch_size = 32
eval_batch_size = 16
num_train_epochs = 10

# seed for training
seed = 1024

# adam optimizer with warmup
learning_rate = 7e-5
weight_decay = 1e-6
adam_epsilon = 1e-8
warmup_proportion = 0.1

decoding_method = 'sampling'
# topk decoding
top_k = 5
top_p = 0.9
sampling_temperature = 5
num_samples = 1
max_decode_length = 20
min_decode_length = 2

# glove embedding path for computing embedding similarity
glove_path = '/home/liuyuhan/.cache/nlgeval/'

# VAD4 means 4 types of emotion, VAD7 means 7 types of emotion
emotion_type = 'VAD7'

# fine_grained or coarse_grained
emotion_cls = 'coarse'

# it means in epochs=k+1 it begin to calculate dist
calc_hid_dist_step = 4

# it means in epochs=k it only calculate nll
only_nll_step = 2
# it means in epochs=k it use gold emotion label
leak_emotion_step = 3
temperature = 1000
adapt = 'exp'
max_update_step = 7000