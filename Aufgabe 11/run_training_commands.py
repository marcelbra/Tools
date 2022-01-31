import os

"""
cd PycharmProjects/Tools; source venv/bin/activate; cd "Aufgabe 11 (Kopie)"; python3 run_training_commands.py
"""

# import torch.optim as optim
# from random import choice

# model_config = {"embeddings_dim": [100, 150, 200, 250],
#                 "word_encoder_hidden_dim": [100, 150, 200, 250],
#                 "span_encoder_hidden_dim": [100, 150, 200, 250],
#                 "fc_hidden_dim": [100, 150, 200, 250],
#                 "dropout": [0.0, 0.1, 0.2, 0.3, 0.4],
#                 "span_encoder_num_layers": [1, 2],
#                 "optimizer": ["optim.Adam", "optim.AdamW"],
#                 "lr": [eval(f"{i}e-3") for i in range(1,3)]}
# while True:
#     config = {key: choice(value) for key, value in model_config.items()}
# command = f"python3 train-parser.py " \
#           f"--embeddings_dim {config['embeddings_dim']} " \
#           f"--word_encoder_hidden_dim {config['word_encoder_hidden_dim']} " \
#           f"--span_encoder_hidden_dim {config['span_encoder_hidden_dim']} " \
#           f"--fc_hidden_dim {config['fc_hidden_dim']} --dropout {config['dropout']} " \
#           f"--span_encoder_num_layers {config['span_encoder_num_layers']} " \
#           f"--optimizer {config['optimizer']} " \
#           f"--lr {config['lr']} "



command = "python3 train-parser.py " \
          "--embeddings_dim 512 " \
          "--word_encoder_hidden_dim 1024 " \
          "--span_encoder_hidden_dim 1024 " \
          "--fc_hidden_dim 1024 " \
          "--dropout 0.0 " \
          "--span_encoder_num_layers 2 " \
          "--optimizer optim.Adam  " \
          "--lr 1e-4 " \
          "--patience 3 " \
          "--validation_amount 2 " \
          "--decay_from_step 9 " \
          "--lr_decay 0.9"

os.system(command)