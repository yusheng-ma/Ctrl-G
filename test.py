import os
import json
device = 'cuda'
os.environ['CUDA_VISIBLE_DEVICES'] = '0' # set your cuda device
os.environ['TOKENIZERS_PARALLELISM'] = 'false'

import torch
import ctrlg
from transformers import AutoModelForCausalLM, AutoTokenizer, LogitsProcessorList

# load the pretrained base_model and hmm_model; see README.md for a complete list of 
# released checkpoints. note that the hmm_model and base_model must share the same 
# vocabulary of tokens: i.e., one cannot apply hmm_gpt2-large_common-gen_4096 to 
# tulu2-7b_writing-prompts. To apply Ctrl-G to a custom base_model or to achieve 
# best performance on a specific domain, users would need to distill an hmm_model
# from the base_model. Please refer to tutorial_distillation.ipynb for details.
BASE_MODEL_PATH = f'ctrlg/gpt2-large_common-gen' # a gpt2-large checkpoint domain adapted to the common-gen corpus
HMM_MODEL_PATH = f'ctrlg/hmm_gpt2-large_common-gen_4096' # alternatively 'ctrlg/hmm_gpt2-large_common-gen_32768' for better quality

base_model = AutoModelForCausalLM.from_pretrained(BASE_MODEL_PATH).to(device)
base_model.eval()
tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_PATH)
hmm_model = ctrlg.HMM.from_pretrained(HMM_MODEL_PATH).to(device)

vocab_size = hmm_model.vocab_size

token_id_to_token = {}

for token_id in range(0, vocab_size):
    token = tokenizer.decode([tokenizer.all_special_ids[0], token_id])
    token = token[len(tokenizer.decode(tokenizer.all_special_ids[0])):]

    try:
        token_str = str(token)
    except:
        token_str = None

    token_id_to_token[token_id] = token_str

# === 分開輸出特殊與一般 token 的 mapping（更新版）===

# 定義要檢查的特殊字元（不含空格，因需特殊判斷）
special_chars = {'{', '}', ':', '"'}

with open("special_tokens.txt", "w", encoding="utf-8") as f_special, \
     open("normal_tokens.txt", "w", encoding="utf-8") as f_normal:

    f_special.write(f"{'ID':<8} {'Token (repr)':<40} {'Comment'}\n")
    f_special.write("-" * 80 + "\n")

    f_normal.write(f"{'ID':<8} {'Token (repr)':<40} {'Comment'}\n")
    f_normal.write("-" * 80 + "\n")

    for token_id in range(vocab_size):
        token_str = token_id_to_token[token_id]
        token_repr = repr(token_str)

        if token_str is None:
            f_normal.write(f"{token_id:<8} {token_repr:<40} # None/invalid token\n")
            continue

        has_special = False

        # 檢查 {, }, :, " 是否存在
        if any(char in token_str for char in special_chars):
            has_special = True
        else:
            # 檢查空格是否出現在「非第一個位置」
            if ' ' in token_str:
                # 找出所有空格的位置
                space_positions = [i for i, ch in enumerate(token_str) if ch == ' ']
                # 如果有任何一個空格不在 index 0（即不是第一個字元）
                if any(pos > 0 for pos in space_positions):
                    has_special = True

        line = f"{token_id:<8} {token_repr:<40} # \n"
        (f_special if has_special else f_normal).write(line)

print("✅ Special tokens (with {, }, :, \", or internal/late space) saved to 'special_tokens.txt'")
print("✅ Normal tokens saved to 'normal_tokens.txt'")