# export HF_ENDPOINT=https://hf-mirror.com

from huggingface_hub import snapshot_download

snapshot_download(repo_id='TheBloke/LLaMa-7B-GPTQ',
                  repo_type='model',
                  local_dir='../model_dir',
                  resume_download=True)