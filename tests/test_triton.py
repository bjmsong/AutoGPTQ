import os
import unittest

import torch
import torch.utils.benchmark as benchmark
from transformers import AutoTokenizer

import sys
sys.path.insert(0, "/root/autodl-tmp/AutoGPTQ")
# change folder name from [auto_gptq] to [auto_gptq_local]
from auto_gptq_local import AutoGPTQForCausalLM

MODEL_ID = "/root/autodl-tmp/model_dir"  # "TheBloke/Llama-7B-GPTQ"
DATASET_ID = "timdettmers/openassistant-guanaco"
LEARNING_RATE = 3e-5
MAX_SEQ_LEN = 10
BATCH_SIZE = 5
NUM_TRAIN_STEPS = 10

os.environ["TOKENIZERS_PARALLELISM"] = "false"

def benchmark_forward(
    fn,
    *inputs,
    repeats="auto",
    desc="",
    verbose=True,
    amp=False,
    amp_dtype=torch.float16,
    **kwinputs,
):
    if verbose:
        print(desc, "- Forward pass")

    def amp_wrapper(*inputs, **kwinputs):
        with torch.autocast(device_type="cuda", dtype=amp_dtype, enabled=amp):
            fn(*inputs, **kwinputs)

    t = benchmark.Timer(
        stmt="fn_amp(*inputs, **kwinputs)",
        globals={"fn_amp": amp_wrapper, "inputs": inputs, "kwinputs": kwinputs},
        num_threads=torch.get_num_threads(),
    )
    if repeats == "auto":
        m = t.blocked_autorange()
    else:
        m = t.timeit(repeats)
    if verbose:
        print(m)
    return t, m

def get_model_and_tokenizer(
    model_id=MODEL_ID,
    inject_fused_attention=False,
    inject_fused_mlp=False,
    **model_kwargs,
):
    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_ID,
        use_fast=True,
    )
    if not tokenizer.pad_token_id:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    model = AutoGPTQForCausalLM.from_quantized(
        model_id,
        trainable=True,
        inject_fused_attention=inject_fused_attention,
        inject_fused_mlp=inject_fused_mlp,
        disable_exllamav2=True,
        disable_exllama=True,
        **model_kwargs,
    )

    model.warmup_triton()
    return model, tokenizer

class TestTriton(unittest.TestCase):
    def test_triton_qlinear(self):
        ref_model, _ = get_model_and_tokenizer(
            model_id=MODEL_ID,
            use_triton=True,
            inject_fused_attention=False,
            inject_fused_mlp=False,
        )
        test_model, _ = get_model_and_tokenizer(
            model_id=MODEL_ID,
            use_tritonv2=True,
            inject_fused_attention=False,
            inject_fused_mlp=False,
        )
        hidden_size = ref_model.model.model.embed_tokens.weight.shape[1]
        test_data = torch.randn((1, 256, hidden_size), dtype=torch.float16).cuda()

        qlinear_ref = ref_model.model.model.layers[0].self_attn.q_proj
        qlinear_test = test_model.model.model.layers[0].self_attn.q_proj

        test_out = qlinear_test(test_data)
        ref_out = qlinear_ref(test_data)
        print(test_out)
        print(ref_out)
        # self.assertTrue(torch.allclose(test_out, ref_out))

        _, measure_triton = benchmark_forward(qlinear_ref, test_data, desc="Triton", verbose=True)
        _, measure_tritonv2 = benchmark_forward(qlinear_test, test_data, desc="Triton-v2", verbose=True)
        print(f"measure_triton: {measure_triton.mean}")
        print(f"measure_tritonv2: {measure_tritonv2.mean}")
        self.assertTrue(measure_tritonv2.mean < measure_triton.mean)

if __name__ == '__main__':
    unittest.main()