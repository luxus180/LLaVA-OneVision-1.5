# LLaVA-OneVision-1.5: Fully Open Framework for Democratized Multimodal Training


[🤗 Mid-Training-Data (Uploading!)](https://huggingface.co/datasets/lmms-lab/LLaVA-One-Vision-1.5-Mid-Training-85M) | 
[🤗 Insturct-Data (Uploading!)](https://huggingface.co/datasets/lmms-lab/LLaVA-One-Vision-1.5-Insturct-26M) 

**LLaVA-OneVision1.5** introduces a novel family of **fully open-source** Large Multimodal Models (LMMs) that achieves **state-of-the-art performance**  with substantially **lower cost** through training on **native resolution** images.

1. **Superior Performance**
A family of fully open-source large multimodal models demonstrating **superior performance** across multiple multimodal benchmarks, **outperforming Qwen2.5-VL** in most evaluation tasks.

2. **High-Quality Data at Scale**
Meticulously curated **pre-training and SFT data** with rigorous filtering and quality control, achieving **superior data efficiency** with only **64B tokens**.
- Concept-balanced, highly diverse, high-quality caption data
- Comprehensive instruction fine-tuning data covering a wide range of tasks

3. **Ultra-Efficient Training Framework**
Complete end-to-end training framework designed for maximum efficiency:
- $16000 total budget for full model training on A100 GPUs  ($0.6 per GPU/Hour)
- 45% HFU efficiency in 8k context length
- Built on **MegatronLM** with support for **MoE**, **FP8**, and **long sequence parallelization**
- Optimized codebase for cost-effective scaling

- - [ ] Better data load balancing optimization
- - [ ] More efficient multimodal model parallelism strategy
- - [ ] FP8 training support cases/examples

4. **Fully Open Framework** for community access and reproducibility:
- ✅ High-quality pre-training & SFT data
- ✅ Complete training framework & code
- ✅ Training recipes & configurations
- ✅ Base & instruct model checkpoints
- ✅ Comprehensive training logs & metrics


## Model

| Model                  | #Vision Param | #Language Param | #Total Param | HF Link                                                                      |
|------------------------|---------------|-----------------|--------------|------------------------------------------------------------------------------|
| LLaVA-OV-1.5-4B-Instruct      | 0.3B          | 4.4B            | 4.7B         | [🤗 link]()                |
| LLaVA-OV-1.5-8B-Instruct      | 0.3B          | 8.2B            | 8.5B         | [🤗 link](https://huggingface.co/lmms-lab/LLaVA-OneVision-1.5-8B-Instruct) |


## Dataset

![Dataset Visualization](asset/dataset.jpg)


| Description | Link |
|-------------|------|
| Mid-training data for LLaVA-OneVision-1.5 | [🤗 Download (Uploading!)](https://huggingface.co/datasets/lmms-lab/LLaVA-One-Vision-1.5-Mid-Training-85M) |
| SFT data for LLaVA-OneVision-1.5 | [🤗 Download (Uploading!)](https://huggingface.co/datasets/lmms-lab/LLaVA-OneVision-1.5-Insturct-Data) |


## Evaluation Results


All evaluations were conducted using lmms_eval.

|                                  | **LLaVA-OV-1.5-8B** | **Qwen2.5 VL 7B** | **LLaVA-OV-1.5-4B** | **Qwen2.5 VL 3B** |
|:----------------------------------|:---------------:|:-------------:|:---------------:|:-------------:|
| MMMU (Validation)                 |    **55.44**    |     51.33     |    **51.44**    |     46.44     |
| MMMU-Pro (Standard)               |    **37.40**    |     36.30     |    **33.24**    |     31.10     |
| MMMU-Pro (Vision)                 |      25.15      |   **32.83**   |    **23.53**    |     21.27     |
| MMBench (English; Test)           |    **84.14**    |     83.40     |    **82.29**    |     77.97     |
| MMBench (Chinese; Test)           |      81.00      |   **81.61**   |    **76.73**    |     74.55     |
| MME-RealWorld (English)           |    **62.31**    |     57.33     |    **57.16**    |     51.60     |
| MME-RealWorld (Chinese)           |    **56.11**    |     51.50     |      21.38      |   **45.38**   |
| AI2D (With Mask)                  |    **84.16**    |     82.58     |    **84.62**    |     78.56     |
| AI2D (Without Mask)               |    **94.11**    |     93.36     |    **92.84**    |     90.74     |
| CV-Bench                          |    **80.82**    |     79.95     |    **74.00**    |     71.53     |
| VL-RewardBench                    |      45.90      |   **49.65**   |    **45.90**    |     42.06     |
| V*                                |    **78.01**    |     76.96     |      66.49      |   **69.63**   |
| PixmoCount                        |      62.19      |   **63.33**   |    **59.17**    |     50.85     |
| CountBench                        |    **88.19**    |     86.35     |    **77.80**    |     72.51     |
| ChartQA                           |    **86.48**    |     84.08     |    **85.11**    |     83.36     |
| CharXiv (Direct Questions)        |    **74.10**    |     69.80     |    **70.70**    |     58.20     |
| DocVQA (Test)                     |    **95.00**    |     94.93     |    **93.48**    |     92.67     |
| InfoVQA (Test)                    |      78.42      |   **81.67**   |    **75.27**    |     75.63     |
| WeMath                            |    **33.62**    |     33.33     |    **28.00**    |     18.38     |
| MathVista (Mini)                  |    **69.57**    |     68.60     |    **67.36**    |     60.23     |
| MathVision                        |    **25.56**    |     22.37     |    **22.76**    |     21.25     |
| MMStar                            |    **67.72**    |     62.54     |    **64.22**    |     55.86     |
| SEED-Bench (Image)                |      77.32      |   **77.53**   |    **76.74**    |     74.81     |
| ScienceQA                         |    **94.98**    |     88.75     |    **92.05**    |     83.33     |
| SEED-Bench 2-Plus                 |      69.21      |   **70.93**   |    **68.42**    |     68.64     |
| OCRBench                          |      82.90      |   **84.20**   |      77.80      |   **79.20**   |
| RealWorldQA                       |      68.10      |   **68.50**   |    **64.05**    |     60.00     |


## Quick Start with HuggingFace

```python
from transformers import AutoTokenizer, AutoProcessor, AutoModelForCausalLM
from qwen_vl_utils import process_vision_info
model_path = "lmms-lab/LLaVA-One-Vision-1.5-8B-Instruct"

# default: Load the model on the available device(s)
model = AutoModelForCausalLM.from_pretrained(
    model_path, torch_dtype="auto", device_map="auto", trust_remote_code=True
)

# default processer
processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)

messages = [
    {
        "role": "user",
        "content": [
            {
                "type": "image",
                "image": "https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen-VL/assets/demo.jpeg",
            },
            {"type": "text", "text": "Describe this image."},
        ],
    }
]

# Preparation for inference
text = processor.apply_chat_template(
    messages, tokenize=False, add_generation_prompt=True
)
image_inputs, video_inputs = process_vision_info(messages)
inputs = processor(
    text=[text],
    images=image_inputs,
    videos=video_inputs,
    padding=True,
    return_tensors="pt",
)
inputs = inputs.to("cuda")

# Inference: Generation of the output
generated_ids = model.generate(**inputs, max_new_tokens=1024)
generated_ids_trimmed = [
    out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
]
output_text = processor.batch_decode(
    generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
)
print(output_text)

```

## Evaluation
```
# pip install git+https://github.com/EvolvingLMMs-Lab/lmms-eval.git  

accelerate launch --num_processes=8 --main_process_port 12399 -m lmms_eval \
    --model=llava_onevision1_5 \
    --model_args=pretrained=lmms-lab/LLaVA-OneVision-1.5-8B-Instruct,attn_implementation=flash_attention_2,max_pixels=3240000 \
    --tasks=mmmu_val,mmmu_pro_standard,mmbench_en_test,mmerealworld,mmerealworld_cn,ai2d,ai2d_no_mask,vstar_bench,chartqa,charxiv,docvqa_test,mathvista_testmini,mmstar,scienceqa \
    --batch_size=1
```


## Quick Start Guide

### 1.🐳 Docker (Recommended)

We strongly recommend using the docker environment for a seamless experience. The following instructions are tailored for the A100 80GB GPU environment.


```bash
# Clone repository
git clone https://github.com/EvolvingLMMs-Lab/LLaVA-OneVision-1.5.git
cd LLaVA-OneVision-1.5

docker build -t llava_megatron:25.04 .

# Run container with -w to set working directory directly to the mounted volume
docker run -it --gpus all \
    --ipc host --net host --privileged --cap-add IPC_LOCK \
    --ulimit memlock=-1 --ulimit stack=67108864 --rm \
    -v $(pwd):/workspace/LLaVA-OneVision-1.5 \
    -w /workspace/LLaVA-OneVision-1.5 \
    --name "llava_megatron_container" \
    llava_megatron:25.04 /bin/bash
```

### 2. Checkpoint and Format Conversion

You have two options to get started with LLaVA-OneVision-1.5-stage-0:

#### Option 1: Download pre-trained model from HuggingFace
Download our `LLaVA-OneVision-1.5-4B-stage0` model directly from [HuggingFace](https://huggingface.co/lmms-lab/LLaVA-OneVision-1.5-4B-stage0).

#### Option 2: Merge initial weights yourself
Alternatively, you can merge the initial weights from the original ViT and LLM:
```bash
python ds/merge_model.py \
--vit_path DeepGlint-AI/rice-vit-large-patch14-560 \
--llm_path Qwen/Qwen3-4B-Instruct-2507 \
--output LLaVA-OneVision-1.5-4B-stage0
```
Note: When merging weights, the adapter component will be initialized with default values.

Convert the model from HuggingFace format to Megatron format:

```bash
AIAK_TRAINING_PATH=/workspace/LLaVA-OneVision-1.5 bash examples/llava_ov_1_5/convert/convert_4b_hf_to_mcore.sh \
LLaVA-OneVision-1.5-4B-stage0 \
LLaVA-OneVision-1.5-4B-stage0_mcore_tp1_pp1 \
1 1
```

### 3. Stage 1 Alignment-Training

Download LLaVA from [LLaVA-558K-Webdataset](https://huggingface.co/datasets/lmms-lab/LLaVA-558K-Webdataset).


```bash
# ============================================================
# Required environment variables:
#   AIAK_TRAINING_PATH  Root directory of the AIAK-Training-LLM project
#   DATA_PATH           Directory with WebDataset shards (.tar) for pretraining
#   TOKENIZER_PATH      Hugging Face tokenizer directory
#   CHECKPOINT_PATH     Megatron-formatted checkpoint directory (e.g., mcore TP1/PP1)
#   SAVE_CKPT_PATH      Output directory for saving training checkpoints
AIAK_TRAINING_PATH=/workspace/LLaVA-OneVision-1.5 \
DATA_PATH=LLaVA-558K-Webdataset \
TOKENIZER_PATH=LLaVA-OneVision-1.5-4B-stage0 \
CHECKPOINT_PATH=LLaVA-OneVision-1.5-4B-stage0_mcore_tp1_pp1 \
bash examples/llava_ov_1_5/quick_start/stage_1_alignment_llava_ov_4b.sh
```

### 4. Stage 1.5 Mid-Training 

Download our lightweight packed subset from [LLaVA-OneVision-1.5-Mid-Training-Quick-Start-3M-Webdataset](https://huggingface.co/datasets/lmms-lab/LLaVA-OneVision-1.5-Mid-Training-Webdataset-Quick-Start-3M).

```bash
# ============================================================
# Convert model to release format
bash examples/llava_ov_1_5/convert/convert_4b_mcore_to_release.sh \
stage_1_alignment_llava_ov_4b/iter_0002500/ \
stage_1_alignment_llava_ov_4b_release 1 1
# ============================================================
# Launch
AIAK_TRAINING_PATH=/workspace/LLaVA-OneVision-1.5 \
DATA_PATH=LLaVA-OneVision-1.5-Mid-Training-Quick-Start-3M-Webdataset \
TOKENIZER_PATH=LLaVA-OneVision-1.5-4B-stage0 \
CHECKPOINT_PATH=stage_1_alignment_llava_ov_4b_release \
bash examples/llava_ov_1_5/quick_start/stage_1.5_mid_training_llava_ov_4b.sh
```


### 5. Stage 2 Instruct-Training

Download LLaVA-NeXT-780k-webdataset at [LLaVA-NeXT-780K Dataset](https://huggingface.co/datasets/lmms-lab/LLaVA-NeXT-780k-webdataset).

```bash
# ============================================================
# Convert model to release format
bash examples/llava_ov_1_5/convert/convert_4b_mcore_to_release.sh \
stage_1.5_mid_training_llava_ov_4b/iter_0020000/ \
stage_1.5_mid_training_llava_ov_4b_release 1 1
# ============================================================
# # Launch
AIAK_TRAINING_PATH=/workspace/LLaVA-OneVision-1.5 \
DATA_PATH=LLaVA-NeXT-780k-Webdataset \
TOKENIZER_PATH=LLaVA-OneVision-1.5-4B-stage0 \
CHECKPOINT_PATH=stage_1.5_mid_training_llava_ov_4b_release \
bash examples/llava_ov_1_5/quick_start/stage_2_instruct_llava_ov_4b.sh
```


### 6. Convert mcore to huggingface
```bash
AIAK_TRAINING_PATH=/workspace/LLaVA-OneVision-1.5 \
bash examples/llava_ov_1_5/convert/convert_4b_mcore_to_hf.sh \
stage_2_instruct_llava_ov_4b/iter_0003500 \
LLaVA-OneVision-1.5-4B-3M-Mid-Training-780K-Instruct \
1 1
# Copy non-model files (e.g., tokenizer config) to the new directory
find LLaVA-OneVision-1.5-4B-stage0/ -type f -not -iname '*safetensors*' -exec cp {}  LLaVA-OneVision-1.5-4B-3M-Mid-Training-780K-Instruct/ ';'
```

### 7. Evaluation
```bash
# pip install git+https://github.com/EvolvingLMMs-Lab/lmms-eval.git
CUDA_VISIBLE_DEVICES=4,5,6,7 accelerate launch \
--num_processes=4 --main_process_port 12399 -m lmms_eval --model=llava_onevision1_5 --batch_size=1 --tasks=mme \
--model_args=pretrained=/workspace/LLaVA-OneVision-1.5/LLaVA-OneVision-1.5-4B-3M-Mid-Training-780K-Instruct,max_pixels=3240000
```

## Fully Reproducing Guide

### 1. Data Processing
To improve model training efficiency, we implement offline sample packing:

1. Download the [lmms-lab/LLaVA-One-Vision-1.5-Mid-Training-85M dataset](https://huggingface.co/datasets/lmms-lab/LLaVA-One-Vision-1.5-Mid-Training-85M)
2. Pack the mid-training data into webdataset format, For detailed instructions, refer to [examples/llava_ov_1_5/sample_packing/README.md](examples/llava_ov_1_5/sample_packing/README.md)


### 2. Training
More detailed reproduction steps for the complete process will be provided after the dataset upload is completed.

## Roadmaps

Q4 2025 Key Deliverables:

1. **Ultra-efficient MoE Training**  
2. **Full Video Input LLM**  

## Citation

If you find *LLaVA-OneVision-1.5* useful in your research, please consider to cite the following related papers:

```
@inproceedings{LLaVA-OneVision-1.5,
  title={LLaVA-OneVision-1.5: Fully Open Framework for Democratized Multimodal Training},
  author={LLaVA Community Contributors},
  booktitle={arxiv},  
  year={2025}
 }

@inproceedings{xie2025region,
  title={Region-based Cluster Discrimination for Visual Representation Learning},
  author={Xie, Yin and Yang, Kaicheng and An, Xiang and Wu, Kun and Zhao, Yongle and Deng, Weimo and Ran, Zimin and Wang, Yumeng and Feng, Ziyong and Miles, Roy and Elezi, Ismail and Deng, Jiankang},
  booktitle={ICCV},
  year={2025}
}

@article{lillava,
  title={LLaVA-OneVision: Easy Visual Task Transfer},
  author={Li, Bo and Zhang, Yuanhan and Guo, Dong and Zhang, Renrui and Li, Feng and Zhang, Hao and Zhang, Kaichen and Zhang, Peiyuan and Li, Yanwei and Liu, Ziwei and Li, Chunyuan},
  journal={Transactions on Machine Learning Research}
  year={2024}
}
```

## Acknowledgement

We extend our sincere gratitude to **AIAK team of the** [**Baige AI computing platform**](https://cloud.baidu.com/product/aihc.html) **from Baidu AI Cloud** for providing the exceptional training framework. The outstanding capabilities of AIAK-Training-LLM and AIAK-Megatron have significantly accelerated our training process with remarkable efficiency. These cutting-edge frameworks have been instrumental in achieving our research goals. `To get full AIAK support, you can contact Baidu Cloud.`


We also thank the maintainers and contributors of the following open-source projects, whose work greatly inspired and supported our research:

- LLaVA: Large Language-and-Vision Assistant — [LLaVA](https://github.com/haotian-liu/LLaVA)
- LLaVA-NeXT: Next-generation multi-modal assistant — [LLaVA-NeXT](https://github.com/LLaVA-VL/LLaVA-NeXT)
- lmms-eval: A standardized evaluation framework for Large Multimodal Models — [lmms-eval](https://github.com/EvolvingLMMs-Lab/lmms-eval)
- Megatron-LM: Efficient, scalable training for large language models — [Megatron-LM](https://github.com/NVIDIA/Megatron-LM)
- Qwen2.5-VL: Strong vision-language foundation model — [Qwen2.5-VL](https://github.com/QwenLM/Qwen2.5-VL)
- Qwen3: Next-generation Qwen LLM — [Qwen](https://github.com/QwenLM/Qwen)
- MetaCLIP: Scalable contrastive pretraining — [MetaCLIP](https://github.com/facebookresearch/MetaCLIP)
