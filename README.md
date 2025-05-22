
<h1 align="center"> 🔧✨ Tool-Star: Empowering LLM-Brained Multi-Tool Reasoner via Reinforcement Learning</a></h1>


<div align="center"> 

[![Paper](https://img.shields.io/badge/Paper-arXiv-b5212f.svg?logo=arxiv)]()
[![Paper](https://img.shields.io/badge/Paper-Hugging%20Face-yellow?logo=huggingface)]()
[![License](https://img.shields.io/badge/LICENSE-MIT-green.svg)](https://opensource.org/licenses/MIT) 
[![Python 3.9+](https://img.shields.io/badge/Python-3.9+-blue.svg)](https://www.python.org/downloads/release/python-390/) 
[![X (formerly Twitter) URL](https://img.shields.io/twitter/url?url=https%3A%2F%2Fx.com%2FKevin_GuoweiXu%2Fstatus%2F1858338565463421244)]()
</div>

<p align="center">
🤗 <a href="https://huggingface.co/dongguanting/Tool-Star-Qwen-3B" target="_blank">Tool-Star-Qwen-3B</a> ｜
🤗 <a href="https://huggingface.co/datasets/dongguanting/Tool-Star-SFT-54K" target="_blank">Tool-Star-SFT-54K</a> ｜
🤗 <a href="https://github.com/dongguanting/Tool-Star/blob/main/Tool_Star_RL/mix_grpo/grpo_mix_train.parquet" target="_blank">Multi-Tool-RL-10K</a> ｜
</p>


<h5 align="center"> If you like our project, please give us a star ⭐ on GitHub for the latest update.</h5>

## 📣 Latest News
- **[May 21, 2025]**: 🔥 We released an our cold-star SFT and RL dataset for tool-integrated reasoning. Checkout **[🤗Tool-Star-SFT-54K](https://huggingface.co/datasets/dongguanting/Tool-Star-SFT-54K)** and **[Multi-Tool-RL-10K](https://github.com/dongguanting/Tool-Star/blob/main/Tool_Star_RL/mix_grpo/grpo_mix_train.parquet)** here.
- **[May 21, 2025]**: 🔥 We released our Tool-Star-Qwen-3B checkpoint. Checkout **[🤗Tool-Star-Qwen-3B](https://huggingface.co/dongguanting/Tool-Star-Qwen-3B)** here.
- **[May 21, 2025]**: 📄 Our paper is now available on **[arXiv]()** and **[Hugging Face]()** daily paper.
- **[May 21, 2025]**: 🚀 Full codebase released. Tool-Star supports multiple Tools with several open-source models like Qwen2.5-3B-Instruct.

## :mag_right: Roadmap

Tool-star is still under development and there are many issues and room for improvement. We will continue to update. And we also sincerely welcome contributions on this open-source toolkit.

- [ ] Support larger parameter size LLM (e.g. 7B, 32B)
- [ ] Provide More Tools for improving Inference


## 💡 Overview


**Tool-Star** is a **reinforcement learning-based framework** designed to empower LLMs to autonomously invoke **multiple external tools** during stepwise reasoning. Specifically, Tool-Star integrates six types of tools into the reasoning process (three for training and three for inference-time optimization) and incorporates systematic designs in both data synthesis and training algorithms.

<p align="center">
<img width="100%" alt="image" src="https://github.com/user-attachments/assets/edb21f58-7a18-47b9-8b1c-9522f7c9c56d" />
</p>

---

### 📊 Overall Performance
As shown below, Tool-Star demonstrates strong overall reasoning performance across more than **10** challenging computational reasoning tasks (e.g., AIME24 and MATH500) and knowledge-intensive reasoning tasks (e.g., WebWalker and HotpotQA), while ensuring both efficiency and reliability in tool usage.


<p align="center">
<img width="100%" alt="image" src="https://github.com/user-attachments/assets/5f60be15-6992-413e-a405-a72e3b352fc1" />
</p>




# 🏃 Quick Start for Training

## ❄️ Cold-Start SFT Stage

### 1. Environment Setup

In this step, we will describe how to perform a cold start for the SFT stage using the Llama Factory repository. Please first set up the environment for [Llama Factory](https://github.com/hiyouga/LLaMA-Factory).

1. Download your SFT dataset from [🤗Tool-Star-SFT-54K](https://huggingface.co/datasets/dongguanting/Tool-Star-SFT-54K) and place it in `LLaMA-Factory-main/data/final_sft_edition9.json`. Define the dataset in `dataset_info.json`.

2. Complete the path information in `LLaMA-Factory-main/examples/train_full/qwen_sft_tool_star.yaml`. The file content should be as follows:

```yaml
### model
model_name_or_path: {your_path_to_model}/Qwen2.5-3B-Instruct
trust_remote_code: true

### method
stage: sft
do_train: true
finetuning_type: full
deepspeed: examples/deepspeed/ds_z3_config.json  # choices: [ds_z0_config.json, ds_z2_config.json, ds_z3_config.json]

### dataset
dataset: final_sft_edition9
template: qwen
cutoff_len: 15000
max_samples: 1000000
overwrite_cache: true
preprocessing_num_workers: 16

### output
output_dir: {your_save_path}/Qwen2.5-3B-Instruct-final_sft_edition10-52
logging_steps: 10
save_steps: 2000
plot_loss: true
overwrite_output_dir: true

### train
per_device_train_batch_size: 1
gradient_accumulation_steps: 4
learning_rate: 7.0e-6
num_train_epochs: 3.0
lr_scheduler_type: cosine
warmup_ratio: 0.1
bf16: true
ddp_timeout: 180000000
```

After completing the information, you can fine-tune the model using the following command:

```bash
cd LLaMA-Factory-main
bash ./examples/train_full/train_sft.sh
```

---

## 🔥 Self-Critic RL Stage

In this step, we will load the cold-start data for GRPO training. We reference the [ReCall](https://github.com/Agent-RL/ReCall) and [VERL](https://github.com/volcengine/verl) frameworks for RL training.


### 1. Environment Setup

First, please set up the [ReCall environment](https://github.com/Agent-RL/ReCall). After that, install our environment:

We suggest that you follow the environment installation steps of [ReCall](https://github.com/Agent-RL/ReCall)(very good open-sourced codebase!). On this basis, you can install our additional environment as follow: 


```bash
# Create conda environment
conda create -n tool_star python=3.10
conda activate tool_star

# Install requirements
cd tool_star
pip install -r requirements.txt
```

### 2. Vanilla RL Training

Our training framework is based on [verl](https://github.com/volcengine/verl) and [ReCall](https://github.com/Agent-RL/ReCall). The training scripts can be found under `scripts/train`. First, you need to complete the information in `scripts/train/run_tool_star.sh`, 
we have provided both [train parquet](https://github.com/dongguanting/Tool-Star/blob/main/Tool_Star_RL/mix_grpo/grpo_mix_train.parquet) and [test parquet](https://github.com/dongguanting/Tool-Star/blob/main/Tool_Star_RL/mix_grpo/grpo_mix_test.parquet) for RL:

```bash
export PYTHONPATH=/src/verl:$PYTHONPATH
export MKL_SERVICE_FORCE_INTEL=1
export MKL_THREADING_LAYER=GNU

bash scripts/train/train.sh \
    --train_batch_size 128 \
    --ppo_mini_batch_size 16 \
    --rollout_n 8 \
    --apply_chat True \
    --prompt_template_name re_search_template_sys \
    --actor_model_path {your_actor_model_path} \
    --project_name {your_project_name} \
    --experiment_name {your_experiment_name} \
    --nnodes 1 \
    --n_gpus_per_node 8 \
    --save_freq 10 \
    --test_freq 10 \
    --total_epochs 2 \
    --wandb_api_key {your_wandb_api_key} \
    --save_path {your_save_path} \
    --train_files {path_to_train_file}/grpo_mix_train_shuffle.parquet \
    --test_files {path_to_test_file}/grpo_mix_test.parquet
```

Since the rollout process involves Bing web search calls, please configure the `deep_search_snippet()` function in `/src/verl/verl/workers/rollout/vllm_rollout/web_search/web_search_main.py` with your search API:

```python
def deep_search_snippet(search_query, top_k=10, use_jina=False, jina_api_key="empty", bing_subscription_key="your bing api key", bing_endpoint="https://api.bing.microsoft.com/v7.0/search"):
    args = Namespace(
        dataset_name='qa',
        split='test',
        subset_num=-1,
        max_search_limit=15,
        top_k=top_k,  
        use_jina=use_jina,  
        jina_api_key=jina_api_key,  
        temperature=0.7,
        top_p=0.8,
        min_p=0.05,
        top_k_sampling=20,
        repetition_penalty=1.05,
        max_tokens=4096,
        bing_subscription_key=bing_subscription_key, 
        bing_endpoint=bing_endpoint, 
        eval=False,
        seed=1742208600,
        concurrent_limit=200
    )
```

Replace `bing_subscription_key`, `bing_endpoint`, and `api_base_url` with your own values. Various web search modes are provided in this file for you to choose from.

You can then run the following script to start training:

```bash
cd ./Tool_Star_RL/scripts/train/
bash run_tool_star.sh
```

For the core code of the rollout process, please refer to `/src/verl/verl/workers/rollout/vllm_rollout/vllm_rollout.py`, and for the reward calculation part, refer to `/Tool_Star_RL/src/verl/verl/utils/reward_score`. You can modify them according to your needs.

### 3. Self-Critic DPO Training (Optional)

In our experiments, completing SFT + Vanilla RL has been sufficient to almost reproduce Tool-Star's performance (refer to the ablation study).

If you wish to proceed with Self-Critic DPO training, please refer to the training algorithm in **Appendix B.1** of the paper and the data format process in **Appendix E.2**. You can self-sample reward data using the saved checkpoints for RL and SFT training data. We also provide DPO training code based on [Llama Factory](https://github.com/hiyouga/LLaMA-Factory) for your reference.

Please complete the path information in `LLaMA-Factory-main/examples/train_lora/qwen_lora_dpo_2.yaml` and place the synthesized DPO data in `LLaMA-Factory-main/data/`. You can then run the following script for training:

```bash
cd LLaMA-Factory-main
bash ./examples/train_lora/train_dpo.sh
```

---

## ✅ TIR Evaluation

If you have already trained a model, you can refer to the following process for TIR capability evaluation. Of course, you can also download our checkpoint **[🤗Tool-Star-Qwen-3B](https://huggingface.co/dongguanting/Tool-Star-Qwen-3B)** for directly testing.

### 1. Environment Setup

```bash
# Create conda environment
conda create -n tool_star python=3.9
conda activate tool_star

# Install requirements
cd tool_star
pip install -r requirements.txt
```

### 2. LLM Service Deployment

In this step, we will use the VLLM framework to deploy additional large language models (LLMs). This includes deploying an LLM as a judging model to evaluate the accuracy of the generated answers in the subsequent steps, as well as deploying inference-time tools such as code debugging and chain refinement.

- We use Qwen2.5-72B-Instruct as the judging model.

- We use Qwen2.5-3B-Instruct, which has the same parameter scale as the base model, as the foundation for the inference-time tools.

For the specific deployment, you can refer to the following script.

```bash
cd evaluation
bash vllm_server.sh
```

### 3. Retriever Serving Deployment

In this section, we will deploy the retriever for performing search tasks on Wikipedia-based datasets. We provide a Wikipedia retriever service implemented using FlashRAG and FastAPI. Before starting the retriever serving, you need to download the [pre-indexed Wikipedia](https://github.com/RUC-NLPIR/FlashRAG?tab=readme-ov-file#index), [Wikipedia corpus, and corresponding retriever models](https://github.com/RUC-NLPIR/FlashRAG/blob/main/docs/original_docs/reproduce_experiment.md#preliminary). More details can be found in the FlashRAG documentation.

To start the retriever serving, first fill in `scripts/serving/retriever_config.yaml` with the correct paths to the retrieval model, index, and corpus, as well as available GPU IDs. Then, run the following command to start the retriever serving:

```bash
cd evaluation/search
python host_wiki.py \
    --config serving_config.yaml \
    --num_retriever {num_retriever} \
    --port {port}
```

### 4. Inference Your Model

In this section, we infer answers using a trained model. We support five types of mathematical reasoning datasets: AIME24, AIME25, GSM8K, MATH, and MATH500, as well as seven QA reasoning datasets: WebWalker, HotpotQA, 2WikiMultiHopQA, Bamboogle, MuSiQue, GAIA, and HLE. Due to resource constraints, all models and baselines will test a maximum of 500 samples for mathematical reasoning, 200 samples for all QA datasets, and 500 samples for HLE (please refer our code).

First, replace the API_URL and API key with your own in the following files:

In `evaluation/utils.py`:

```python
def search(query: str):
    if query == '':
        return 'invalid query'

    url = f'your_search_api_url'
    ...

def batch_search(query: Union[str, List[str]], top_n=5) -> List[str]:
    if len(query) == 0:
        return 'invalid query'

    url = f'your_search_api_url'
    ...
```

In `evaluation/tools/web_search_main.py`:

```python
def deep_search(search_query, top_k=10, use_jina=False, jina_api_key="empty", bing_subscription_key="xxxxx", bing_endpoint="xxxxx/search"):
    args = Namespace(
        dataset_name='qa',
        split='test',
        subset_num=-1,
        max_search_limit=15,
        top_k=top_k,  
        use_jina=use_jina,  
        jina_api_key=jina_api_key,  
        temperature=0.7,
        top_p=0.8,
        min_p=0.05,
        top_k_sampling=20,
        repetition_penalty=1.05,
        max_tokens=4096,
        bing_subscription_key=bing_subscription_key,  
        bing_endpoint=bing_endpoint,  
        eval=False,
        seed=1742208600,
        api_base_url='xxxxx',  
        model_name='search-agent',
        concurrent_limit=200
    )
    ...
```

In `evaluation/tools/debug_code.py`:

```python
def debug_code_function(code, error, api_key="your_api_key"):

    API_BASE_URL = api_key
    MODEL_NAME = "Qwen2.5-72B-Instruct"
    client = OpenAI(
        api_key="empty",
        base_url=API_BASE_URL,
    )
    ...
```

Then, start the inference. We recommend that you use the default parameters as:

```bash
cd evaluation
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export TOKENIZERS_PARALLELISM=true
export PYTHONPATH=/path/to/your_path:$PYTHONPATH
module load cuda/11.8
python run.py \
    --model_path /path/to/your_model_path \
    --dataset_name math \
    --task math \
    --gpu_use 0.95 \
    --max_tokens 16384 \ #you can change this, 8192 is enough for most tasks
    --max_input_len 16384 \ #you can change this, 8192 is enough for most tasks
    --output_path /path/to/your_results/your_exp_math_result.json \
    --counts 500 \
    --batch_size 100 \
    --use_debug 
```

**Parameter Explanations:**
- `--model_path`: Path to your model.
- `--dataset_name`: Name of your dataset (supports AIME24, AIME25, GSM8K, MATH, MATH500, WebWalker, HotpotQA, 2WikiMultiHopQA, Bamboogle, MuSiQue, GAIA, and HLE).
- `--task`: Set to `math` for mathematical reasoning datasets and `qa` for QA reasoning datasets.
- `--gpu_use`: GPU memory utilization.
- `--max_tokens`: Maximum number of tokens the model can generate.
- `--max_input_len`: Maximum input tokens the model can accept.
- `--output_path`: Path to save the results.
- `--counts`: Number of samples to take from the test set during testing.
- `--batch_size`: Batch size for parallel inference.
- `--use_debug`: Enable the debug mechanism.

**Additional Parameters（Optional）:**

In practical, only in the cases of HLE and GAIA is there a possibility of exceeding the length limit, you can use refiner. Generally, it won't occur in other situations.  

- `--use_rollback`: Whether to use the rollback mechanism.
- `--use_refiner`: Whether to use the refine mechanism.


In `evaluation/tools/refine_code.py`:

```python
def refine(prompt, response):

    API_BASE_URL = "your_api_base_url"
    MODEL_NAME = "Qwen2.5-7B-Instruct"
    client = OpenAI(
        api_key="empty",
        base_url=API_BASE_URL,
    )
    ...
```

### 5. Calculate Metrics

First, replace the API URL and API key with your own in the following file:

In `evaluation/evaluate/scripts/evaluate.py`:

```python
async def llm_evaluate_equivalence_batch(
    questions: List[str],
    labeled_answers: List[str], 
    pred_answers: List[str],
    api_base_url: str = None,
    model_name: str = None,
    api_key: str = "empty",
    concurrent_limit: int = 50,
    extract_answer: bool = False
) -> List[bool]:
    """
    Evaluate multiple answer pairs concurrently using LLM
    """
    if api_base_url is None:
        api_base_url = "http://114514.1919810/v1"
    if model_name is None:
        model_name = "Qwen2.5-72B-Instruct"
    ...
```

Replace `api_base_url` with the API_URL of your deployed model.

Then, run the following command:

```bash
cd evaluation
python evaluate/scripts/evaluate.py \
    --output_path /path/to/your_results/your_exp_math_result.json \
    --task math \
    --dataset_name math \
    --use_llm \
    --extract_answer
```

**Parameter Explanations:**
- `--output_path`: Path to save the results.
- `--task`: Set to `math` for mathematical reasoning datasets and `qa` for QA reasoning datasets.
- `--dataset_name`: Name of your dataset.
- `--use_llm`: Whether to use the LLM-as-judge mechanism.
- `--extract_answer`: Whether to use exact matching (removes \text and other redundant symbols).







## 📄 Citation

If you find this work helpful, please cite our paper:
```bibtex

```

## 🤝 Acknowledge

This training implementation builds upon [Llama Factory](https://github.com/hiyouga/LLaMA-Factory), [verl](https://github.com/volcengine/verl) and [ReCall](https://github.com/Agent-RL/ReCall). For evaluation, we rely on [WebThinker](https://github.com/RUC-NLPIR/WebThinker), [Search-o1](https://github.com/sunnynexus/Search-o1), and [FlashRAG](https://github.com/RUC-NLPIR/FlashRAG). The Python interpreter design references [ToRA](https://github.com/microsoft/ToRA) and [ToRL](https://github.com/GAIR-NLP/ToRL), while our models are trained using [Qwen2.5](https://qwenlm.github.io/blog/qwen2.5/). We express our sincere gratitude to these projects for their invaluable contributions to the open-source community. 


## 📄 License

This project is released under the [MIT License](LICENSE).

## 📞 Contact

For any questions or feedback, please reach out to us at [dongguanting@ruc.edu.cn](dongguanting@ruc.edu.cn).
