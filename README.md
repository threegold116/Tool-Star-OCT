
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
</p>


<h5 align="center"> If you like our project, please give us a star ⭐ on GitHub for the latest update.</h5>

## 📣 Latest News
- **[May 21, 2025]**: 🔥 We released an our cold-star SFT and RL dataset for tool-integrated reasoninhg. Checkout **[🤗Tool-Star-SFT-54K](https://huggingface.co/datasets/dongguanting/Tool-Star-SFT-54K)** and **[RL-dataset]()** here.
- **[May 21, 2025]**: 📄 Our paper is now available on **[arXiv]()** and **[Hugging Face]()**.
- **[May 21, 2025]**: 🚀 Full codebase released. Tool-Star supports multiple Tools with several open-source models like Qwen2.5-3B-Instruct.



## 💡 Overview


**Tool-Star** is a **reinforcement learning-based framework** designed to empower LLMs to autonomously invoke **multiple external tools** during stepwise reasoning. Specifically, Tool-Star integrates six types of tools into the reasoning process (three for training and three for inference-time optimization) and incorporates systematic designs in both data synthesis and training algorithms.

<p align="center">
<img width="100%" alt="image" src="https://github.com/user-attachments/assets/edb21f58-7a18-47b9-8b1c-9522f7c9c56d" />
</p>


### 📊 Overall Performance
As shown below, Tool-Star demonstrates strong overall reasoning performance across more than **10** challenging computational reasoning tasks (e.g., AIME24 and MATH500) and knowledge-intensive reasoning tasks (e.g., WebWalker and HotpotQA), while ensuring both efficiency and reliability in tool usage.


<p align="center">
<img width="100%" alt="image" src="https://github.com/user-attachments/assets/5f60be15-6992-413e-a405-a72e3b352fc1" />
</p>




## 🔧 Installation

###  Environment Setup
```bash
# Create conda environment
conda create -n webthinker python=3.9
conda activate webthinker

# Install requirements
cd WebThinker-main
pip install -r requirements.txt
```

## 🏃 Quick Start

**1. Environment Setup**
```bash
# Create conda environment
conda create -n tool_star python=3.9
conda activate tool_star

# Install requirements
cd tool_star
pip install -r requirements.txt
```

**2. Qwen2.5-72B-Instruct deployment**

In this step, we will deploy a Qwen2.5-72B-Instruct. This model is used to perform functions such as code debugging, refinement, and evaluate the accuracy of the generated answers in subsequent steps.

```bash
cd evaluation
bash vllm_server.sh
```


**3. Retriever Serving deployment**

In this section, we will deploy the retriever for performing search tasks on Wikipedia-based datasets. we provide Wikipedia retriever service implemented using FlashRAG and FastAPI. Before starting the retriever serving, you need download the [pre-indexed wikipedia](https://github.com/RUC-NLPIR/FlashRAG?tab=readme-ov-file#index), [wikipedia corpus and corresponding retriever models.](https://github.com/RUC-NLPIR/FlashRAG/blob/main/docs/original_docs/reproduce_experiment.md#preliminary) More details can be found in the documentation of FlashRAG.

For starting the retriever serving, you need to first fill the `scripts/serving/retriever_config.yaml` with the correct path to the retrieval model, index, and corpus, and available GPU ids. Then, you can run the following command to start the retriever serving:

```bash
cd evaluation/search
python host_wiki.py \
    --config serving_config.yaml \
    --num_retriever {num_retriever} \  
    --port {port}
```

**4. Inference Your Model**

In this section, we infer answers using a trained model. We support five types of mathematical reasoning datasets: aime24, aime25, gsm8k, math, and math500, as well as seven QA reasoning datasets: WebWalker, HotpotQA, 2WikiMultiHopQA, Bamboogle, MuSiQue, GAIA, and HLE.

First, you need to replace the API_URL and API key with your own in the following files:
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
Change the URL to the API_URL of your deployed retriever.

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
Replace `bing_subscription_key`, `bing_endpoint`, and `api_base_url` with your own values.

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
Replace `api_key` with the API key of your deployed model.

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
Replace `API_BASE_URL` with the API base URL of your deployed model.

Then, start the inference:
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
    --max_tokens 16384 \
    --max_input_len 16384 \
    --output_path /path/to/your_results/your_exp_math_result.json \
    --counts 500 \
    --batch_size 100 \
    --use_debug 
```
Parameter explanations:
- `--model_path`: Your model path.
- `--dataset_name`: Your dataset name, we support five types of mathematical reasoning datasets: AIME24, AIME25, GSM8K, MATH, and MATH500, as well as seven QA reasoning datasets: WebWalker, HotpotQA, 2WikiMultiHopQA, Bamboogle, MuSiQue, GAIA, and HLE.
- `--task`: Task type, for mathematical reasoning datasets, set it to `math`, and for QA reasoning datasets, set it to `qa`.
- `--gpu_use`: GPU memory utilization.
- `--max_tokens`: The maximum number of tokens the model can generate.
- `--max_input_len`: The maximum input tokens the model can accept.
- `--output_path`: Path to save the results.
- `--counts`: Number of samples to take from the test set during testing.
- `--batch_size`: Batch size for parallel inference.
- `--use_debug`: Use the debug mechanism.
Additionally, we have set other parameters for calling more tools:
- `--use_rollback`: Whether to use the rollback mechanism.
- `--use_refiner`: Whether to use the refine mechanism.

**5. Calculate Metrics**

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

Then, run the following code:
```bash
cd evaluation
python evaluate/scripts/evaluate.py \
    --output_path /path/to/your_results/your_exp_math_result.json \
    --task math \
    --dataset_name math \
    --use_llm \
    --extract_answer
```
Parameter explanations:
- `--output_path`: Path to save the results.
- `--task`: Task type, for mathematical reasoning datasets, set it to `math`, and for QA reasoning datasets, set it to `qa`.
- `--dataset_name`: Dataset name.
- `--use_llm`: Whether to use the LLM-as-judge mechanism.
- `--extract_answer`: Whether to use exact matching (removes \text and other redundant symbols).






## 📄 Citation

If you find this work helpful, please cite our paper:
```bibtex
@article{Li2025WebThinker,
  author       = {Xiaoxi Li and
                  Jiajie Jin and
                  Guanting Dong and
                  Hongjin Qian and
                  Yutao Zhu and
                  Yongkang Wu and
                  Ji{-}Rong Wen and
                  Zhicheng Dou},
  title        = {WebThinker: Empowering Large Reasoning Models with Deep Research Capability},
  journal      = {CoRR},
  volume       = {abs/2504.21776},
  year         = {2025},
  url          = {https://arxiv.org/abs/2504.21776},
  doi          = {10.48550/ARXIV.2504.21776},
  eprinttype    = {arXiv},
  eprint       = {2504.21776}
}
```






## 📄 License

This project is released under the [MIT License](LICENSE).

## 📞 Contact

For any questions or feedback, please reach out to us at [xiaoxi_li@ruc.edu.cn](xiaoxi_li@ruc.edu.cn).
