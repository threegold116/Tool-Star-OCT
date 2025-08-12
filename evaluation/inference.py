from transformers import AutoTokenizer,AutoModelForCausalLM
import os
os.environ["CUDA_VISIBLE_DEVICES"]="3"
model_path = "/home/sxjiang/model/Tool-Star-Qwen-3B"
tokenizer = AutoTokenizer.from_pretrained(model_path,trust_remote_code=True,device_map="auto")
model = AutoModelForCausalLM.from_pretrained(model_path,trust_remote_code=True,device_map="auto")


def inference(question):
    input_messages = [
        {
            "role": "system",
            "content": "You are a helpful assistant that can solve the given question step by step with the help of the wikipedia search tool and python interpreter tool. \
Given a question, you need to first think about the reasoning process in the mind and then provide the answer. \
For example, <think> This is the reasoning process. </think> <answer> The final answer is \\[ \\boxed{answer here} \\] </answer>. \
In the last part of the answer, the final exact answer is enclosed within \\boxed{} with latex format."
        },
        {
            "role": "user",
            "content": question
        }
    ]
    inputs = tokenizer.apply_chat_template(input_messages, tokenize=False, add_generation_prompt=True, add_model_prefix=True)
    print(inputs)
    inputs = tokenizer(inputs,return_tensors="pt")
    inputs = inputs.to(model.device)
    print(inputs)
    outputs = model.generate(**inputs,max_new_tokens=4096)
    print(outputs)
    return tokenizer.decode(outputs[0],skip_special_tokens=False)


if __name__ == "__main__":
    question = "What is the capital of France?"
    print(inference(question))

