# Super efficient LLM serving backend x25 speed

# A. Concept

![IMG_0778.jpeg](Super%20efficient%20LLM%20serving%20backend%20x25%20speed%200bd7ff8f8f0e42a6a80177712d75da8a/IMG_0778.jpeg)

1. **Model to day**: https://huggingface.co/Qwen/Qwen2-7B-Instruct
2. **TooL: vLLM (**[https://github.com/vllm-project/vllm?tab=readme-ov-file](https://github.com/vllm-project/vllm?tab=readme-ov-file))
- As models like GPT-3 and LLaMA **grow in size**, the computational demands for inference — producing responses from these models — can **slow down applications and increase operational costs**.
- **vLLM** addresses these issues head-on by providing a highly optimized solution for **faster and more cost-efficient inference and serving**
- Standout:
    - **Maximizing throughput while minimizing memory overhead.**
        - Hugging Face Transformers face significant challenges when serving large models, especially due to inefficient memory management.
        - The core of vLLM’s innovation lies in its **PagedAttention** mechanism, which efficiently handles the memory challenges associated with LLM serving:
            - In the context of LLMs, every input token generates attention keys and values (KV), which need to be cached in memory for future use during inference → takes up substantial memory
            - PagedAttention solves this problem by **partitioning the KV cache into smaller, non-contiguous blocks→ store these blocks efficiently and retrieve them as needed, reducing memory waste to less than 4%.**
            - Ability to serve more requests in parallel, which is particularly useful for large-scale deployments
                - In real-world benchmarks, vLLM outperforms Hugging Face Transformers by **14x to 24x** in terms of throughput.
                - Even when compared to Hugging Face’s Text Generation Inference (TGI), which was previously considered the gold standard for inference speed, vLLM delivers **3.5x faster throughput** in some tests.
                - With vLLM, the same infrastructure can handle **5x more traffic** without needing additional GPUs
    - **Efficient Memory Sharing for Parallel Sampling:**
        - Another key feature of vLLM is its ability to handle **parallel sampling** efficiently
        - However, vLLM’s PagedAttention enables **memory sharing** across multiple outputs, reducing the memory overhead of parallel sampling by up to **55%**.
1. **Ollama vs vLLM**

![image.png](Super%20efficient%20LLM%20serving%20backend%20x25%20speed%200bd7ff8f8f0e42a6a80177712d75da8a/image.png)

|  | Ollama | vLLM |
| --- | --- | --- |
| **Purpose** | A tool that makes it easy to use LLMs on your own computer. | A tool designed to run LLMs very efficiently, especially when serving many users at once. |
| **Handling Multiple Requests (Concurrency)** | It can handle multiple requests, but it slows down as more requests come in. | It handles multiple requests like a champ, staying speedy even with lots of requests. |
| **Speed** | With 16 requests at once, it took about 17 seconds per request. | With 16 requests at once, it only took about 9 seconds per request. |
| **Output (Tokens Generated)** |  | At 16 concurrent requests, VLLM produced twice as many tokens (words) per second as Ollama. |
| **Pushing the Limits** | It struggled with 32 requests at once, showing it has a lower limit. | It handled 32 requests smoothly, producing 1000 tokens per second. |

**Summary**:

- While Ollama is user-friendly and great for personal use, VLLM shines when you need to handle many requests efficiently. VLLM is like a sports car — it performs better under pressure and can handle more “traffic” (requests) without slowing down
- We tested both tools using the same AI model (Llama2 8B) and compared how they performed.
- Source: https://medium.com/@naman1011/ollama-vs-vllm-which-tool-handles-ai-models-better-a93345b911e6

**Extra**: Video use Ollama from Mì AI:

[https://www.youtube.com/watch?v=mvyRYDYjHZs](https://www.youtube.com/watch?v=mvyRYDYjHZs)

# **B. Handson**

1. **GPU (prebuilt)**

| Install venv | sudo apt update
sudo apt install -y python3-venv |  |
| --- | --- | --- |
| Create vllm_gpu vent | python3 -m venv vllm
source vllm/bin/activate |  |
| Prebuilt, install via pip | pip install vllm |  |
| Serve | vllm serve Qwen/Qwen2-7B-Instruct |  |
1. **CPU (built from source - not recommend - ollama instead)**

```python
,"—model","Qwen/Qwen2-7B-Instruct"]
```

| Change dockerfile.cpu | Add above code in bottom line |  |
| --- | --- | --- |
| Build | docker build -f Dockerfile.cpu -t vllm-cpu-env --shm-size=4g . |  |
| Run | docker run -it \
--rm \
--network=host \
vllm-cpu-env |  |
1. F**lask (old way)**
- Create env:

```python
python3 -m venv flask_env
source flask_env/bin/activate
```

- Setup.txt

```python
transformers
flask
torch
bitsandbytes
accelerate>=0.26.0
```

- Serving code:

```python
from transformers import BitsAndBytesConfig
from transformers import AutoModelForCausalLM, AutoTokenizer
from flask import Flask, jsonify

# Config
device = "cuda"  # the device to load the model onto
model_name = "Qwen/Qwen2-7B-Instruct"
quantization_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype="float16",
)

# Load LLM
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    quantization_config=quantization_config,
    torch_dtype="auto",
    device_map=device
)
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Infer function
def infer_llm(prompt):
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": prompt}
    ]
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )
    model_inputs = tokenizer([text], return_tensors="pt").to(device)

    generated_ids = model.generate(
        model_inputs.input_ids,
        max_new_tokens=512
    )
    generated_ids = [
        output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
    ]

    response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
    return response

# Setup flask
app = Flask(__name__)

@app.route('/v1/chat/completions', methods=['POST'])
def index():
    prompt = "Tell me about AI in today economic"
    response = infer_llm(prompt)
    return jsonify({"result": response})

if __name__ == '__main__':
    app.run(debug=True, host="0.0.0.0", port=8000, use_reloader=False)

```

- Load test code

```python
from openai import OpenAI
from concurrent.futures import ThreadPoolExecutor, as_completed
import time

openai_api_key = "EMPTY"
openai_api_base = "http://103.20.97.111:8000/v1"

client = OpenAI(
    api_key=openai_api_key,
    base_url=openai_api_base,
)

n_request = 10
n_thread = 100
users = ['Thang'] * n_request

def make_request(i_user):
    chat_response = client.chat.completions.create(
        model="Qwen/Qwen2-7B-Instruct",
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "Tell me a joke for user " + i_user},
        ]
    )
    return chat_response

start = time.time()
processes = []
with ThreadPoolExecutor(max_workers=n_thread) as executor:
    for user in users:
        processes.append(executor.submit(make_request, user))

for task in as_completed(processes):
    print(str(task.result())[-10:])

print("Total time for {} request with concurrent {} threads".format(n_request, n_thread))
print(time.time() - start, 's')
```

1. Result

| GPU | Total time for 100 request with concurrent 100 threads
2.6431617736816406 s
 |
| --- | --- |
| Flask | Total time for 1 request with concurrent 100 threads
25.873571157455444 s
 |
| CPU | Below is CPU%
Too slow, can’t wait |

![image.png](Super%20efficient%20LLM%20serving%20backend%20x25%20speed%200bd7ff8f8f0e42a6a80177712d75da8a/image%201.png)