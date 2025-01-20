from openai import OpenAI
from concurrent.futures import ThreadPoolExecutor, as_completed
import time

openai_api_key = "EMPTY"
openai_api_base = "http://103.20.97.113:8000/v1"

client = OpenAI(
    api_key=openai_api_key,
    base_url=openai_api_base,
)

n_request = 1
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