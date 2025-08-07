# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from collections import defaultdict
import torch
import time
from verl import DataProto
from verl.utils.reward_score import _default_compute_score
from mmagent.retrieve import verify_qa
import json

import openai
from mmagent.utils.chat_api import generate_messages
from mmagent.prompts import prompt_agent_verify_answer_referencing
config = json.load(open("/opt/tiger/open_verl/api_config.json"))

def get_response(model, client, messages, timeout=30):
    response = client.chat.completions.create(
        model=model, messages=messages, temperature=0, timeout=timeout, max_tokens=2048
    )
    return response.choices[0].message.content, response.usage.total_tokens

def get_response_with_retry(model, client, messages, timeout=30):
    for i in range(5):
        try:
            return get_response(model, client, messages, timeout)
        except Exception as e:
            time.sleep(20)
            print(f"Retry {i} times, exception: {e} from message {messages}")
            continue
    raise Exception(f"Failed to get response after 5 retries")

def eval_answer(question, predict, ground_truth, model):
    if model is None:
        model = "gpt-4o-2024-11-20"
    client = openai.AzureOpenAI(
        azure_endpoint=config[model]["azure_endpoint"],
        api_version=config[model]["api_version"],
        api_key=config[model]["api_key"],
    )
    if predict == "":
        return 0
    try:
        input = [
            {
                "type": "text",
                "content": prompt_agent_verify_answer_referencing.format(
                    question=question,
                    ground_truth_answer=ground_truth,
                    agent_answer=predict,
                ),
            }   
        ]
        messages = generate_messages(input)
        response = get_response_with_retry(model, client, messages)
        result = response[0].lower()
    except Exception as e:
        print(f"Error verifying qa: {question} | {str(e)}")
        return 0
    return 1 if "yes" in result else 0


class MultiTurnRewardManager:
    """The reward manager."""

    def __init__(self, tokenizer, num_examine, compute_score=None, reward_fn_key="data_source") -> None:
        self.tokenizer = tokenizer
        self.num_examine = num_examine  # the number of batches of decoded responses to print to the console
        self.compute_score = compute_score or _default_compute_score
        self.reward_fn_key = reward_fn_key

    def __call__(self, data: DataProto, return_dict=False, model=None):
        """We will expand this function gradually based on the available datasets"""

        reward_tensor = torch.zeros_like(data.batch["attention_masks"], dtype=torch.float32)
        scores = []
        for i in range(len(data)):
            data_item = data[i]
            valid_response_length = data_item.batch["attention_masks"].sum()
            if "seq_final_reward" in data_item.non_tensor_batch:
                reward = data_item.non_tensor_batch["seq_final_reward"]
            else:
                if data_item.non_tensor_batch["responses"] != "":
                    if data_item.non_tensor_batch["type"] == "web":
                        reward = eval_answer(data_item.non_tensor_batch["question"], data_item.non_tensor_batch["responses"], data_item.non_tensor_batch["answer"], model)
                    else:
                        reward = 1 if data_item.non_tensor_batch["responses"].strip() == data_item.non_tensor_batch["answer"].strip() else 0
                else:
                    reward = 0
                scores.append(reward)
                reward += data_item.non_tensor_batch["bonus"]
            reward_tensor[i, valid_response_length - 1] = reward

        if return_dict:
            return {"reward_tensor": reward_tensor, "correct": scores}
        else:
            return reward_tensor