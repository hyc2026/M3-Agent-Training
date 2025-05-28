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

def eval_answer(question, predict, ground_truth, model):
    if predict == "":
        return False
    if model is None:
        response = verify_qa(question, ground_truth, predict)
    else:
        response = verify_qa(question, ground_truth, predict, model=model)
    if response is None:
        return False
    response = response.lower()
    return 1 if "yes" in response else 0

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
            if data_item.non_tensor_batch["responses"] != "":
                reward = eval_answer(data_item.non_tensor_batch["question"], data_item.non_tensor_batch["responses"], data_item.non_tensor_batch["answer"], model)
                # time.sleep(0.4) # control the qps of the GPT-4o
            else:
                reward = 0
            scores.append(reward)
            reward += data_item.non_tensor_batch["bonus"]
            # related_id_scores = json.loads(data_item.non_tensor_batch["related_id_scores"])
            # similarity_score = 0
            # if len(related_id_scores) > 0:
            #     for j in related_id_scores:
            #         if j > 0.4:
            #             similarity_score += j
            #     reward += similarity_score / len(related_id_scores)
            reward_tensor[i, valid_response_length - 1] = reward

        if return_dict:
            return {"reward_tensor": reward_tensor, "correct": scores}
        else:
            return reward_tensor