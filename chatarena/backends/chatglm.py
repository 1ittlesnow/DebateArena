from typing import List
import os
import re
import logging
from tenacity import retry, stop_after_attempt, wait_random_exponential

from .base import IntelligenceBackend
from ..message import Message

# import anthropic

from io import BytesIO
import json
import requests


# Default config follows the OpenAI playground
DEFAULT_TEMPERATURE = 0.7
DEFAULT_MAX_TOKENS = 256
DEFAULT_MODEL = "chatglm"
SYSTEM = "System"
STOP = ("<EOS>", "[EOS]", "(EOS)")  # End of sentence token
HUMAN_PROMPT = '\n\n问:'

AI_PROMPT = '\n\n答:'

CHATGLM_URL = 'http://127.0.0.1:5000/api/chatglm'
# CHATGLM_URL = "http://localhost:21002"
class ChatGLM(IntelligenceBackend):
    """
    Interface to the ChatGPT style model with system, user, assistant roles separation
    """
    stateful = False
    type_name = "openai-chat"

    def __init__(self, temperature: float = DEFAULT_TEMPERATURE, max_tokens: int = DEFAULT_MAX_TOKENS,
                 model: str = DEFAULT_MODEL,url=CHATGLM_URL, **kwargs):
        # assert is_openai_available, "openai package is not installed or the API key is not set"
        super().__init__(temperature=temperature, max_tokens=max_tokens, model=model, **kwargs)
        ## 不重要，统统不重要，也没用到
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.model = model
        self.url = url
        
    # 原版
    @retry(stop=stop_after_attempt(6), wait=wait_random_exponential(min=1, max=60))
    def _get_response(self, messages):
        """
        messages is a string
        return a string
        
        data is a dict like {'prompt': prompt}
        response is a dict like {'result': response}
        """
        data = {'prompt': messages}
        response = requests.post(self.url, json=data).json()
        return response['result'].strip()

    def keep_prompt_length(self, input_prompt, global_prompt, role_desc):
        input_prompt_length = len(input_prompt)
        global_prompt_length = len(global_prompt)
        role_desc_prompt_length = len(role_desc)
        
        if input_prompt_length>2000:
            input_prompt = input_prompt[input_prompt_length-1000+global_prompt_length+role_desc_prompt_length:]
            input_prompt = global_prompt+'\n\n'+role_desc+input_prompt
            return input_prompt
        else:
            return input_prompt

    def query(self, agent_name: str, role_desc: str, history_messages: List[Message], global_prompt: str = None,
              request_msg: Message = None,*args, **kwargs) -> str:
        """
        format the input and call the Claude API
        args:
            agent_name: the name of the agent
            role_desc: the description of the role of the agent
            env_desc: the description of the environment
            history_messages: the history of the conversation, or the observation for the agent
            request_msg: the request for the chatGPT
        """
        all_messages = [(SYSTEM, global_prompt), (SYSTEM, role_desc)] if global_prompt else [(SYSTEM, role_desc)]

        for message in history_messages:
            all_messages.append((message.agent_name, message.content))
        if request_msg:
            all_messages.append((request_msg.agent_name, request_msg.content))

        prompt = ""
        prev_is_human = False  # Whether the previous message is from human (in anthropic, the human is the user)
        for i, message in enumerate(all_messages):
            if i == 0:
                assert message[0] == SYSTEM  # The first message should be from the system

            if message[0] == agent_name:
                if prev_is_human:
                    prompt = f"{prompt}{AI_PROMPT} {message[1]}"
                else:
                    prompt = f"{prompt}\n\n{message[1]}"
                prev_is_human = False
            else:
                if prev_is_human:
                    prompt = f"{prompt}\n\n[{message[0]}]: {message[1]}"
                else:
                    prompt = f"{prompt}{HUMAN_PROMPT}\n[{message[0]}]: {message[1]}"
                prev_is_human = True
        assert prev_is_human  # The last message should be from the human
        # Add the AI prompt for Claude to generate the response
        
        
        prompt = f"{prompt}{AI_PROMPT}"
        prompt = self.keep_prompt_length(prompt, global_prompt, role_desc)
        response = self._get_response(prompt, *args, **kwargs)

        # Remove the agent name if the response starts with it
        response = re.sub(rf"^\s*\[{agent_name}]:?", "", response).strip()
        
        return response
    
    # 改为流式输出
    @retry(stop=stop_after_attempt(6), wait=wait_random_exponential(min=1, max=60))
    def _get_response_gen(self, messages, temperature, top_p, max_new_tokens):
        """
        messages is a string
        return a string
        
        data is a dict like {'prompt': prompt}
        response is a dict like {'result': response}
        """
        print("=========MODEL:chatglm=========")
        temperature = float(temperature)
        top_p = float(top_p)
        max_new_tokens = int(max_new_tokens)
        controller_addr = "http://localhost:21001"
        # model_name = "chatglm-6b"
        model_name = "chatglm2-6b"
        ret = requests.post(
            controller_addr + "/get_worker_address", json={"model": model_name}
        )
        worker_addr = ret.json()["address"]
        if worker_addr == "":
            print(f"No available workers for {model_name}")
            return
        headers = {"User-Agent": "FastChat Client"}
        gen_params = {
            "model": model_name,
            "prompt": messages,
            "temperature": temperature,
            "max_new_tokens": max_new_tokens,
            "top_p":top_p,
            "stop": '###',
            "stop_token_ids": None,
            "echo": False,
        }
        print(f"PARAMS：\n{gen_params}")
        response = requests.post(
            worker_addr + "/worker_generate_stream",
            headers=headers,
            json=gen_params,
            stream=True,
        )
        # prev = 0
        for chunk in response.iter_lines(decode_unicode=False, delimiter=b"\0"):
            if chunk:
                data = json.loads(chunk.decode())
                output = data['text'].strip()
                yield output
                # print(output[prev:], end="", flush=True)
                # prev = len(output)
        print("")
        
        
        
        # data = {'prompt': messages}
        # response = requests.post(self.url, json=data, stream=True)
        # for chunk in response.iter_lines(decode_unicode=False, delimiter=b"\0"):
        #     if chunk:
        #         res_json = json.loads(chunk.decode())
        #         yield res_json
    
    
    
    
    
    def query_with_conclusion(self, agent_name: str, role_desc: str, history_messages: List[Message], global_prompt: str = None,
              request_msg: Message = None, is_end=False,*args, **kwargs) -> str:
        """
        format the input and call the Claude API
        args:
            agent_name: the name of the agent
            role_desc: the description of the role of the agent
            env_desc: the description of the environment
            history_messages: the history of the conversation, or the observation for the agent
            request_msg: the request for the chatGPT
        """
        all_messages = [(SYSTEM, global_prompt), (SYSTEM, role_desc)] if global_prompt else [(SYSTEM, role_desc)]

        for message in history_messages:
            all_messages.append((message.agent_name, message.content))
        if request_msg:
            all_messages.append((request_msg.agent_name, request_msg.content))

        prompt = ""
        prev_is_human = False  # Whether the previous message is from human (in anthropic, the human is the user)
        for i, message in enumerate(all_messages):
            if i == 0:
                assert message[0] == SYSTEM  # The first message should be from the system

            if message[0] == agent_name:
                if prev_is_human:
                    prompt = f"{prompt}{AI_PROMPT} {message[1]}"
                else:
                    prompt = f"{prompt}\n\n{message[1]}"
                prev_is_human = False
            else:
                if prev_is_human:
                    prompt = f"{prompt}\n\n[{message[0]}]: {message[1]}"
                else:
                    prompt = f"{prompt}{HUMAN_PROMPT}\n[{message[0]}]: {message[1]}"
                prev_is_human = True
        assert prev_is_human  # The last message should be from the human
        # Add the AI prompt for Claude to generate the response
        if is_end:
            prompt += f"{prompt}\n{SYSTEM} 请总结你的辩论陈词。"
        
        prompt = f"{prompt}{AI_PROMPT}"
        prompt = self.keep_prompt_length(prompt, global_prompt, role_desc)
        response = self._get_response(prompt, *args, **kwargs)

        ###
        # response_gen = self._get_response_gen(prompt, *args, **kwargs)
        # print(next(response_gen))
        ###

        # Remove the agent name if the response starts with it
        response = re.sub(rf"^\s*\[{agent_name}]:?", "", response).strip()
        
        return response

    def query_with_conclusion_gen(self, agent_name: str, role_desc: str, history_messages: List[Message], global_prompt: str = None,
              request_msg: Message = None, is_end=False, temperature=0.7, top_p=1, max_new_tokens=512, *args, **kwargs) -> str:
        """
        format the input and call the Claude API
        args:
            agent_name: the name of the agent
            role_desc: the description of the role of the agent
            env_desc: the description of the environment
            history_messages: the history of the conversation, or the observation for the agent
            request_msg: the request for the chatGPT
        """
        all_messages = [(SYSTEM, global_prompt+'\n'+role_desc)] if global_prompt else [(SYSTEM, role_desc)]

        for message in history_messages:
            all_messages.append((message.agent_name, message.content))
        if request_msg:
            all_messages.append((request_msg.agent_name, request_msg.content))

        prompt = ""
        prev_is_human = False  # Whether the previous message is from human (in anthropic, the human is the user)
        for i, message in enumerate(all_messages):
            if i == 0:
                assert message[0] == SYSTEM  # The first message should be from the system

            if message[0] == agent_name:
                if prev_is_human:
                    prompt = f"{prompt}{AI_PROMPT} {message[1]}"
                else:
                    prompt = f"{prompt}\n\n{message[1]}"
                prev_is_human = False
            else:
                if prev_is_human:
                    prompt = f"{prompt}\n\n[{message[0]}]: {message[1]}"
                else:
                    prompt = f"{prompt}{HUMAN_PROMPT}\n[{message[0]}]: {message[1]}"
                prev_is_human = True
        assert prev_is_human  # The last message should be from the human
        # Add the AI prompt for Claude to generate the response
        if is_end:
            prompt = f"{prompt}\n 本次回答为最后一轮，请总结你的辩论陈词。"
        
        prompt = f"{prompt}{AI_PROMPT}"
        prompt = self.keep_prompt_length(prompt, global_prompt, role_desc)

        response_gen = self._get_response_gen(prompt, temperature=temperature, top_p=top_p, max_new_tokens=max_new_tokens, *args, **kwargs)

        return response_gen
