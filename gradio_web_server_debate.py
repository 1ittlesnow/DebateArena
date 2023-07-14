import gradio as gr
import json
# from fastchat.serve.gradio_patch import Chatbot
from gradio import Chatbot
# from fastchat.serve.gradio_web_server import (
#     block_css,
#     get_model_list,
#     disable_btn,
#     enable_btn,
# )
import requests
import argparse
import logging
import os
os.environ['OPENAI_API_KEY'] = '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'
from chatarena.agent import Player
from chatarena.backends import OpenAIChat,ChatGLM,Belle,Lizhi,Moss,OpenAIChatSb
from chatarena.environments.conversation import Conversation
from chatarena.message import Message
from chatarena.arena import Arena
from tqdm import tqdm
import datetime
import time
import random
from fastchat.utils import (
    parse_gradio_auth_creds,
)
from bs4 import BeautifulSoup
import openai

enable_btn = gr.Button.update(interactive=True)
disable_btn = gr.Button.update(interactive=False)

code_highlight_css = """
#chatbot .hll { background-color: #ffffcc }
#chatbot .c { color: #408080; font-style: italic }
#chatbot .err { border: 1px solid #FF0000 }
#chatbot .k { color: #008000; font-weight: bold }
#chatbot .o { color: #666666 }
#chatbot .ch { color: #408080; font-style: italic }
#chatbot .cm { color: #408080; font-style: italic }
#chatbot .cp { color: #BC7A00 }
#chatbot .cpf { color: #408080; font-style: italic }
#chatbot .c1 { color: #408080; font-style: italic }
#chatbot .cs { color: #408080; font-style: italic }
#chatbot .gd { color: #A00000 }
#chatbot .ge { font-style: italic }
#chatbot .gr { color: #FF0000 }
#chatbot .gh { color: #000080; font-weight: bold }
#chatbot .gi { color: #00A000 }
#chatbot .go { color: #888888 }
#chatbot .gp { color: #000080; font-weight: bold }
#chatbot .gs { font-weight: bold }
#chatbot .gu { color: #800080; font-weight: bold }
#chatbot .gt { color: #0044DD }
#chatbot .kc { color: #008000; font-weight: bold }
#chatbot .kd { color: #008000; font-weight: bold }
#chatbot .kn { color: #008000; font-weight: bold }
#chatbot .kp { color: #008000 }
#chatbot .kr { color: #008000; font-weight: bold }
#chatbot .kt { color: #B00040 }
#chatbot .m { color: #666666 }
#chatbot .s { color: #BA2121 }
#chatbot .na { color: #7D9029 }
#chatbot .nb { color: #008000 }
#chatbot .nc { color: #0000FF; font-weight: bold }
#chatbot .no { color: #880000 }
#chatbot .nd { color: #AA22FF }
#chatbot .ni { color: #999999; font-weight: bold }
#chatbot .ne { color: #D2413A; font-weight: bold }
#chatbot .nf { color: #0000FF }
#chatbot .nl { color: #A0A000 }
#chatbot .nn { color: #0000FF; font-weight: bold }
#chatbot .nt { color: #008000; font-weight: bold }
#chatbot .nv { color: #19177C }
#chatbot .ow { color: #AA22FF; font-weight: bold }
#chatbot .w { color: #bbbbbb }
#chatbot .mb { color: #666666 }
#chatbot .mf { color: #666666 }
#chatbot .mh { color: #666666 }
#chatbot .mi { color: #666666 }
#chatbot .mo { color: #666666 }
#chatbot .sa { color: #BA2121 }
#chatbot .sb { color: #BA2121 }
#chatbot .sc { color: #BA2121 }
#chatbot .dl { color: #BA2121 }
#chatbot .sd { color: #BA2121; font-style: italic }
#chatbot .s2 { color: #BA2121 }
#chatbot .se { color: #BB6622; font-weight: bold }
#chatbot .sh { color: #BA2121 }
#chatbot .si { color: #BB6688; font-weight: bold }
#chatbot .sx { color: #008000 }
#chatbot .sr { color: #BB6688 }
#chatbot .s1 { color: #BA2121 }
#chatbot .ss { color: #19177C }
#chatbot .bp { color: #008000 }
#chatbot .fm { color: #0000FF }
#chatbot .vc { color: #19177C }
#chatbot .vg { color: #19177C }
#chatbot .vi { color: #19177C }
#chatbot .vm { color: #19177C }
#chatbot .il { color: #666666 }
"""
# .highlight  { background: #f8f8f8; }

table_css = """
table {
    line-height: 0em
}
"""


block_css = (
    code_highlight_css
    + """
pre {
    white-space: pre-wrap;       /* Since CSS 2.1 */
    white-space: -moz-pre-wrap;  /* Mozilla, since 1999 */
    white-space: -pre-wrap;      /* Opera 4-6 */
    white-space: -o-pre-wrap;    /* Opera 7 */
    word-wrap: break-word;       /* Internet Explorer 5.5+ */
}
#notice_markdown th {
    display: none;
}
#notice_markdown td {
    padding-top: 8px;
    padding-bottom: 8px;
}
#leaderboard_markdown td {
    padding-top: 8px;
    padding-bottom: 8px;
}
[data-testid = "bot"] {
    max-width: 75%;
    width: auto !important;
    border-bottom-left-radius: 0 !important;
}
[data-testid = "user"] {
    max-width: 75%;
    width: auto !important;
    border-bottom-right-radius: 0 !important;
}
"""
)

def get_model_list(controller_url, add_chatgpt, add_claude, add_palm):
    ret = requests.post(controller_url + "/refresh_all_workers")
    assert ret.status_code == 200
    ret = requests.post(controller_url + "/list_models")
    models = ret.json()["models"]

    # Add API providers
    if add_chatgpt:
        models += ["gpt-3.5-turbo"]
        # models += ["gpt-3.5-turbo", "gpt-4"]
    if add_claude:
        models += ["claude-v1", "claude-instant-v1"]
    if add_palm:
        models += ["palm-2"]
    return models

def read_json(input_path):
    with open(input_path, 'r', encoding='utf-8') as f:
        input_data = json.load(f)
    return input_data

debate_path = '/raid/yjd/arena/data/debate.json'
debate_list = read_json(debate_path)
theme_list = [i['theme'] for i in debate_list]
positive_theme_list = [i['positive'] for i in debate_list]
negative_theme_list = [i['negative'] for i in debate_list]
theme2num = {j:i for i,j in enumerate(theme_list)}
num2theme = {j:i for i,j in theme2num.items()}

def get_conv_log_filename():
    LOGDIR = '/raid/yjd/arena/logs'
    t = datetime.datetime.now()
    name = os.path.join(LOGDIR, f"{t.year}-{t.month:02d}-{t.day:02d}-conv.json")
    return name

def debate_test(text):
    print(text)
    return [["hello","111"]]

## theme   left_theme  right_theme
def save_debate(chatbot, vote_type, model_selector0, model_selector1, theme, left_theme, right_theme):
    print("save debate")
    # for pair in chatbot:
    #     pair[0] = pair[0].encode('ascii').decode('unicode_escape')
    #     pair[1] = pair[1].encode('ascii').decode('unicode_escape')
    with open(get_conv_log_filename(), "a") as fout:
        data = {
            "WARNING":"right_model is first",
            "tstamp": round(time.time(), 4),
            "theme":theme,
            "left_theme":left_theme,
            "right_theme":right_theme,
            "vote_type": vote_type,
            "left_model":model_selector0,
            "right_model":model_selector1,
            "history": chatbot,
        }
        fout.write(json.dumps(data, ensure_ascii=False) + "\n")
    
def left_vote(chatbot, model_selector0, model_selector1, theme, left_theme, right_theme):
    print("left")
    save_debate(chatbot, "left_vote", model_selector0, model_selector1, theme, left_theme, right_theme)
    return [disable_btn] * 5

def right_vote(chatbot, model_selector0, model_selector1, theme, left_theme, right_theme):
    print("right")
    save_debate(chatbot, "right_vote", model_selector0, model_selector1, theme, left_theme, right_theme)
    return [disable_btn] * 5

def tie_vote(chatbot, model_selector0, model_selector1, theme, left_theme, right_theme):
    print("tie")
    save_debate(chatbot, "tie_vote", model_selector0, model_selector1, theme, left_theme, right_theme)
    return [disable_btn] * 5

def bothbad_vote(chatbot, model_selector0, model_selector1, theme, left_theme, right_theme):
    print("bothbad")
    save_debate(chatbot, "both_bad_vote", model_selector0, model_selector1, theme, left_theme, right_theme)
    return [disable_btn] * 5

def clear_history():
    print("clear_history")
    return [None, theme_list[0]] + [disable_btn]*6 + [0] + [gr.Button.update(interactive=True, value="开始"), disable_btn, gr.Textbox(value="")]

def share_click():
    print("share")
    pass

def flash_buttons(loop_num):
    if loop_num==6:
        btn_updates = [
            [disable_btn] * 4 + [enable_btn] * 2,
            [enable_btn] * 6,
        ]
        for i in range(10):
            yield btn_updates[i % 2]
            time.sleep(0.2)
    elif loop_num==2 or loop_num==4:
        btn_updates = [
                [disable_btn] * 4 + [enable_btn] * 2,
                [enable_btn] * 6,
        ]
        for i in range(6):
            yield btn_updates[i % 2]
            time.sleep(0.2)
    else:
        yield [disable_btn] * 4 + [enable_btn] * 2

def exchange_theme(standA_text, standB_text):
    print("exchange_theme")
    return standB_text, standA_text

def theme_selector_change(selected_value):
    print("theme_selector_change RUN!")
    if selected_value in theme2num:
        idx = theme2num[selected_value]
        return debate_list[idx]['positive'], debate_list[idx]['negative']
    else:
        return "",""

def disable_theme_models():
    return [gr.update(interactive=False)]*6

def enable_theme_models():
    return [gr.update(interactive=True)]*6

def arena_to_chatbot(arena):
    messages = arena.environment.get_observation()
    # message_row = {
    #                 "agent_name": message.agent_name,
    #                 "content": message.content,
    #                 "turn": message.turn,
    #                 "timestamp": str(message.timestamp),
    #                 "visible_to": message.visible_to,
    #                 "msg_type": message.msg_type,
    #             }
    chats = []
    len_messages = len(messages)
    for i in range(int(len_messages/2)):
        chats.append([messages[i].content.replace(':','：'), messages[i+1].content.replace(':','：')])
    if len_messages%2!=0:
        chats.append([messages[-1].content.replace(':','：'), None])
    return chats

def get_player(name, backend, role_desc, global_prompt):
    if 'chatglm' in backend:
        player = Player(name=name, backend=ChatGLM(),
                            role_desc=role_desc,
                            global_prompt=global_prompt)
    elif 'turbo' in backend:
        player = Player(name=name, backend=OpenAIChat(),
                            role_desc=role_desc,
                            global_prompt=global_prompt)
    return player
    
def initialize_environment(theme, standA_text, standB_text):
    # 返回chatbot   开始按钮    loop_num
    left_theme = standA_text.strip()
    right_theme = standB_text.strip()
    
    player1_name = f"正方辩手" #right
    player2_name = f"反方辩手"
    
    if theme in theme_list and left_theme in positive_theme_list: # 如何 theme是我们自己的题目 而且左边的辩题是正辩题，则左边为正方辩手 右边为反方辩手
        player1_name = f"反方辩手" # right
        player2_name = f"正方辩手" # left
        
    environment_description = f"你现在处于一场辩论中，请你扮演辩手的角色。辩论的主题是'{theme}'。\n规则:\n反方辩手反对该主题，而正方辩手则支持该主题。\n你在与对方辩论时坚定的支持自己的观点，并且采用一切你能想到的论据用来攻击对手的论点，或者为自己的论点辩护。\n不要以任何其他角色的身份做出回应，仅以你自己的身份做出回应。\n主席不会打断。\n不要对对方辩手进行提问。\n请在三句话之内回答完毕。"
    
    if right_theme:
        player1_role_desc = f"你是{player1_name}，你的持方是{right_theme}。"   # right player
    else:
        player1_role_desc = ""
    if left_theme:
        player2_role_desc = f"你是{player2_name}，你的持方是{left_theme}。"   # left player
    else:
        player2_role_desc = ""
    
    return player1_name, player2_name, player1_role_desc, player2_role_desc, environment_description


def regenerate(model_selector0, model_selector1, theme, standA_text, standB_text, chatbot, loop_num, temperature, top_p, max_new_tokens):

    # player1是右边的辩手，是第一个发言的
    player1_name, player2_name, player1_role_desc, player2_role_desc, environment_description = initialize_environment(theme, standA_text, standB_text)
    
    ### player1是右边的  所以是model_selector1    player2是左边的所以是model_selector0
    player1 = get_player(player1_name, model_selector1, player1_role_desc, environment_description)
    player2 = get_player(player2_name, model_selector0, player2_role_desc, environment_description)
    ###
    env = Conversation(player_names=[p.name for p in [player1, player2]])
    
    arena = Arena(players=[player1, player2],
                    environment=env, global_prompt=environment_description)
    ## 因为是regenerate，所以要把chatbot里面最新的一个去掉
    if chatbot[-1][-1] is not None:
        chatbot[-1][-1] = None
    elif chatbot[-1][0] is not None:
        chatbot[-1][0] = None
    ## observation与chatbot 拼接一下 放到 env里面,  手动模拟arena已经执行过chatbot了
    for i,pair in enumerate(chatbot):
        if pair[0]:
            message0 = Message(agent_name=player1_name, content=BeautifulSoup(pair[0],'html.parser').get_text(), turn=arena.environment._current_turn)
            arena.environment.message_pool.append_message(message0)
            arena.environment._current_turn += 1
        if pair[1]:
            message1 = Message(agent_name=player2_name, content=BeautifulSoup(pair[1],'html.parser').get_text(), turn=arena.environment._current_turn)
            arena.environment.message_pool.append_message(message1)
            arena.environment._current_turn += 1
    
    ## 手动判断 现在是 哪个辩手发言
    loop_num -= 1
    if loop_num%2==0:
        player_name = player1.name
    else:
        player_name = player2.name
    
    player = arena.name_to_player[player_name]  # get the player object
    observation = arena.environment.get_observation(player_name)  # get the observation for the player

    
    
    timestep = None
    is_end = False
    if loop_num==4 or loop_num==5:
        is_end = True
    gen = player.call_with_conclusion_gen(observation, is_end=is_end, temperature=temperature, top_p=top_p, max_new_tokens=max_new_tokens)  # take an action
    
    chatbot = arena_to_chatbot(arena)
    if chatbot==[]:
        chatbot.append([None, None])
    if chatbot[-1][-1] is not None and chatbot[-1][0] is not None:
        chatbot.append([None, None])
        
    if chatbot[-1][0] is None:
        tmp_idx = 0
    elif chatbot[-1][-1] is None:
        tmp_idx = -1
    while True:
        stop = True
        try:
            action = next(gen)
            chatbot[-1][tmp_idx]=action.replace(':','：')
            stop = False
        except StopIteration:
            pass
        yield [chatbot,gr.update(value="下一步", interactive=False)] + [disable_btn]*6 + [loop_num]
        if stop:
            if loop_num==5:
                yield [chatbot,gr.update(value="下一步", interactive=False)] + [enable_btn]*4 + [enable_btn]*2 + [loop_num+1]
            else:
                yield [chatbot,gr.update(value="下一步", interactive=True)] + [disable_btn]*4 + [enable_btn]*2 + [loop_num+1]
            break
    # print(f"RESPONSE_{player_name}:\n{action}")
    if arena.environment.check_action(action, player_name):
        timestep = arena.environment.step(player_name, action)
        print()
    else:
        logging.warning(f"{player_name} made an invalid action {action}")

def next_step_arena_debate(model_selector0, model_selector1, theme, standA_text, standB_text, chatbot, loop_num, temperature, top_p, max_new_tokens):
    # player1是右边的辩手，是第一个发言的
    player1_name, player2_name, player1_role_desc, player2_role_desc, environment_description = initialize_environment(theme, standA_text, standB_text)
    
    ### player1是右边的  所以是model_selector1    player2是左边的所以是model_selector0
    player1 = get_player(player1_name, model_selector1, player1_role_desc, environment_description)
    player2 = get_player(player2_name, model_selector0, player2_role_desc, environment_description)
    ###
    env = Conversation(player_names=[p.name for p in [player1, player2]])
    
    arena = Arena(players=[player1, player2],
                    environment=env, global_prompt=environment_description)
    
    ## observation与chatbot 拼接一下 放到 env里面,  手动模拟arena已经执行过chatbot了
    for i,pair in enumerate(chatbot):
        if pair[0]:
            message0 = Message(agent_name=player1_name, content=BeautifulSoup(pair[0],'html.parser').get_text(), turn=arena.environment._current_turn)
            arena.environment.message_pool.append_message(message0)
            arena.environment._current_turn += 1
        if pair[1]:
            message1 = Message(agent_name=player2_name, content=BeautifulSoup(pair[1],'html.parser').get_text(), turn=arena.environment._current_turn)
            arena.environment.message_pool.append_message(message1)
            arena.environment._current_turn += 1
    
    ## 手动判断 现在是 哪个辩手发言
    if loop_num%2==0:
        player_name = player1.name
    else:
        player_name = player2.name
    
    player = arena.name_to_player[player_name]  # get the player object
    observation = arena.environment.get_observation(player_name)  # get the observation for the player

    
    
    timestep = None
    is_end = False
    if loop_num==4 or loop_num==5:
        is_end = True
    gen = player.call_with_conclusion_gen(observation, is_end=is_end, temperature=temperature, top_p=top_p, max_new_tokens=max_new_tokens)  # take an action
    
    chatbot = arena_to_chatbot(arena)
    if chatbot==[]:
        chatbot.append([None, None])
    if chatbot[-1][-1] is not None and chatbot[-1][0] is not None:
        chatbot.append([None, None])
        
    if chatbot[-1][0] is None:
        tmp_idx = 0
    elif chatbot[-1][-1] is None:
        tmp_idx = -1
    # loop_num   0~6
    action = ""
    while True:
        stop = True
        try:
            action = next(gen)
            chatbot[-1][tmp_idx]=action.replace(':','：')
            stop = False
        except StopIteration:
            pass
        yield [chatbot,gr.update(value="下一步", interactive=False)] + [disable_btn]*6 +[loop_num]
        if stop:
            print(f"RESPONSE:\n{action}")
            # print(f"CHATBOT:\n{chatbot}")
            if loop_num==5:
                yield [chatbot,gr.update(value="下一步", interactive=False)] + [enable_btn]*6 +[loop_num+1]
            else:
                yield [chatbot,gr.update(value="下一步", interactive=True)] + [disable_btn]*4 + [enable_btn]*2 +[loop_num+1]
            break
    # print(f"RESPONSE_{player_name}:\n{action}")
    if arena.environment.check_action(action, player_name):
        timestep = arena.environment.step(player_name, action)
        print()
    else:
        logging.warning(f"{player_name} made an invalid action {action}")
        

def ask_chatgpt(messages):
    res = openai.ChatCompletion.create(
        model="gpt-3.5-turbo", messages=messages, temperature=0.7, stream=True
    )
    text = ""
    for chunk in res:
        text += chunk["choices"][0]["delta"].get("content", "")
        yield text


def gpt4_evaluation(chatbot):
    # model = "gpt-3.5-turbo"
    # temperature = 0.7
    prompt = ""
    for pair in chatbot:
        prompt += "右方辩手：" + pair[0] + '\n\n'
        prompt += "左方辩手：" + pair[1] + '\n\n'
    messages = [
        {"role": "system", "content": "你是一位资深的辩论专家，你现在处于一场辩论中，你是本次辩论的主席。"},
        {"role": "user", "content": f"{prompt}\n\n上面为本次辩论的内容。\n\n请言简意赅地总结上述辩论中的双方的观点\n\n请你根据每位辩手的表现和说服力而不是立场的道德来决定谁是本次辩论的优胜者。"},
    ]
    gen = ask_chatgpt(messages)
    action = ""
    while True:
        stop = True
        try:
            action = next(gen)
            stop = False
        except StopIteration:
            pass
        yield action
        if stop:
            break
    

    
def enable_gpt4_evaluation_btn(loop_num):
    if loop_num>1:
        return enable_btn
    return disable_btn

# def visible_evaluation_text():
#     return gr.Textbox(open=True)

def build_debate(models):
    # print(f"RANDOM:{random.random()}")
    notice_markdown = """
# ⚔️  DebateArena ⚔️ 
### Rules
- 你可以选择两个模型来进行辩论（最多三轮）。
- 在每轮辩论结束后，你都可以评判哪个模型更好一些。
- 你可以选择或者自定义辩论的主题和双方的立场。
- 点击"清空"可以重新选择模型和设定辩论主题。
- [[GitHub]](https://github.com/lm-sys/FastChat)
"""
# ### Terms of use
# By using this service, users are required to agree to the following terms: The service is a research preview intended for non-commercial use only. It only provides limited safety measures and may generate offensive content. It must not be used for any illegal, harmful, violent, racist, or sexual purposes. **The service collects user dialogue data and reserves the right to distribute it under a Creative Commons Attribution (CC-BY) license.** The demo works better on desktop devices with a wide screen.


    gr.Markdown(notice_markdown , elem_id="notice_markdown")

    gr.Markdown("### 请选择辩论的主题", elem_id="notice_markdown")
    
    ## 记录第几轮的数字
    loop_num = gr.Number(value=0, visible=False)
    
    theme_selector = gr.Dropdown(
        choices=theme_list,
        value=theme_list[0] if len(theme_list)>0 else "",
        interactive=True,
        show_label=False,
        allow_custom_value=True,
    ).style(container=False)
    
    model_selectors = [None] * 2
    
    with gr.Box(elem_id="share-region-named"):
        with gr.Column():
            with gr.Row():
                with gr.Column():
                    model_selectors[0] = gr.Dropdown(
                        choices=models,
                        value=models[0] if len(models) > 1 else models[0],
                        interactive=True,
                        show_label=False,
                    ).style(container=False)
                with gr.Column():
                    model_selectors[1] = gr.Dropdown(
                        choices=models,
                        value=models[1] if len(models) > 1 else models[0],
                        interactive=True,
                        show_label=False,
                    ).style(container=False)
            with gr.Row():
                #
                with gr.Column(scale=1, min_width=50):
                    standA_text = gr.Textbox(value=f"{debate_list[0]['positive']}", show_label=False, max_lines=1)
                    # standA_text = gr.Textbox(value=f"", show_label=False)
                with gr.Column(scale=0.3, min_width=50):
                    theme_exchange_btn = gr.Button(value="交换持方", interactive=True)
                with gr.Column(scale=1, min_width=50):
                    standB_text = gr.Textbox(value=f"{debate_list[0]['negative']}", show_label=False, max_lines=1)
                    # standB_text = gr.Textbox(value=f"", show_label=False)
                # 
        
        with gr.Column():
            with gr.Row():
                chatbot = Chatbot(
                    label="Debate", elem_id=f"chatbot", visible=True
                ).style(height=880)
        
        with gr.Box() as button_row:
            with gr.Column():
                with gr.Row():
                    leftvote_btn = gr.Button(value="👈  左边更好", interactive=False)
                    rightvote_btn = gr.Button(value="👉  右边更好", interactive=False)
                    tie_btn = gr.Button(value="🤝  我无法做出判断", interactive=False)
                    bothbad_btn = gr.Button(value="👎  我都不赞同", interactive=False)
                # with gr.Row():
                #     evaluation_btn = gr.Button(value="GPT4 总结打分评估", interactive=False)
                # with gr.Row():
                #     evaluation_text = gr.Textbox(value="", visible=True, open=False, label="GPT4 evaluation")

    # with gr.Row():
    #     send_btn = gr.Button(value="开始", visible=True)

    with gr.Row() as button_row2:
        regenerate_btn = gr.Button(value="🔄  重新生成该句子", interactive=False)
        clear_btn = gr.Button(value="🗑️  清空", interactive=False)
        evaluation_btn = gr.Button(value="GPT4 总结打分评估", interactive=False)
        send_btn = gr.Button(value="开始", visible=True)
        # share_btn = gr.Button(value="📷  分享")
    with gr.Row():
        evaluation_text = gr.Textbox(value="", visible=True, open=False, label="GPT4 evaluation")

    with gr.Accordion("Parameters", open=False, visible=True) as parameter_row:
        temperature = gr.Slider(
            minimum=0.0,
            maximum=1.0,
            value=0.7,
            step=0.1,
            interactive=True,
            label="Temperature",
        )
        top_p = gr.Slider(
            minimum=0.0,
            maximum=1.0,
            value=1.0,
            step=0.1,
            interactive=True,
            label="Top P",
        )
        max_output_tokens = gr.Slider(
            minimum=16,
            maximum=1024,
            value=512,
            step=64,
            interactive=True,
            label="Max output tokens",
        )

    
    # Register listeners
    btn_list = [
        leftvote_btn,
        rightvote_btn,
        tie_btn,
        bothbad_btn,
        regenerate_btn,
        clear_btn,
    ]
    
    btn_list_except_clear = [
        leftvote_btn,
        rightvote_btn,
        tie_btn,
        bothbad_btn,
        regenerate_btn,
    ]
    
    send_btn.click(disable_theme_models, None, [theme_selector, standA_text, standB_text, theme_exchange_btn]+model_selectors).then(next_step_arena_debate, model_selectors+[theme_selector, standA_text, standB_text, chatbot, loop_num, temperature, top_p, max_output_tokens], [chatbot, send_btn]+ btn_list + [loop_num]).then(enable_gpt4_evaluation_btn, loop_num, evaluation_btn).then(flash_buttons, loop_num, btn_list)
    
    regenerate_btn.click(regenerate, model_selectors+[theme_selector, standA_text, standB_text, chatbot, loop_num, temperature, top_p, max_output_tokens], [chatbot, send_btn]+ btn_list + [loop_num]).then(enable_gpt4_evaluation_btn, loop_num, evaluation_btn).then(flash_buttons, loop_num, btn_list)
    
    leftvote_btn.click(left_vote, [chatbot] + model_selectors + [theme_selector, standA_text, standB_text], btn_list_except_clear)
    rightvote_btn.click(right_vote, [chatbot] + model_selectors + [theme_selector, standA_text, standB_text], btn_list_except_clear)
    tie_btn.click(tie_vote, [chatbot] + model_selectors + [theme_selector, standA_text, standB_text], btn_list_except_clear)
    bothbad_btn.click(bothbad_vote, [chatbot] + model_selectors + [theme_selector, standA_text, standB_text], btn_list_except_clear)
    
    clear_btn.click(clear_history, None, [chatbot, theme_selector]+btn_list+[loop_num, send_btn, evaluation_btn, evaluation_text]).then(enable_theme_models, None, [theme_selector, standA_text, standB_text, theme_exchange_btn]+model_selectors)
    
    theme_selector.change(theme_selector_change, 
                          theme_selector, 
                          [standA_text, standB_text])
    
    theme_exchange_btn.click(exchange_theme, 
                             [standA_text, standB_text], 
                             [standA_text, standB_text])

    evaluation_btn.click(gpt4_evaluation, chatbot, evaluation_text)

    return (
        model_selectors,
        chatbot,
        theme_selector,
        send_btn,
        button_row,
        button_row2,
        parameter_row,
    )

def build_demo(models):
    with gr.Blocks(
        title="Chat with Open Large Language Models",
        theme=gr.themes.Base(),
        css=block_css,
    ) as demo:
        with gr.Tabs() as tabs:
            with gr.Tab("Debate", id=0):
                (
                    model_selectors,
                    chatbot,
                    theme_selector,
                    send_btn,
                    button_row,
                    button_row2,
                    parameter_row,
                ) = build_debate(models)
    return demo

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument("--port", type=int)
    parser.add_argument(
        "--controller-url",
        type=str,
        default="http://localhost:21001",
        help="The address of the controller.",
    )
    parser.add_argument(
        "--add-chatgpt",
        action="store_true",
        help="Add OpenAI's ChatGPT models (gpt-3.5-turbo, gpt-4)",
    )
    parser.add_argument(
        "--add-claude",
        action="store_true",
        help="Add Anthropic's Claude models (claude-v1, claude-instant-v1)",
    )
    parser.add_argument(
        "--add-palm",
        action="store_true",
        help="Add Google's PaLM model (PaLM 2 for Chat: chat-bison@001)",
    )
    parser.add_argument(
        "--concurrency-count",
        type=int,
        default=10,
        help="The concurrency count of the gradio queue.",
    )
    parser.add_argument(
        "--share",
        action="store_true",
        help="Whether to generate a public, shareable link.",
    )
    parser.add_argument(
        "--gradio-auth-path",
        type=str,
        help='Set the gradio authentication file path. The file should contain one or more user:password pairs in this format: "u1:p1,u2:p2,u3:p3"',
        default=None,
    )
    args = parser.parse_args()
    
    ###
    args.add_chatgpt = True
    ###
    
    # Set authorization credentials
    auth = None
    if args.gradio_auth_path is not None:
        auth = parse_gradio_auth_creds(args.gradio_auth_path)
    
    models = get_model_list(
        args.controller_url, args.add_chatgpt, args.add_claude, args.add_palm
    )
    demo = build_demo(models)
    demo.queue(
            concurrency_count=args.concurrency_count, status_update_rate=10, api_open=False
    ).launch(
        server_name=args.host,
        server_port=args.port,
        share=args.share,
        max_threads=200,
        auth=auth,
    )


main()