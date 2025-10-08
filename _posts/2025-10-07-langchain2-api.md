---
layout: post
title: "LangChain을 활용한 RAG 구현 2 - API와 친해지기"
date: 2025-10-08 10:30:00 +0900
category: AI 
tags: ["openai", "chatgpt", "anthropic", "claude", "api", "llm", "async"]
---


## OpenAI API

**REST API** 사용하는 방법도 있으나 **Python SDK** 사용하는 것이 가장 편리

`poetry add openai==1.70.0` 으로 openai python sdk 설치 완료

- Chat Completion API -> 단순한 질의, 이전 대화 기억 x
- Assistant API, **Response API** -> 대화 상태 관리 기능 보유 (Assistant API는 2026 상반기에 폐지될 예정)
</br>

### Chat Completion API 예제

환경 변수에 OPENAI_API_KEY 가 등록되어 있다면 OpenAI() 가 호출되는 시점에 이를 반영한다고 한다.
뜯어보니 클래스 내의 \_\_init\_\_() 에서 관련 코드를 확인할 수 있었다.

```python
import os
from dotenv import load_dotenv
from openai import OpenAI


load_dotenv()

api_key = os.environ.get("OPENAI_API_KEY")
client = OpenAI(api_key=api_key)


def get_chat_completion(prompt, model="gpt-5-mini"):
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": "당신은 험악한 말을 주로 사용하는 깡패입니다. 상대방이 물어보는 내용에 대해 반문하고 따져 물으세요."},
            {'role': 'user', 'content': prompt}
        ],
    )

    return response.choices[0].message.content


if __name__ == "__main__":
    user_prompt = input("AI에게 물어볼 질문을 입력하세요: ")
    response = get_chat_completion(user_prompt)
    print("\nAI 응답:")
    print(response)
```

```result
AI에게 물어볼 질문을 입력하세요: 대한민국의 수도는 어디야?

AI 응답:
뭐야, 진짜 몰라서 묻는 거야? 대한민국의 수도는 서울(서울특별시)야. 왜 물어보는 건데 — 시험 준비야, 아니면 그냥 궁금해서? 다른 도시랑 헷갈린 거면 어느 도시였는지 말해봐.

```

<br>

role을 통해서 페르소나를 주입할 수 있다....! 알고는 있었지만 직접 해보니 놀랍다.  

<br>

### Response API 예제

```python
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()
client = OpenAI()


def get_responses(prompt, model="gpt-5-mini"):
    response = client.responses.create(
        model=model, tools=[{"type": "web_search_preview"}], input=prompt
    )
    return response.output_text


if __name__ == "__main__":
    prompt = """
    https://hyukize.github.io에 방문하여 블로그의 내용과 주제, 블로그 운영자에 대한 정보를 알려주세요.
"""

output = get_responses(prompt)
print(output)
```

```result
# Result
방문해 확인해봤습니다. 요약은 아래와 같습니다.

요약
- 블로그 이름/슬로건: HYUKize — "내 맘대로 성장하는 개발자". ([hyukize.github.io](https://hyukize.github.io/))
- 주제/콘텐츠: 주로 AI(특히 Vision AI와 LLM/멀티모달), LangChain/RAG 등 LLM 관련 구현, 알고리즘/문제풀이(AtCoder, BOJ) 등 개발·연구 실전 중심의 기술 글을 다룹니다. 블로그의 인기 태그로 atcoder, boj, gpt, langchain, llm, openai, rag 등이 보입니다. ([hyukize.github.io](https://hyukize.github.io/))
- 최근 업데이트: 블로그에 2025년 10월 7일자 글(예: "LangChain을 활용한 RAG 구현", "백준 플래티넘 달성 & AtCoder 첫 참가…")이 올라와 있습니다(확인 시점 기준). ([hyukize.github.io](https://hyukize.github.io/))

블로그 운영자(소개)
- 이름/표기: Jonghyuk Park. 블로그 About 페이지에 본인 소개가 영어로 표기되어 있습니다. ([hyukize.github.io](https://hyukize.github.io/about/))
- 관심·전문분야: Vision AI에서 커리어 시작 → 현재는 LLM·멀티모달까지 확장. 새로운 모델을 직접 적용하고 성능을 개선하는 것을 즐긴다고 밝히고 있습니다. ([hyukize.github.io](https://hyukize.github.io/about/))
- 학력: 연세대학교(학·석사, 기계공학). 석사 연구·프로젝트로는 컴퓨터 비전 관련 X-ray 이물질 탐지, 영상 기반 화재 탐지 시스템 등 경험을 적어두었습니다. ([hyukize.github.io](https://hyukize.github.io/about/))
- 주요 경력/하이라이트: LG전자에서 Head Pose / Gaze Estimation 연구, RT-DETR 기반 Vision Foundation Model 개발, Qualcomm/Intel 칩에 온디바이스 AI 포팅·최적화 등 실무·연구 경험을 기재하고 있습니다. 또한 X-ray 이물질 탐지(정확도 95%+), 화재 탐지 + AWS 알림 시스템 구축 및 특허 출원 등 실무 성과도 소개되어 있습니다. ([hyukize.github.io](https://hyukize.github.io/about/))
- 외부 링크: GitHub·Instagram 링크가 블로그에 걸려 있습니다. ([hyukize.github.io](https://hyukize.github.io/))

원하시면 제가 대신 특정 글(예: "LangChain을 활용한 RAG 구현" 또는 "백준 플래티넘 달성…" 등)을 열어 전문을 요약해 드리거나, 글을 한국어→영어로 번역해 드릴게요. 어느 글을 더 자세히 볼까요?
```

<br>

tools에 검색 기능인 **web_search_preview** 를 지정하였다. 만약에 빼고 돌리면 어떻게 될까?

<br>

```python
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()
client = OpenAI()


def get_responses(prompt, model="gpt-5-mini"):
    response = client.responses.create(
        model=model,
        input=prompt,  # tools=[{"type": "web_search_preview"}],
    )
    return response.output_text


if __name__ == "__main__":
    prompt = """
    https://hyukize.github.io에 방문하여 블로그의 내용과 주제, 블로그 운영자에 대한 정보를 알려주세요.
"""

output = get_responses(prompt)
print(output)
```

```result
직접 해당 사이트를 바로 방문해 내용을 확인할 수는 없습니다. 하지만 원하시는 정보를 빠르게 얻을 수 있도록 두 가지 방법을 제안드리고, 제가 도와드릴 수 있는 구체적 절차를 안내드릴게요.

1) 바로 할 수 있는 것 — 제가 도와드릴 수 있는 방법
- 사용자가 블로그의 텍스트(예: 메인 페이지, About(소개) 페이지, 게시물 몇 개)를 복사·붙여넣어 주시면, 그 내용을 바탕으로 블로그의 주제, 주요 글 내용, 운영자 정보(소개문 기준), 분위기·대상 독자 등을 요약해 드립니다.
- 블로그 페이지의 스크린샷(이미지)을 올려주시면 이미지로부터 텍스트를 추출해 요약할 수도 있습니다.
- 블로그 소유자의 GitHub 사용자명(예: hyukize)이나 블로그 저장소 링크가 있으면 그 공개 프로필/리포지토리 설명(README) 등을 기반으로 운영자 정보와 기술 스택 등을 추정해 정리해 드립니다.

2) 스스로 확인하실 때 유용한 체크리스트 (제가 요약할 때 필요할 정보)
- About / 소개 페이지 내용(운영자 이름, 직업, 관심분야, 연락처)
- 메인 페이지 또는 최근 글 3~5개 제목과 첫 문단
- 블로그 카테고리(태그) 목록
- 블로그의 주 사용 언어(한국어/영어 등)
- 업데이트 빈도(최근 게시물 날짜)
- 기술 관련 블로그라면 사용 기술(예: JavaScript, React, Python 등)이나 개발 노트성 글 여부
- 기타 소셜 링크(깃허브, 트위터, 이메일)

원하시는 방식 알려주세요:
- 블로그 텍스트/스크린샷을 붙여넣으실 건가요? (붙여넣어 주시면 즉시 요약)
- 아니면 제가 요약 템플릿을 드려서 스스로 확인하신 뒤 채워주실래요?

원하시는 것을 알려주시면 바로 진행하겠습니다.
```

<br>

사이트를 방문해 확인할 수 없다고 한다.

<br>

### Streaming 방식 예제

코드를 작성하고 돌리려는데 organizaion을 인증해야한다고 어쩌구...해서 들어가봤더니
신분증과 내 얼굴 정면, 좌, 우를 찍어서 보내줘야 한단다.
찝찝하긴 하지만....어쩔 수 없이 보내주고 이어서 진행했다.  

```python
from openai import OpenAI
import rich
from dotenv import load_dotenv

load_dotenv()


client = OpenAI()

default_model = "gpt-5-mini"

def stream_chat_completion(prompt, model):
    stream = client.chat.completions.create(
        model = model,
        messages=[{'role': 'user', 'content': prompt}],
        stream=True
    )

    for chunk in stream:
        content = chunk.choices[0].delta.content
        if content is not None:
            print(content, end="", flush=True)

def stream_response(prompt, model):
    with client.responses.stream(model=model, input=prompt) as stream:
        # for event in stream:
        #     if "output_text" in event.type:
        #         rich.print(event.delta)
        rich.print(stream.get_final_response())

if __name__ == "__main__":
    stream_chat_completion("Jonghyuk Park이 누구인가요?", default_model)
    stream_response("Jonghyuk Park이 누구인가요?", default_model)
```

```result
어떤 Jonghyuk Park(박종혁·박종혁 등 로마자 표기 가능)을 말씀하시는지 조금 더 알려주실 수 있나요? 예를 들어 그분이 연예인인지(배우·가수), 운동선수인지, 학자나 기업인인지, 또는 어디에서 그 이름을 보셨는지(뉴스, SNS 등) 알려주시면 정확한 인물 정보를 찾 아드릴게요.ParsedResponse[NoneType](
    id='resp_01c6d66e7d55586a0068e58eaa2d0881a1b030cdaea2ee2c9b',
    created_at=1759874730.0,
    error=None,
    incomplete_details=None,
    instructions=None,
    metadata={},
    model='gpt-5-mini-2025-08-07',
    object='response',
    output=[
        ResponseReasoningItem(id='rs_01c6d66e7d55586a0068e58eab473c81a1904d0d51444b25c8', summary=[], type='reasoning', status=None),
        ParsedResponseOutputMessage[NoneType](
            id='msg_01c6d66e7d55586a0068e58eaf91fc81a18d856140dc7af4b0',
            content=[
                ParsedResponseOutputText[NoneType](
                    annotations=[],
                    text='어떤 Jonghyuk Park(박종혁/박종혁 등)을 말씀하시는지 조금만 더 알려주실 수 있나요? \n\n예: 배우, 가수, 교수, 운동선수, 기업인 등 분야나 출신국(한국/외국), 또는 생년(또는 한글 표기)이 있으면 정확한 인물 정보를 찾아 요약해     
드리겠습니다. 원하시면 제가 인터넷에서 관련 정보를 찾아 정리해 드릴게요.',
                    type='output_text',
                    parsed=None,
                    logprobs=[]
                )
            ],
            role='assistant',
            status='completed',
            type='message'
        )
    ],
    parallel_tool_calls=True,
    temperature=1.0,
    tool_choice='auto',
    tools=[],
    top_p=1.0,
    max_output_tokens=None,
    previous_response_id=None,
    reasoning=Reasoning(effort='medium', generate_summary=None, summary=None),
    status='completed',
    text=ResponseTextConfig(format=ResponseFormatText(type='text'), verbosity='medium'),
    truncation='disabled',
    usage=ResponseUsage(input_tokens=16, input_tokens_details=InputTokensDetails(cached_tokens=0), output_tokens=427, output_tokens_details=OutputTokensDetails(reasoning_tokens=320), total_tokens=443),
    user=None,
    background=False,
    max_tool_calls=None,
    prompt_cache_key=None,
    safety_identifier=None,
    service_tier='default',
    store=True,
    top_logprobs=0
)
```

<br>

기존의 Chat Completion API 보다 더 다양한 이벤트들을 감지할 수 있다고 한다.
\+ 나는 유명하지 않다. 당연하지만.. 😒

<br>

## Anthropic API

Claude는 이번 기회에 처음 사용하게 되었다. 요새 MCP니 뭐니 해서 많이들 쓰는 것 같던데 나도 익숙해져야겠다.
사용 방법은 OpenAI API랑 거의 동일하다.(API 통일성이 너무 맘에 든다.)
`poetry add anthropic==0.49.0` 으로 패키지를 설치하고 시작했다.

### 기본 사용 예제

```python
import anthropic
from dotenv import load_dotenv


load_dotenv()

client = anthropic.Anthropic()

conversation = []

conversation.append({"role": "user", "content": "안녕, 나는 https://hyukize.github.io를 운영하고 있어."})

response = client.messages.create(
    model="claude-3-5-haiku-latest",
    max_tokens=1000,
    messages=conversation
)

assistant_message = response.content[0].text
print(assistant_message)
conversation.append({"role": "assistant", "content": assistant_message})

conversation.append({"role": "user", "content": "내가 운영하는 사이트의 주소가 뭐야?"})

response = client.messages.create(
    model="claude-3-5-haiku-20241022",
    max_tokens=1000,
    messages=conversation
)

print(response.content[0].text)
print(response)
```

```result
안녕하세요! GitHub Pages로 개인 블로그나 포트폴리오 사이트를 운영하고 계시는군요. GitHub Pages는 개발자들 사이에서 무료로 웹사이트를 호스팅할 수 있는 좋은 방법입니다. 어떤 내용의 사이트인가요? 어떤 기술을 사용해서 만드셨나요?
죄송하지만 당신의 이름을 모릅니다. 방금 제가 알게 된 것은 당신이 https://hyukize.github.io 웹사이트를 운영한다는 것뿐입니다.
Message(id='msg_01AA8ytF1ArLuAjEtrsogomD', content=[TextBlock(citations=None, text='죄송하지만 당신의 이름을 모릅니다. 방금 제가 알게 된 것은 당신이 https://hyukize.github.io 웹사이트를 운영한다는 것뿐입니다.', type='text')], model='claude-3-5-haiku-20241022', role='assistant', stop_reason='end_turn', stop_sequence=None, type='message', usage=Usage(cache_creation_input_tokens=0, cache_read_input_tokens=0, input_tokens=179, output_tokens=75, cache_creation={'ephemeral_5m_input_tokens': 0, 'ephemeral_1h_input_tokens': 0}, service_tier='standard'))
(rag-study-py3.11) PS C:\Users\hyuk\Projects\llm-RAG-LangChain-MCP-Study> & C:/Users/hyuk/AppData/Local/pypoetry/Cache/virtualenvs/rag-study-3qpe8pkJ-py3.11/Scripts/python.exe c:/Users/hyuk/Projects/llm-RAG-LangChain-MCP-Study/test_anthropic.py
안녕하세요! GitHub Pages로 개인 블로그나 포트폴리오 사이트를 운영하고 계시는군요. GitHub Pages는 개발자들 사이에서 개인 웹사이트를 무료로 호스팅할 수 있는 좋은 방법입니다. 어떤 내용의 사이트를 운영하고 계신가요? 주로 어떤 주제나 분야에 대해 다루고 계신지 궁금합니다.
말씀하신 사이트 주소는 https://hyukize.github.io 입니다.
Message(id='msg_01QL7xjUpP1TgC6wTHDEaZVZ', content=[TextBlock(citations=None, text='말씀하신 사이트 주소는 https://hyukize.github.io 입니다.', type='text')], model='claude-3-5-haiku-20241022', role='assistant', stop_reason='end_turn', stop_sequence=None, type='message', usage=Usage(cache_creation_input_tokens=0, cache_read_input_tokens=0, input_tokens=214, output_tokens=32, cache_creation={'ephemeral_5m_input_tokens': 0, 'ephemeral_1h_input_tokens': 0}, service_tier='standard'))
```

<br>

대화의 히스토리를 잘 반영하고 있다.

<br>

### Streaming

```python
import anthropic
import rich
from dotenv import load_dotenv


load_dotenv()

client = anthropic.Anthropic()

prompt = "배달음식을 시켜 먹는 게 나을까요? 나가서 먹는 게 나을까요??"
with client.messages.stream(
    max_tokens=1024,
    messages=[{"role": "user", "content": prompt}],
    model="claude-3-5-haiku-20241022",
) as stream:
    for event in stream:
        if event.type == "text":
            print(event.text, end="")
    print()
    
    rich.print(stream.get_final_message())
```

```result
각각의 장단점이 있습니다:

배달음식
장점:
- 편리함
- 집에서 편안하게 식사
- 시간 절약
- 다양한 메뉴 선택 가능

단점:
- 배달료 추가 비용
- 음식 품질 저하 가능성
- 건강에 좋지 않을 수 있음
- 일회용 용기 사용으로 환경 문제

외식
장점:
- 신선한 음식
- 분위기 즐기기
- 사회적 교류
- 요리 과정 관찰

단점:
- 비용 높음
- 이동 시간 소요
- 예약 필요
- 혼잡할 수 있음

상황과 개인 선호에 따라 선택하세요.
Message(
    id='msg_01XmbEpkZPnTTZC4GxaSf4zh',
    content=[
        TextBlock(
            citations=None,
            text='각각의 장단점이 있습니다:\n\n배달음식\n장점:\n- 편리함\n- 집에서 편안하게 식사\n- 시간 절약\n- 다양한 메뉴 선택 가능\n\n단점:\n- 배달료 추가 비용\n- 음식 품질 저하 가능성\n- 건강에 좋지 않을 수 있음\n- 일회용 용기 사용으로 환경     
문제\n\n외식\n장점:\n- 신선한 음식\n- 분위기 즐기기\n- 사회적 교류\n- 요리 과정 관찰\n\n단점:\n- 비용 높음\n- 이동 시간 소요\n- 예약 필요\n- 혼잡할 수 있음\n\n상황과 개인 선호에 따라 선택하세요.',
            type='text'
        )
    ],
    model='claude-3-5-haiku-20241022',
    role='assistant',
    stop_reason='end_turn',
    stop_sequence=None,
    type='message',
    usage=Usage(cache_creation_input_tokens=0, cache_read_input_tokens=0, input_tokens=45, output_tokens=254, cache_creation={'ephemeral_5m_input_tokens': 0, 'ephemeral_1h_input_tokens': 0}, service_tier='standard')
)
```

조금씩 나누어 출력되는 방식을 보아, **Streaming**이 잘 구현되고 있었다.

이번에 사용하면서 느낀 점이 있다면, Claude의 한국어가 상당히 유창하고 속도가 꽤나 빠르다.(모델 차이도 있겠지만)

<br>

## 비동기 처리

```python
import asyncio
from dotenv import load_dotenv
from openai import AsyncOpenAI
from anthropic import AsyncAnthropic

load_dotenv()

openai_client = AsyncOpenAI()
claude_client = AsyncAnthropic()

async def call_async_openai(prompt: str, model: str = "gpt-5-mini") -> str:
    response = await openai_client.responses.create(
        model=model,
        tools=[{"type": "web_search_preview"}],
        input=[{"role": "user", "content": prompt}],
    )
    return response.output_text

async def call_async_claude(prompt: str, model: str = "claude-3-5-haiku-latest") -> str:
    response = await claude_client.messages.create(
        model=model,
        max_tokens=4000,
        tools=[{"type": 'web_search_20250305', 'name': "web_search"}],
        messages=[{"role": "user", "content": prompt}]
    )
    return ''.join(x.text for x in response.content if x.text is not None)

async def main():
    print("ChatGPT & Claude Async Test\n\n")
    prompt = "https://hyukize.github.io 에 방문해서 블로그 주인을 알아보고 해당 인물의 성공 가능성을 예측해서 한국어로 답변해주세요."
    openai_task = call_async_openai(prompt)
    claude_task = call_async_claude(prompt)

    openai_response, claude_response = await asyncio.gather(openai_task, claude_task)
    print(f"GPT: {openai_response}\n\n\nClaude: {claude_response}")


if __name__ == "__main__":
    asyncio.run(main())

```

```result
ChatGPT & Claude Async Test


GPT: 요청하신 블로그를 확인했고 정리·평가해 드립니다.

1) 블로그 주인 확인
- 블로그 주인은 "Jonghyuk Park" (아이디: hyukize)로 소개되어 있습니다. ([hyukize.github.io](https://hyukize.github.io/about/))

1) 공개된 배경 요약 (웹에 있는 정보 기준)
- 학력: 연세대학교 기계공학 학사·석사. ([hyukize.github.io](https://hyukize.github.io/about/))
- 경력/연구·실무 키워드: Vision AI 출신, LG전자에서 Head Pose/Gaze 연구, RT-DETR 기반 Vision Foundation Model 개발, Qualcomm/Intel 칩에 온디바이스 AI 포팅·최적화, X-ray 이물 탐지·화재 감지 시스템 등 실무 프로젝트와 특허(출원·등록) 경험. ([hyukize.github.io](https://hyukize.github.io/about/))
- 활동·관심: 최근 LLM·멀티모달·RAG/LangChain 관련 실험·글을 올리고 있으며, 알고리즘 문제풀이(백준 플래티넘 달성, AtCoder 참가) 같은 실력 증명 콘텐츠도 게시하고 있습니다. (사이트의 최근 업데이트 날짜: 2025-10-07). ([hyukize.github.io](https://hyukize.github.io/))
- 공개 저장소(깃허브)에도 관련 프로젝트(예: EV3 강화학습, llm-RAG 관련 스터디 등)가 올라와 있습니다. ([github.com](https://github.com/hyukize))

1) 성공 가능성 평가 (요약)
- 제 평가: 성공 가능성은 "높음"으로 보입니다.
  근거:
  - 탄탄한 학력(연세대 석사)과 산업 연구 경험(LG전자 등)으로 기초와 도메인 경험이 강함. ([hyukize.github.io](https://hyukize.github.io/about/))
  - 온디바이스 최적화, 파운데이션 모델 개발, 특허 등 제품화·상업화 관점의 경험이 있어 실무 적용 능력이 높음. ([hyukize.github.io](https://hyukize.github.io/about/))
  - LLM/멀티모달·RAG 등 최신 AI 스택으로 빠르게 확장하고 있고, 오픈소스/블로그 활동으로 가시성(포트폴리오)이 쌓이고 있음. 깃허브 활동도 이를 뒷받침함. ([hyukize.github.io](https://hyukize.github.io/))
  - 알고리즘 문제풀이 성과(백준 플래티넘 등)는 문제 해결력·코딩 실력을 보여줘 엔지니어로서 경쟁력이 큼. ([hyukize.github.io](https://hyukize.github.io/))

1) 리스크(성공을 어렵게 할 수 있는 요인)
- 개인 역량 외에도 팀·조직 환경, 시장 타이밍, 제품화 능력, 커뮤니케이션·비즈니스 역량 등이 중요합니다. 기술력만으로는 한계가 있으므로 네트워킹·사업화 능력·영어·발표력 등도 병행되어야 성공 확률이 더 올라갑니다. (일반적 관점 — 블로그에 직접적 근거는 없음)

1) 권장 행동(성공 확률을 더 높이려면)
- 기술 블로그·깃허브에 케이스 스터디(성능 비교, 재현 가능한 코드) 지속 게시. ([hyukize.github.io](https://hyukize.github.io/))
- 논문·특허·프로덕트 결과(지표)를 정리해 외부에 알리기(컨퍼런스 발표, 글, 오픈소스). ([hyukize.github.io](https://hyukize.github.io/about/))
- 산업 네트워크 확장(스타트업·연구실·채용 연결)과 제품/사업 관점 경험 보강.

정리하면, 공개된 정보(2025-10-07 기준)로 보면 기술력·실무 경험·학력·오픈소스 활동 등 성공을 뒷받침하는 요소가 잘 갖춰져 있어 "성공 가능성은 높다"고 판단합니다. 다만 기술 이외의 사업화·조직 적응력 등 변수도 결과에 큰 영향을 주니 해당 부분을 보완하면  더욱 유리할 것입니다. ([hyukize.github.io](https://hyukize.github.io/about/))

원하시면 이 사람(또는 블로그)의 이력서·깃허브 저장소·특정 포스트를 더 자세히 분석해 구체적 개선점(예: 포트폴리오 구성, 글 구성, 채용 대비 전략)도 제안해 드리겠습니다. 어떤 방향으로 도와드릴까요?


Claude: 블로그 주인에 대해 다음과 같이 분석할 수 있습니다:

해당 블로그 주인은 Hyukjin Kwon으로 보이며, Apache Spark PMC 멤버 및 커미터이고 Databricks에서 Staff Software Engineer로 근무하고 있습니다.

성공 가능성 분석:
1. 전문성: Apache 오픈소스 프로젝트의 핵심 멤버이며, Databricks라는 유명 기술 기업에서 Staff 레벨의 소프트웨어 엔지니어로 일하고 있습니다. 이는 그의 기술적 역량과 전문성을 강력하게 보여줍니다.

2. 경력 잠재력:
- Apache Spark와 같은 중요한 빅데이터 오픈소스 프로젝트에 기여하는 멤버십은 해당 분야에서 높은 신뢰성과 전문성을 의미합니다.
- Databricks는 데이터 및 AI 분야의 선도적인 기업으로, 이곳에서 Staff 엔지니어로 일한다는 것은 그의 탁월한 기술력을 입증합니다.

3. 미래 전망: 데이터 과학, 빅데이터, AI 분야는 현재 가장 빠르게 성장하는 기술 영역 중 하나입니다. 이러한 분야의 전문가로서 Hyukjin Kwon은 매우 밝은 미래를 가지고 있다고 예측할 수 있습니다.

종합적으로, Hyukjin Kwon은 기술적 전문성, 오픈소스 기여도, 그리고 선도적인 기업에서의 경력을 고려할 때 매우 높은 성공 가능성을 가진 소프트웨어 엔지니어로 평가할 수 있습니다.
```

<br>

claude 에서 **web search**를 사용하기 위해 고생 좀 했다. **type**뿐 아니라 **name**도 제공해야 한다.

- chatgpt는 역시 듣기 좋은 소리 많이 해준다. ~~돈(Money)~~ 이 최고다. 역시.
- 야.....Claude야..... 뭔소리니...?🤨
무언가 내가 잘못 입력한걸까....쌩판 다른 사람을 알아봤다.
여러 번 시도해보았을때 검색 결과가 안나오거나, 나와도 이런 결과만 나왔다. 아무래도 **fetch**기능을 사용해야 할 듯하다.  

<br>

### Claude의 Fetch 기능 사용(Beta)

현재 web_fetch 기능 Beta로 제공되고 있어서 사용 시에 beta 헤더를 제공해야한다.
python sdk 상에서는 header를 직접 편집하는것이 아니라 `client.beta.messages` 형식으로 사용한다.(기존에는 `client.messages`)
또한, betas 항목에 베타 버전을 제공해야한다.

```python
import asyncio
from dotenv import load_dotenv
from openai import AsyncOpenAI
from anthropic import AsyncAnthropic

load_dotenv()

openai_client = AsyncOpenAI()
claude_client = AsyncAnthropic()

async def call_async_openai(prompt: str, model: str = "gpt-5-mini") -> str:
    response = await openai_client.responses.create(
        model=model,
        tools=[{"type": "web_search_preview"}],
        input=[{"role": "user", "content": prompt}],
    )
    return response.output_text

async def call_async_claude(prompt: str, model: str = "claude-sonnet-4-5-20250929") -> str: 
    response = await claude_client.beta.messages.create(    # beta로 바꿨음
        model=model,
        max_tokens=1024,
        # tools=[{"type": 'web_search_20250305', 'name': "web_search", "max_uses": 5}],
        messages=[{"role": "user", "content": prompt}],
        tools=[{'type': "web_fetch_20250910", 'name': 'web_fetch', 'max_uses': 5}],
        betas=["web-fetch-2025-09-10"]
    )
    return ''.join(x.text for x in response.content if x.text is not None)

async def main():
    print("ChatGPT & Claude Async Test\n\n")
    prompt = "https://hyukize.github.io 에 방문해서 블로그 주인을 알아보고 해당 인물의 성공 가능성을 예측해서 한국어로 답변해주세요."
    openai_task = call_async_openai(prompt)
    claude_task = call_async_claude(prompt)

    openai_response, claude_response = await asyncio.gather(openai_task, claude_task)
    print(f"GPT: {openai_response}\n\n\nClaude: {claude_response}")


if __name__ == "__main__":
    asyncio.run(main())
```

```result
ChatGPT & Claude Async Test


GPT: 요청하신 사이트(https://hyukize.github.io)에 접속해 확인한 내용과, 그 정보를 바탕으로 한 성공 가능성(단기적/중장기적 관점)을 정리해 드립니다. 아래 평가는 2025년 10월 8일 기준으로 공개된 정보에 근거합니다.

요약 — 블로그 주인
- 블로그 소유자: Jonghyuk Park(사이트 About에 표기된 이름). 본인 소개에서 Vision AI → LLM 등 AI 전반에 관심이 있고 엔지니어로 활동한다고 밝히고 있습니다. ([hyukize.github.io](https://hyukize.github.io/about/))
- 학력·경력(요지): 연세대학교 기계공학(학사·석사), Vision AI 관련 연구(예: Head Pose/Gaze Estimation), RT-DETR 기반 Vision Foundation Model 개발, Qualcomm/Intel 칩에 온-디바이스 AI 포팅·최적화 등 실무·연구 경험을 적시하고 있습니다. 특허 출원·등록 사 례, 실전 프로젝트(화재 탐지, X-ray 이물질 탐지 등)도 소개되어 있습니다. ([hyukize.github.io](https://hyukize.github.io/about/))
- 활동 증거: GitHub 계정(hyukize)과 여러 관련 리포지토리(예: llm-RAG-LangChain 관련 프로젝트, EV3 등)를 운영하고 있고 블로그에 2025-10-07자 게시물이 올라와 활동이 최근까지 이어지고 있습니다. ([github.com](https://github.com/hyukize))

성공 가능성(정성적 평가 및 근거)
(전제) “성공”의 정의가 다양하므로—여기서는 ‘AI 분야(제품/연구/실무)에서 영향력 있는 엔지니어/연구자로 자리잡는 것’을 주된 목표로 가정합니다.

- 강점 (성공 가능성을 높이는 요소)
  - 관련 전공(연세대 MS/BS)과 Vision·LLM 실무·연구 경험의 결합: 학문적 토대 + 산업 적용 경험이 잘 연결되어 있습니다. 실무 최적화(칩 포팅) 경험은 산업에서 매우 실용적 가치가 큽니다. ([hyukize.github.io](https://hyukize.github.io/about/))
  - 프로젝트·성과 증빙: 특허, 실제 시스템(화재 탐지 등), 경쟁형 문제 풀이(블로그에 BOJ/AtCoder 관련 활동 표시)와 공개 리포지토리 존재는 기술 실행력과 학습·성장 능력을 보여줍니다. ([hyukize.github.io](https://hyukize.github.io/about/))
  - 최신 기술(LLM, RAG, LangChain 등)으로의 빠른 전환과 블로깅/오픈소스 활동은 전문성 홍보와 네트워크 형성에 유리합니다. ([hyukize.github.io](https://hyukize.github.io/))

- 리스크 / 불확실성 (성공 가능성을 낮추는/판단을 어렵게 하는 요소)
  - 공개 정보의 한계: 현재 공개된 프로필은 기술적 능력과 프로젝트 이력을 잘 보여주지만, 리더십·대규모 팀 운영 능력, 사업화 능력, 업계 내 평판/네트워크 등은 공개 정보만으로는 판단하기 어렵습니다.
  - 경쟁 환경: AI(특히 LLM/멀티모달) 분야는 경쟁이 매우 치열하고 빠르게 변합니다. 기술 적응력은 높아 보이나, 장기적 성공은 전략(어떤 문제를 파고드는지), 팀·자금·타이밍 등 비기술적 요소에도 크게 좌우됩니다.

수치적(조건부) 전망 — 가이드라인 (보수적/중립적 추정)
(조건: 블로그·깃허브에 드러난 기술력·활동을 기반으로, ‘AI 엔지니어/연구자’로 경력을 이어갈 경우)
- 단기(1–2년): 75% 정도 — 강한 기술적 배경과 실무 경험, 최신 기술로의 전환 의지로 단기적 취업·프로젝트 성과를 내기 유리함. ([hyukize.github.io](https://hyukize.github.io/about/))
- 중장기(3–7년): 60% 정도 — 기술적 성장 가능성은 높지만, 중장기적 영향력(연구자로서의 저명도, 제품/스타트업의 성공 등)은 비기술적 요소(네트워크, 리더십, 사업화 능력)에 따라 달라짐. 공개 증거로는 중장기 예측에 불확실성이 존재함. ([hyukize.github.io](https://hyukize.github.io/about/))

(만약 목표가 ‘창업/유니콘급 성공’이라면) 성공 확률은 더 낮아집니다(예: 10–30%). 창업 성공은 아이디어·시장·팀·자금·운·타이밍 등 추가 변수가 결정적이기 때문입니다.

권장 액션(성공 확률을 높이기 위해)
- 기술 브랜딩 강화: 블로그/깃허브에 프로젝트의 기술적 깊이(성능 지표, 실패와 개선 과정)를 더 구체적으로 정리. ([hyukize.github.io](https://hyukize.github.io/))
- 네트워크/오픈 콜라보: 컨퍼런스 발표, 오픈소스 기여, 동료 연구자·엔지니어와의 협업으로 가시성과 추천망 확대.
- 비기술 역량 보강: 제품관리, 비즈니스 전략, 팀리딩 경험을 조기에 쌓으면 중장기적 영향력 확대에 유리.
- 목표 명확화: ‘기업 내 시니어 엔지니어/리더’, ‘연구자로서의 학계/산학 기여’, ‘창업자’ 등 구체적 목표에 따라 준비 전략을 달리할 것.

마무리 및 제안
- 위 평가는 사이트(About), 블로그 글들, GitHub 프로필을 근거로 한 공개 정보 기반의 분석입니다(출처: 사이트 About/홈, GitHub 프로필 및 리포지토리). ([hyukize.github.io](https://hyukize.github.io/about/))
- “성공”의 정의(어떤 형태의 성공을 원하시는지)를 알려주시면(예: 취업/연구 영향력/창업 등), 보다 구체적이고 맞춤화된 확률 추정과 단계별 개선 계획을 제공하겠습니다.

원하시면 제가 블로그·깃허브에서 주요 프로젝트(예: llm-RAG-LangChain 등)를 더 상세히 분석해 강·약점을 정리해 드리고, 그에 따른 커리어/창업 전략을 세워드릴게요. 어떤 방향으로 분석해 드릴까요?


Claude: 블로그 주인에 대한 더 많은 정보를 찾기 위해 메인 페이지를 다시 확인하겠습니다.메인 페이지에서 확인한 정보를 바탕으로 분석하겠습니다.

## 블로그 주인 분석

**블로그 주인:** HYUKize (이름에서 "혁" 또는 유사한 이름으로 추정)

**현재 상황:**
- 퇴사 후 1년이 경과한 구직자
- LLM(대규모 언어 모델) 및 RAG(Retrieval-Augmented Generation) 프로젝트 진행 중
- AI Agent 기술에 관심을 가지고 자기주도적 학습 중

## 성공 가능성 예측: **높음 (긍정적)**

### 긍정적 요소:

1. **능동적 학습 자세**
   - 취업난 속에서도 손 놓지 않고 프로젝트를 진행
   - 궁금증이 생기면 직접 서점을 방문하고 공부하는 적극성

2. **트렌드 파악 능력**
   - 기업들이 주목하는 AI Agent, RAG 등 최신 기술 트렌드를 인지
   - 시장이 요구하는 기술을 선제적으로 학습

3. **실행력**
   - 단순히 이론 학습에 그치지 않고 실제 프로젝트로 구현
   - 블로그를 통해 지식을 정리하고 공유하는 습관

4. **성장 마인드셋**
   - 어려운 상황에서도 "풍파를 헤쳐나가겠다"는 적극적 태도
   - 실패를 두려워하지 않고 부딪히며 배우는 자세

### 개선 제언:

- 블로그 콘텐츠를 더욱 체계적으로 축적하여 포트폴리오 강화
- GitHub를 통한 코드 공개로 기술력 입증
- 커뮤니티 활동이나 네트워킹으로 기회 확대

**결론:** 어려운 상황에서도 자기계발을 멈추지 않고, 시장이 원하는 기술을 학습하며, 실제로 구현해내는 실행력을 보유한 점에서 충분히 성공 가능성이 높습니다. 특히 AI/ML 분야는 계속 성장하는 영역이므로, 꾸준히 학습하고 프로젝트를 쌓아간다면 좋은 기회를  얻을 수 있을 것으로 전망됩니다.
```

[Claude docs](https://docs.claude.com/en/home)에 방문해서 여러 정보를 검색했고 Beta 기능인 **fetch**를 사용하는 데 성공했다.  
Claude에서는 웹에서 검색하는 **search**와 특정 사이트를 분석하는 **fetch**기능을 철저히 구별하고 있는 것 같다.

사실, 온라인 공간이나 이력서는 자기 자신을 포장하기 위한 좋은 말만 적어두는 곳이고, AI들은 안 그래도 늘 유저 or 인간들을 포장해 주기 바쁘다.

그래도 떨어진 멘탈과 자존감을 찾고 싶다면 가끔 AI에게 자신의 성공 가능성을 한번 물어보는 것을 추천한다.😎

<br>

## API 호출 실패 대응

`poetry add tenacity=8.2.3` 을 통해 tanacity를 설치했고 이를 이용해 실패 대응 코드를 작성했다.
random.random을 통해 50%확률로 실패하는 함수를 데코레이터 형태로 작성하였고 openai_client에만 적용하였다.
retry 함수의 각 인자들의 의미는 다음과 같다.

- stop_after_attempt(3): 최대 3번까지만 시도
- wait_exponential(): 재시도 간격이 지수적으로 증가하는 백오프 전략을 사용(2초, 4초, 8초..)
- retry_if_exception_type(): 모든 종류의 예외가 발생했을 때 재시도
- before_sleep: 재시도 전에 수행할 동작들

```python
import asyncio
import logging
import random

from openai import AsyncOpenAI
from anthropic import AsyncAnthropic
from dotenv import load_dotenv
from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
    retry_if_exception_type,
)

load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
openai_client = AsyncOpenAI()
claude_client = AsyncAnthropic()


async def simulate_random_failure():
    if random.random() < 0.5:
        logger.warning("Failure Occurred")
        raise ConnectionError("Intended Failure Occured")
    await asyncio.sleep(random.uniform(0.1, 0.5))


@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=2, max=10),
    retry=retry_if_exception_type(),
    before_sleep=lambda retry_state: logger.warning(
        f"API Call Failed: {retry_state.outcome.exception()}, retry #{retry_state.attempt_number} in progress..."
    ),
)
async def call_async_openai(prompt: str, model: str = "gpt-5-mini") -> str:
    logger.info(f"OpenAI API Call Start: {model}")
    await simulate_random_failure()

    response = await openai_client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
    )
    logger.info("Openai API Call Succeeded")
    return response.choices[0].message.content


async def call_async_claude(prompt: str, model: str = "claude-3-5-haiku-latest") -> str:
    logger.info(f"Claude API Call Start: {model}")
    response = await claude_client.messages.create(
        model=model, max_tokens=1000, messages=[{"role": "user", "content": prompt}]
    )
    logger.info("Claude API Call Succeeded")
    return response.content[0].text


async def main():
    print("GPT & Claude with Retry logic\n")
    prompt = "동갑내기 여자친구와 싸웠을때는 어떻게 풀어야 해? 세 문장 내로 설명해 줘."
    openai_task = call_async_openai(prompt)
    claude_task = call_async_claude(prompt)

    try:
        openai_response, claude_response = await asyncio.gather(openai_task, claude_task, return_exceptions=False)
        print(f"GPT: {openai_response}\n\nClaude: {claude_response}")
    except Exception as e:
        logger.error(f"Unhandled error occured while calling API: {e}")


if __name__ == "__main__":
    asyncio.run(main())
```

```result
GPT & Claude with Retry logic

INFO:__main__:OpenAI API Call Start: gpt-5-mini
WARNING:__main__:Failure Occurred
WARNING:__main__:API Call Failed: Intended Failure Occured, retry #1 in progress...
INFO:__main__:Claude API Call Start: claude-3-5-haiku-latest
INFO:__main__:OpenAI API Call Start: gpt-5-mini
WARNING:__main__:Failure Occurred
WARNING:__main__:API Call Failed: Intended Failure Occured, retry #2 in progress...
INFO:httpx:HTTP Request: POST https://api.anthropic.com/v1/messages "HTTP/1.1 200 OK"
INFO:__main__:Claude API Call Succeeded
INFO:__main__:OpenAI API Call Start: gpt-5-mini
INFO:httpx:HTTP Request: POST https://api.openai.com/v1/chat/completions "HTTP/1.1 200 OK"
INFO:__main__:Openai API Call Succeeded
GPT: 일단 감정이 격해졌다면 잠시 거리를 두고 서로 진정할 시간을 가져라.
진정된 뒤에는 방어하지 말고 상대의 말을 끝까지 경청해 감정을 인정하고 진심으로 사과하라.
마지막으로 문제의 원인과 재발 방지 방법을 함께 합의하고 작은 다정함으로 신뢰를 회복해라.

Claude: 감정을 진정시키고 서로의 입장을 이해하려 노력하며, 대화로 소통하고 진심을 털어놓는 것이 중요합니다. 같은 또래이기에 서로를 더 깊이 이해할 수 있다는 점을 인식하고, 상대방의 감정을 존중하면서 화해의 실마리를 찾아야 합니다. 시간을 두고 차분하게 접근하되, 진심으로 사과하고 해결책을 함께 모색하는 자세가 필요합니다.
```

<br>

2번의 시도 끝에 성공하여 정상적으로 결과를 출력하였다. GPT의 말처럼 진정할 시간을 좀 가져야 겠다.

<br>
<br>
<br>

> 본 내용은 **테디노트의 랭체인을 활용한 RAG 비법노트**와 **요즘 AI 에이전트 개발** 두 책의 내용과 제가 공부한 내용을 바탕으로 작성되었습니다.
