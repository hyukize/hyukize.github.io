---
layout: post
title: "LangChain을 활용한 RAG 구현 2 - API와 친해지기"
date: 2025-10-08 11:59:27 +0900
category: AI 
tags: ["langchain"]
---


## OpenAI API

**REST API** 사용하는 방법도 있으나 **Python SDK** 사용하는것이 가장 편리

`poetry add openai==1.70.0` 으로 openai python sdk 설치 완료

- Chat Completion API -> 단순한 질의, 이전 대화 기억 x
- Assistant API, **Response API** -> 대화 상태 관리 기능 보유 (Assistant API는 2026 상반기예 폐지될 예정)
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
    https://hyukize.github.io 에 방문하여 블로그의 내용과 주제, 블로그 운영자에 대한 정보를 알려주세요.
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

tools에 검색 기능인 web_search_preview 를 지정하였다. 만약에 빼고 돌리면 어떻게 될까?

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
    https://hyukize.github.io 에 방문하여 블로그의 내용과 주제, 블로그 운영자에 대한 정보를 알려주세요.
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

조금씩 나누어 출력되는 방식을 보아, Streaming 이 잘 구현되고 있었다.

이번에 사용하면서 느낀 점이 있다면, Claude의 한국어가 상당히 유창하고 속도가 꽤나 빠르다.(모델 차이도 있겠지만)

<br>

## 비동기 처리
