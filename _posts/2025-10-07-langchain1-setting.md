---
layout: post
title: "LangChain을 활용한 RAG 구현 1 - 준비"
date: 2025-10-07 00:08:00 +0900
lastmod: 2025-10-07 00:08:00 +0900
categories: [AI]
tags: [rag, llm, langchain, langsmith, gpt, openai, dotenv, poetry]
toc: true
---

## 프로젝트 시작에 앞서

퇴사 후 1년..  

불지옥 취업난에 가만히 손 놓고 있을 수는 없기에 llm 프로젝트를 진행하기로 했다.  
각 기업에서 AI Agent 관련 내용을 언급할 때마다 꼭 **RAG**라는 단어가 나왔었고 문득 그 단어의 뜻이 궁금해졌다.  
나도 웃긴 게, 인터넷 검색도 안 해보고 냉큼 서점부터 뛰어갔다.  
일단 머리부터 박고 프로젝트의 풍파를 몸으로 다 맞아가며 배우는 게 원래의 나 아니겠는가.  
그렇게 서점에서 책을 사고 돌아오는 동안 너무 기대가 되었다.  
나는 상상 속에서 잠깐이나마 업무 자동화의 신이었으며, 챗봇의 권위자가 되었다.  
언제까지 망상만 하고 있을 수는 없으니 이제 시작해야지.  

**RAG**... 내가 만들어 본다.\

<br>

> &nbsp; ~~내가 할 수 있을까...?~~&nbsp;&nbsp; **Yes**

<br>

## RAG

최근 대형 언어 모델(LLM)의 활용이 늘어나면서 **RAG**라는 개념이 주목받고 있다.
간단히 말해 RAG는 **외부 문서를 검색하여 LLM이 답변 생성에 참고하도록 하는 구조**이다.
이를 통해 LLM의 최신성 부족, 사내 데이터 보안, 환각(Hallucination) 등의 문제를 보완할 수 있다고 한다.

> 쉽게 말해 **RAG**란, LLM에 추가 자료를 줘서 내가 원하는 Task를 더 잘 수행하도록 하는 기술  
>
> - *pdf 강의 자료를 올려줄 테니 이걸 기반으로 시험 때 나에게 답을 알려줘!*
> - *우리 회사에 관련된 DB를 줄 테니 이 내용을 기반으로 자동으로 고객의 질문에 응답해 줘!*
{: .prompt-tip}
<br>

## 왜 RAG 인가?

기존 ChatGPT와 같은 모델은 다음과 같은 한계를 가지고 있다:

1. 최신 정보가 반영되어 있지 않음 - 검색 기능을 활용해 대신 제공
2. 기업 내부의 문서, 데이터 활용 불가
3. 출처 불명으로 인한 잘못된 답변(환각) 발생 가능
4. 특정 도메인 지식에 약함

**RAG는 이러한 한계를 보완**하여, 최신 문서와 DB를 기반으로 답변을 생성하고 신뢰성을 높여준다.  

> ChatGPT에서 제공하는 기본 RAG를 사용해 본 적이 있다. 몇몇 질문에는 답을 잘해주더니 조금만 더 깊게 질문하면 바로 이상한 답을 하더라.. 업로드할 수 있는 문서의 양도 정해져 있고 성능도 한계가 있는 듯하다.
<br>

## RAG 파이프라인 (사전 단계: Pre-process)

RAG는 단순 업로드 이상의 과정을 필요로 한다. 일반적인 사전 단계는 다음과 같다.

1. **문서 로드(Document Load)**: PDF, Word, 웹페이지 등 다양한 문서 불러오기
2. **텍스트 분할(Text Split)**: 문서를 적절한 단위(청크)로 분할, 필요시 오버랩 적용
3. **임베딩(Embedding)**: 각 청크를 벡터로 변환하여 의미를 수치화
4. **벡터 스토어 저장(Vector Store)**: 벡터와 메타데이터를 DB에 저장하여 검색 대비

ex) *“새콤달콤한 맛” → ‘단맛’, ‘신맛’과 유사도 계산을 통해 연관 문서 검색 가능.*
<br>

## RAG 파이프라인 (실행 단계: Runtime)

1. **리트리버(Retriever)**: 쿼리를 임베딩 후 DB에서 유사도 높은 청크 검색
2. **프롬프트 구성(Prompting)**: 검색된 청크를 LLM에 컨텍스트로 전달
3. **LLM 응답**: 근거 기반 답변 생성 (출처 포함 가능)
4. **체인 생성(Chain)**: LangChain 등으로 전체 프로세스를 하나의 파이프라인으로 묶음
<br>

## 성능 비교 및 효과

- **구현 난이도**: Prompt Eng. < **RAG** << PEFT < Full Fine-tuning
- **최신성**: 사용자가 직접 최신 문서를 DB에 넣어 업데이트 가능 → RAG가 압도적
- **투명성**: 어떤 문서가 답변에 사용되었는지 추적 가능
- **환각 방지**: “근거 문서 기반 답변”을 강제하여 잘못된 추측 최소화

실험 결과, 단순 검색 대비 정확도가 **45% → 98%**까지 향상되었다고 한다. &nbsp; -> &nbsp; ~~필수잖아?~~
<br>

## LangChain으로 직접 구축하기

LangChain은 RAG 파이프라인을 쉽게 설계할 수 있는 프레임워크이다.

- **모듈화**: 문서 로드 → 임베딩 → 벡터 저장 → 검색 → 답변 과정을 체인으로 구성
- **유연성**: 임베딩 모델, 벡터 DB(Milvus 등), 검색 알고리즘(MMR 등) 자유롭게 교체 가능
- **추적성**: LangSmith로 검색·생성 과정을 투명하게 모니터링

코드 흐름은 다음과 같다.

```python
chain = (
  {"context": retriever, "question": RunnablePassthrough()}
  | prompt
  | llm
  | StrOutputParser()
)

question = "대한민국의 인구수는?"
response = chain.invoke(question)
print(response)
```

<br>

## 환경설정

1. **poetry**  

   기존에는 virtualenv(wrapper 포함)나 miniconda를 주로 사용했었는데 이번 계기로 poetry를 사용해 보았다.  

   pyproject.toml 파일로 프로젝트를 관리하는데 기존 requirement.txt로 관리하는 방식보다 훨씬 편하다고 한다.  

   사용하고자 하는 폴더에서 poetry init 명령어로 초기화를 진행하면 pyproject.toml이 생성된다.  

   여기서는 프로젝트의 메타데이터와 함께 프로젝트의 의존성을 관리한다.  

   다만 중요한 점은 내가 설치하기로 명시한 패키지들만 구성이 되어있고 나머지들은 poetry.lock에서 관리를 진행한다. (torch 하나 설치했을 때 얼마나 많은 새끼 패키지들이 설치되었었는지.. 아찔한 기억이 났다.)

   기존에는 pip freeze를 통해 requirement.txt를 만들면 직접 설치하지 않은 패키지들이 너무 많이 있었고 이를 수동으로 관리하기도 힘들었다.  

   또 -D 옵션을 통해서 개발용 패키지라는 것을 명시하면 서비스에만 필요한 패키지랑 구분하여 관리할 수도 있다.
   <br>

2. **dotenv**  

   주로 linux, mac 위주로 개발을 진행하다 보니. bashrc에 환경 변수를 등록하고 사용하거나 export로 임시로 진행했었는데 사용해 보니 편리한 것 같다.  
   사용법도 간단하다. 우선. env 파일에 환경 변수를 입력해 준 뒤,

   ```text
   HYUKIZE_AGE=secret
   HYUKIZE_IS_HANDSOME=True
   ```

   불러오고 싶은 python 코드 상에서

   ```python
   from dotenv import load_dotenv
   
   # 같은 경로에 있으면
   load_dotenv()

   # 다른 경로에 있으면
   path = '.env 파일 경로'
   env_file = dotenv.find_dotenv(path)
   dotenv.load_dotenv(env_file)
   ```

   이렇게만 입력해 주면 끝이다.  

   이렇게 등록한 환경 변수를 불러올 때는

   ```python
   import os

   age = os.getenv("HYUKIZE_AGE")
   truth = bool(os.getenv("HYUKIZE_IS_HANDSOM"))
   print(age, truth)  # secret True
   ```

   <br>

3. **OpenAI, LangSmith, Anthropic API key 발급**

   billing에서 결제수단 등록하고 나서 결제했는데, 하아ㅏㅏ... 나는 chatgpt 유료 사용자라 이미 사용 가능한 크레딧이 있었다. 확인해 보고 할걸... 또 메인 페이지에서 billing이랑 api key 발급 페이지로 넘어가는 버튼을 못 찾아서 한참을 헤맸다.
   이렇게 발급받은 API key 들을. env 파일에 올려두었다. (key는 잊어버리지 않게 따로 보관 필수!)  
   <br>

4. **.env에 발급받은 API Key 입력**
   이후 사용할때는 dotenv.load_dotenv()로 불러와 사용한다.
<br>

## Terminology (더 공부하고 찾아봐야 할 내용)

- LCEL
- Context Window
- Context Length
- BPE
- 토큰화 방식 (문자 기반, 단어 기반, 서브 워드 기반)
- Retriever 이름의 유래
- Hyde 검색
- 벡터 스토어(DB)
- PEFT
- Prompt Engineering
- User prompt / System prompt
