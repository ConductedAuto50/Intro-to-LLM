from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.prompts import ChatPromptTemplate
from langchain.schema import StrOutputParser
from langchain.schema.runnable import Runnable
from langchain.schema.runnable.config import RunnableConfig
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from typing import cast

import chainlit as cl


@cl.on_chat_start
async def on_chat_start():
    model=ChatGoogleGenerativeAI(model="gemini-1.5-pro",disable_streaming=False)
    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "You're an obedient servant who does what i say.",
            ),
            ("placeholder", "{hist}"),
            ("human", "{question}"),
        ]
    )

    runnable = prompt | model | StrOutputParser()
    cl.user_session.set("runnable", runnable)
    hist=[]
    cl.user_session.set("hist", hist)

@cl.on_message
async def on_message(message: cl.Message):
    runnable = cast(Runnable, cl.user_session.get("runnable"))  # type: Runnable
    hist=cast(list,cl.user_session.get("hist"))
    msg = cl.Message(content="")
    hist.append(message.content)
    out=''
    async for chunk in runnable.astream(
        {"question": message.content,"hist":hist},
        config=RunnableConfig(callbacks=[cl.LangchainCallbackHandler()]),
    ):
        await msg.stream_token(chunk)
        out+=chunk
    
    await msg.send()
    hist.append("AI:"+out)