from langchain.prompts import ChatPromptTemplate
from langchain.schema.runnable import RunnableMap
from langserve import RemoteRunnable


llama2 = RemoteRunnable("http://localhost:8000/llama2/")
joke_chain = RemoteRunnable("http://localhost:8000/joke/")

joke_chain.invoke({"topic": "parrots"})

prompt = ChatPromptTemplate.from_messages(
    [("system", "Tell me a long story about {topic}")]
)

chain = prompt | RunnableMap({
    "llama2": llama2,
})

chain.batch([{"topic": "parrots"}, {"topic": "cats"}])