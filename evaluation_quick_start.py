from langsmith import Client
from langsmith.schemas import Run, Example
from langsmith.evaluation import evaluate
from langchain_community.llms import Ollama
import dotenv


dotenv.load_dotenv()

client = Client()

# Define dataset: these are your test cases
dataset_name = "Rap Battle Dataset"
dataset = client.create_dataset(dataset_name, description="Rap battle prompts.")
client.create_examples(
    inputs=[
        {"question": "a rap battle between Atticus Finch and Cicero"},
        {"question": "a rap battle between Barbie and Oppenheimer"},
    ],
    outputs=[
        {"must_mention": ["lawyer", "justice"]},
        {"must_mention": ["plastic", "nuclear"]},
    ],
    dataset_id=dataset.id,
)

# Define AI system
model = Ollama(model="llama2")

def predict(inputs: dict) -> dict:
    messages = inputs["question"]
    response = model.invoke(messages)
    return {"output": response}

# Define evaluators
def must_mention(run: Run, example: Example) -> dict:
    prediction = run.outputs.get("output") or ""
    required = example.outputs.get("must_mention") or []
    score = all(phrase in prediction for phrase in required)
    return {"key":"must_mention", "score": score}

experiment_results = evaluate(
    predict,
    data=dataset_name,
    evaluators=[must_mention],
    experiment_prefix="rap-generator",
    metadata={
        "version": "1.0.0",
    },
)