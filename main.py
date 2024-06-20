import os
import json
from crewai import Agent, Task, Crew
from duckduckgo_search import DDGS
from langchain.tools import tool
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from ollama import Client
from langchain_core.prompt_values import StringPromptValue
import time
import threading

print("Script started")

# Load actions from JSON file
with open("actions.json", "r") as file:
    actions = json.load(file)["actions"]

@tool("Internet Search Tool")
def internet_search_tool(query: str) -> list:
    """Search Internet for relevant information based on a query."""
    ddgs = DDGS()
    results = ddgs.text(keywords=query, region='wt-wt', safesearch='moderate', max_results=5)
    return results

print("Internet search tool defined")

class OllamaLLM:
    def __init__(self, model='phi3'):
        self.client = Client()
        self.model = model
        print(f"Ollama LLM initialized with model: {model}")

    def _call_with_timeout(self, prompt_str, timeout=180):  # Timeout set to 180 seconds
        result = [None]
        exception = [None]

        def target():
            try:
                response = self.client.chat(model=self.model, messages=[{'role': 'user', 'content': prompt_str}])
                result[0] = response['message']['content']
            except Exception as e:
                exception[0] = e

        thread = threading.Thread(target=target)
        thread.start()
        thread.join(timeout)
        if thread.is_alive():
            raise TimeoutError("LLM call timed out")
        if exception[0]:
            raise exception[0]
        return result[0]

    def __call__(self, prompt):
        print(f"Received prompt of type {type(prompt)}: {prompt}")

        if not isinstance(prompt, (str, list, StringPromptValue)):
            raise ValueError("Prompt must be of type `str`, `List[str]`, or `StringPromptValue`")

        if isinstance(prompt, StringPromptValue):
            prompt = prompt.text  # Extract text from StringPromptValue

        if isinstance(prompt, list):
            prompt = prompt[0]  # Take the first element if it's a list
        
        prompt_str = str(prompt)  # Ensure prompt is a string
        print(f"Sending prompt to Ollama LLM: {prompt_str}")

        try:
            start_time = time.time()
            response = self._call_with_timeout(prompt_str)
            end_time = time.time()
            print(f"Ollama LLM response time: {end_time - start_time} seconds")
        except Exception as e:
            print(f"Error during LLM call: {e}")
            raise e

        print(f"Ollama LLM response: {response}")
        return response
    
    def bind(self, *args, **kwargs):
        return self

ollama_llm = OllamaLLM(model='phi3')

print("Ollama LLM instance created")

template = """Investigate the latest AI trends in {topic}."""
def setup_llm_chain(template: str, topic: str) -> LLMChain:
    prompt = PromptTemplate(
        input_variables=["prompt", "context"],
        template=template,
    )
    print("LLM chain setup complete")
    return LLMChain(prompt=prompt, llm=ollama_llm)

agents = []
tasks = []

# Create agents and tasks based on the actions defined in the JSON file
for action in actions:
    agent = Agent(
        role=action["role"],
        goal=action["description"],
        backstory=action["context"],
        verbose=True,
        memory=True,
        tools=[internet_search_tool],
        llm=ollama_llm,
    )
    agents.append(agent)

    task = Task(
        description=action["task"],
        expected_output="A comprehensive 4 paragraphs long report on the latest AI trends.",
        tools=[internet_search_tool],
        agent=agent,
        output_file=action["output_file"],
    )
    tasks.append(task)

crew = Crew(
    agents=agents,
    tasks=tasks
)

print("Crew created")

result = crew.kickoff({"topic": "how AI will change Cyber Security Roles in the United Kingdom this coming decade"})
print(result)

# Save the final result to the specified output file
final_output_file = tasks[0].output_file
with open(final_output_file, "w") as f:
    f.write(result)

print(f"Final result saved to {final_output_file}")
