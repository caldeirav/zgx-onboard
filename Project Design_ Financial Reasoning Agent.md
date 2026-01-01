# **Project Design: Financial Reasoning Agent (DeepSeek-R1 Style)**

## **1\. Project Overview**

This notebook project builds a **Financial Analyst Agent** capable of complex reasoning and accurate calculation. We leverage a **"Thinking" Model** (Qwen3-4B-Thinking) fine-tuned with **Unsloth & GRPO**, orchestrate it via **LangGraph**, and empower it with a **Python REPL Tool** for precise arithmetic.

**Key Features:**

* **"Thinking" Architecture:** The model generates explicit reasoning traces (\<think\>...\</think\>) before answering.  
* **Tool-Augmented Reasoning:** The agent uses a Python REPL to execute the math derived during its thinking phase, ensuring 100% arithmetic accuracy.  
* **Before/After Comparison:** We evaluate the *Agent's* performance with the Base Model vs. the Fine-Tuned Model.  
* **Full Observability:** All traces and experiments are logged to **MLflow**.

### **Tech Stack**

* **Model Tuning:** unsloth, trl (GRPO), peft  
* **Orchestration:** langchain, langgraph  
* **Tools:** langchain\_experimental (Python REPL)  
* **Data Processing:** polars, datasets  
* **Evaluation:** mlflow, google-genai  
* **Hardware:** DGX Spark / GB10 GPU / CUDA 13

## **2\. Dataset Selection**

**Dataset:** [ConvFinQA](https://www.google.com/search?q=https://huggingface.co/datasets/fingpt/fingpt-convfinqa).

* **Why:** Contains complex financial questions requiring context retrieval and multi-step math.  
* **Enhancement:** We format the data to reward the model for deriving the correct inputs for the Python tool.

## **3\. Notebook Implementation Plan**

### **Phase 1: Environment & Data Preparation**

We use Polars to prepare the data and define the "Thinking" prompt structure.

\# %%capture  
\# \!pip install unsloth vllm  
\# \!pip install \--no-deps trl peft accelerate bitsandbytes  
\# \!pip install polars langchain langgraph langchain\_experimental mlflow google-genai python-dotenv

import polars as pl  
from datasets import load\_dataset  
from dotenv import load\_dotenv  
import os

\# 1\. Load Secrets (HF\_TOKEN, GOOGLE\_API\_KEY)  
load\_dotenv()

\# 2\. Load Dataset  
dataset \= load\_dataset("FinGPT/fingpt-convfinqa", split="train")  
df \= pl.from\_pandas(dataset.to\_pandas())

\# 3\. Format Data for GRPO  
def format\_data(row):  
    context \= row\["context"\]  
    question \= row\["question"\]  
    \# Instruct model to think, then optional use python, then answer  
    system\_prompt \= """You are a financial analyst.   
    1\. Output your reasoning in \<think\> tags.  
    2\. If calculation is needed, generate Python code.  
    3\. Output final answer in \<answer\> tags."""  
      
    user\_prompt \= f"Context: {context}\\n\\nQuestion: {question}"  
    return {"prompt": \[  
        {"role": "system", "content": system\_prompt},  
        {"role": "user", "content": user\_prompt}  
    \], "answer": row\["answer"\]}

processed\_df \= df.map\_elements(format\_data, return\_dtype=pl.Object)

### **Phase 2: Agent Definition (The "Analyst")**

We define the agent *before* loading the model so we can reuse this structure for both the Base and Tuned versions. We integrate a **Python REPL** tool to handle the math.

from langgraph.graph import StateGraph, START, END  
from langchain\_experimental.utilities import PythonREPL  
from typing import TypedDict, Annotated  
import re

\# 1\. Define Tools  
repl \= PythonREPL()

def python\_calculator(code: str):  
    """Executes Python code to perform financial math."""  
    try:  
        return repl.run(code)  
    except Exception as e:  
        return f"Error: {e}"

\# 2\. Define Agent State  
class AgentState(TypedDict):  
    query: str  
    context: str  
    model\_output: str  
    tool\_output: str  
    final\_answer: str

\# 3\. Factory Function to Create Agent  
def build\_financial\_agent(llm\_engine):  
    """Builds a LangGraph agent wrapping the provided LLM engine."""  
      
    def reason\_node(state: AgentState):  
        \# Construct prompt  
        prompt \= f"{state\['query'\]}\\nContext: {state\['context'\]}"  
        \# Generate (Thinking \+ Potential Code)  
        response \= llm\_engine.generate(prompt)   
        return {"model\_output": response}

    def tool\_node(state: AgentState):  
        \# Extract Python code from model output if present  
        code\_match \= re.search(r"\`\`\`python(.\*?)\`\`\`", state\['model\_output'\], re.DOTALL)  
        if code\_match:  
            code \= code\_match.group(1).strip()  
            result \= python\_calculator(code)  
            return {"tool\_output": result}  
        return {"tool\_output": "No code executed."}

    def answer\_node(state: AgentState):  
        \# Parse final answer from tags or tool result  
        \# If tool was used, we assume the tool output is the answer for this simple design  
        if state\['tool\_output'\] \!= "No code executed.":  
            return {"final\_answer": state\['tool\_output'\]}  
          
        \# Fallback to parsing \<answer\> tags  
        ans\_match \= re.search(r"\<answer\>(.\*?)\</answer\>", state\['model\_output'\])  
        return {"final\_answer": ans\_match.group(1) if ans\_match else "N/A"}

    \# Build Graph  
    workflow \= StateGraph(AgentState)  
    workflow.add\_node("think", reason\_node)  
    workflow.add\_node("calculate", tool\_node)  
    workflow.add\_node("finalize", answer\_node)  
      
    workflow.add\_edge(START, "think")  
    workflow.add\_edge("think", "calculate")  
    workflow.add\_edge("calculate", "finalize")  
    workflow.add\_edge("finalize", END)  
      
    return workflow.compile()

### **Phase 3: Base Model Evaluation**

We load the base model and run the agent to establish a baseline.

from unsloth import FastLanguageModel, PatchFastRL  
import mlflow  
import pandas as pd

\# 1\. Load Base Model  
PatchFastRL("GRPO", FastLanguageModel)  
model, tokenizer \= FastLanguageModel.from\_pretrained(  
    "unsloth/Qwen3-4B-Thinking-2507",   
    max\_seq\_length \= 8192,  
    load\_in\_4bit \= True,  
    fast\_inference \= True,  
    gpu\_memory\_utilization \= 0.8,  
)

\# 2\. Wrap Model for Agent  
\# Simple wrapper to match the agent's expected 'generate' interface  
class ModelWrapper:  
    def generate(self, prompt):  
        inputs \= tokenizer(prompt, return\_tensors="pt").to("cuda")  
        outputs \= model.generate(\*\*inputs, max\_new\_tokens=1024)  
        return tokenizer.decode(outputs\[0\], skip\_special\_tokens=True)

base\_engine \= ModelWrapper()  
base\_agent \= build\_financial\_agent(base\_engine)

\# 3\. Run Baseline Evaluation  
validation\_sample \= processed\_df.head(5).to\_pandas()  
baseline\_results \= \[\]

print("--- Running Base Agent Eval \---")  
mlflow.langchain.autolog()  
with mlflow.start\_run(run\_name="Base\_Model\_Eval"):  
    for \_, row in validation\_sample.iterrows():  
        res \= base\_agent.invoke({"query": row\['prompt'\]\[1\]\['content'\], "context": ""})  
        baseline\_results.append(res\['final\_answer'\])

validation\_sample\['base\_agent\_answer'\] \= baseline\_results

### **Phase 4: Fine-Tuning (Unsloth GRPO)**

We train the model to improve its reasoning and its ability to formulate the correct inputs for the tool.

from trl import GRPOConfig, GRPOTrainer

\# 1\. Define Rewards  
\# We reward:   
\# A) Correct \<answer\> tag format  
\# B) Correct numerical result (extracted from answer)  
\# C) (Implicitly) Correct Python code generation if it leads to the right answer

def accuracy\_reward(completions, answer, \*\*kwargs):  
    rewards \= \[\]  
    for completion, gold in zip(completions, answer):  
        \# Logic to check if completion contains gold answer  
        rewards.append(1.0 if gold in completion\[0\]\['content'\] else 0.0)  
    return rewards

\# 2\. Config & Train  
training\_args \= GRPOConfig(  
    output\_dir \= "qwen3-fin-agent",  
    learning\_rate \= 5e-6,  
    logging\_steps \= 1,  
    per\_device\_train\_batch\_size \= 1,  
    gradient\_accumulation\_steps \= 4,  
    max\_steps \= 300,  
    report\_to \= "mlflow", \# Log training metrics to MLflow  
    use\_vllm \= True,  
)

trainer \= GRPOTrainer(  
    model \= model,  
    processing\_class \= tokenizer,  
    reward\_funcs \= \[accuracy\_reward\],  
    args \= training\_args,  
    train\_dataset \= processed\_df.to\_pandas(),  
)

trainer.train()

### **Phase 5: Tuned Model Evaluation & Comparison**

We reuse the *exact same agent structure*, but now the underlying model has been updated in-place by the trainer.

\# 1\. Re-Wrap Tuned Model (Weights are updated in 'model' object)  
tuned\_engine \= ModelWrapper()   
tuned\_agent \= build\_financial\_agent(tuned\_engine)

\# 2\. Run Tuned Evaluation  
tuned\_results \= \[\]  
print("--- Running Tuned Agent Eval \---")

\# Define Judge Metric  
from mlflow.metrics.genai import make\_genai\_metric, EvaluationExample  
reasoning\_metric \= make\_genai\_metric(  
    name="reasoning\_quality",  
    definition="Grade the clarity of the \<think\> trace and the correctness of the Python code strategy.",  
    grading\_prompt="Score 1-5.",  
    model="google\_genai:/gemini-2.0-flash",  
    parameters={"temperature": 0.0}  
)

with mlflow.start\_run(run\_name="Tuned\_Model\_Eval") as run:  
    mlflow.models.set\_model(tuned\_agent) \# Trace the Tuned Agent  
      
    for i, row in validation\_sample.iterrows():  
        \# Invoke Agent  
        res \= tuned\_agent.invoke({"query": row\['prompt'\]\[1\]\['content'\], "context": ""})  
        tuned\_results.append(res\['final\_answer'\])  
      
    validation\_sample\['tuned\_agent\_answer'\] \= tuned\_results  
      
    \# 3\. Log Comparison Table  
    mlflow.log\_table(data=validation\_sample, artifact\_file="agent\_comparison.json")  
      
    \# 4\. Evaluate with LLM Judge  
    mlflow.evaluate(  
        data=validation\_sample,  
        targets="answer",  
        predictions="tuned\_agent\_answer",  
        model\_type="text",  
        extra\_metrics=\[reasoning\_metric\]  
    )

print("Comparison Complete. View 'agent\_comparison.json' in MLflow artifacts.")

## **4\. Execution Requirements**

* **Hardware:** DGX Spark / GB10 GPU / CUDA 13\.  
* **Secrets:** .env must contain HF\_TOKEN and GOOGLE\_API\_KEY.  
* **File Management:** Understand and leverage current project structure, make sure all settings are defined in the initial settings section and all paths are relative to the project root (not absolute) 