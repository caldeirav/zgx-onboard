# **Design Document: AML Investigation Agent (FunctionGemma \+ GRPO)**

## **1\. Executive Summary**

### **1.1 Objective**

Develop an autonomous AI Agent capable of investigating financial graph structures to identify money laundering patterns (specifically "layering" and "structuring"). The agent will navigate a transaction network (IBM AMLworld), starting from a suspicious entity, tracing funds through intermediaries to identify if they reach a high-risk or sanctioned destination.

### **1.2 Tech Stack**

* **Model:** FunctionGemma (2B/7B) via Unsloth (LoRA/QLoRA).  
* **Data Processing:** **Polars** (High-performance transaction joins and temporal filtering).  
* **Data Sourcing:** **Kaggle API/SDK** (Programmatic dataset retrieval).  
* **Orchestration:** LangGraph (Stateful graph exploration).  
* **Reinforcement Learning:** TRL (**GRPO** \- Group Relative Policy Optimization).  
* **Observability & Eval:** MLflow Agent Tracing \+ LLM-as-Judge.

## **2\. Data Architecture & Environment**

### **2.1 Kaggle SDK & Data Ingestion**

The project automates data retrieval using the Kaggle Python SDK.

* **Authentication:** Uses os.environ for KAGGLE\_USERNAME and KAGGLE\_KEY.  
* **Dataset:** [IBM Transactions for Anti Money Laundering (AML)](https://www.kaggle.com/datasets/ealtman2019/ibm-transactions-for-anti-money-laundering-aml).  
* **Polars Processing:** Used to load LI-Small\_Trans.csv. Polars identifies "Structuring" flags and prepares edge lists for the graph environment significantly faster than Pandas.

### **2.2 FinancialEnvironment (Graph Class)**

The agent interacts with a FinancialEnvironment class that wraps the IBM AML data in a NetworkX graph.

* **Nodes:** Accounts/Entities. Features: Risk\_Score, Country, Account\_Type.  
* **Edges:** Transactions. Features: Amount, Timestamp, Currency, Is\_Laundering (hidden from agent).  
* **Sanctions Overlay:** Destinations of laundering chains are treated as "Sanctioned Entities" (the Win condition).

### **2.3 Tool Definitions (Instrumented with MLflow)**

1. **get\_account\_summary(account\_id)**: Returns metadata (e.g., Jurisdiction: High Risk).  
2. **get\_recent\_transactions(account\_id, direction)**: Returns list of targets. Constraint: Returns top 5 most recent to force prioritization.  
3. **check\_sanctions\_list(entity\_id)**: Returns True if on watchlist (Win Condition).  
4. **submit\_sar(entity\_id, reason)**: Terminal action to end the episode.

## **3\. Training Strategy**

### **3.1 Stage 1: Supervised Fine-Tuning (SFT)**

* **Dataset:** 50+ trajectories derived from LI-Small\_Patterns.txt.  
* **Objective:** Teach the model the XML syntax for tool calling and basic investigation flow (e.g., "If large outflow, check receiver").

### **3.2 Stage 2: GRPO Reinforcement Learning**

* **Objective:** Optimize investigative strategy—learning *which* tools to call to find fraud fastest.  
* **Group Size:** 8 trajectories per prompt.  
* **Reward Function:**  
  * **Success (+2.0):** Correctly identifying the sanctioned entity match.  
  * **Efficiency (+0.1 per step saved):** Finding the node in fewer steps.  
  * **Strategy Bonus:** Reward for checking transaction details before moving to the next hop.  
  * **Negative Rewards:** Penalties for false positives, syntax errors, or infinite loops.

## **4\. Evaluation Framework: LLM-as-Judge**

### **4.1 Judge Scope & Criteria**

The Judge LLM evaluates MLflow Traces based strictly on:

1. **Decision Quality:** Logic of tool calls given the evidence.  
2. **Strategy Efficacy:** Application of sound investigation (e.g., following the "layering" stage).  
3. **Outcome Accuracy:** Final submission match to ground truth.

### **4.2 MLflow Integration**

Each step is captured via @mlflow.trace. At episode end, the full trace is passed to the Judge. Scores (1-5) and reasoning are logged as MLflow artifacts.

## **5\. Notebook Implementation Structure**

### **Section 1: Setup & Data Preparation**

* Install libs (unsloth, trl, polars, mlflow).  
* Kaggle SDK download and Polars graph initialization.

### **Section 2: MLflow Configuration**

* Initialize experiment and define trace decorators.

### **Section 3: The Agent Loop**

* Define Agent class with step() function (decorated with @mlflow.trace).  
* Implement FunctionGemma XML parsing logic.

### **Section 4: RL Training (GRPO)**

* Define GRPOConfig and custom reward functions.  
* Execute training loop and save RL-tuned adapter.

### **Section 5: Final Evaluation & Tracing**

* Compare Baseline vs. RL Win Rates.  
* Retrieve and visualize traces using mlflow.search\_traces().

## **6\. Development Prompts for Coding Agent**

* **Prompt 1 (Data & Tools):** "Create the FinancialEnvironment class using Polars for data loading and NetworkX for the graph. Implement the 4 tools instrumented with @mlflow.trace(span\_type='TOOL')."  
* **Prompt 2 (SFT):** "Fine-tune gemma-2-2b-it using Unsloth on 50 investigation paths using the function-gemma XML tags."  
* **Prompt 3 (RL):** "Implement the TRL training loop using GRPO. Reward correct submit\_sar calls and penalize inefficient traversal."  
* **Prompt 4 (Eval):** "Generate evaluation code that runs 10 episodes, captures full traces in MLflow, and uses an LLM-as-Judge to score the investigative strategy."