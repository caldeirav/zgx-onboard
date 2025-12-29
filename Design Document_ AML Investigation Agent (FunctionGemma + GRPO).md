# **Design Document: AML Investigation Agent (FunctionGemma + GRPO)**

## **1. Executive Summary**

### **1.1 Objective**

Develop an autonomous AI Agent capable of investigating financial graph structures to identify money laundering patterns. The agent navigates the IBM AMLworld transaction network to identify sanctioned destinations through multi-hop tracing.

### **1.2 Tech Stack**

| Component | Technology | Description |
|-----------|------------|-------------|
| **Base Model** | FunctionGemma (Gemma-2-2B-IT) | XML-style function calling format |
| **Fine-Tuning** | Unsloth + LoRA/QLoRA | 4-bit quantization, 2x faster training |
| **Data Processing** | Polars | High-performance dataframes |
| **Graph Analysis** | NetworkX | Transaction network traversal |
| **Data Sourcing** | Kaggle SDK | Automated dataset download |
| **Agent Framework** | LangGraph + MemorySaver | Stateful exploration with checkpointing |
| **RL Training** | TRL GRPOTrainer | Group Relative Policy Optimization |
| **Observability** | MLflow Tracing | Full agent trace logging |
| **Evaluation** | Google GenAI (Gemini) | LLM-as-Judge for strategy scoring |

### **1.3 Implementation Reference**

Primary implementation: `notebooks/agents/aml_investigation_agent_v2.ipynb`

## **2. Investigative Environment (FinancialEnvironment)**

### **2.1 Graph & Data Processing**

**Data Source:** [IBM Transactions for Anti Money Laundering (AML)](https://www.kaggle.com/datasets/ealtman2019/ibm-transactions-for-anti-money-laundering-aml/data)

The notebook automatically downloads the dataset using Kaggle SDK if not present:

```python
from kaggle.api.kaggle_api_extended import KaggleApi
api = KaggleApi()
api.authenticate()
api.dataset_download_files(
    dataset="ealtman2019/ibm-transactions-for-anti-money-laundering-aml",
    path=str(DATA_DIR),
    unzip=True
)
```

**Processing Pipeline:**
* **Polars Engine:** Loads transactions and accounts with high-performance dataframes
* **Account ID Generation:** Creates unique identifiers as `{bank_id}-{account_number}`
* **Sanctioned Flagging:** 30% of laundering destination accounts marked as sanctioned
* **Pattern Parsing:** Extracts laundering patterns from `LI-Small_Patterns.txt` for training data

**State Memory (InvestigationState TypedDict):**
```python
class InvestigationState(TypedDict):
    start_account: str
    accounts_analyzed: Dict[str, dict]
    entities_checked: Dict[str, bool]
    risk_indicators: List[str]
    investigation_path: List[str]
    total_amount_traced: float
    current_strategy: str  # explore/follow_risk/submit_report
    messages: List[dict]
    step_count: int
    terminated: bool
    success: bool
```

**Win Condition:** `submit_sar` on an entity that is both sanctioned AND reachable via a laundering path from the seed account.

### **2.2 Toolset (MLflow Instrumented)**

All tools are decorated with `@mlflow.trace(span_type="TOOL")` for observability:

| Tool | Description | Key Returns |
|------|-------------|-------------|
| `get_account_summary` | Metadata and risk assessment | `risk_score`, `is_sanctioned`, `transitive_illicit` |
| `get_recent_transactions` | Top-5 recent flows (sorted by amount) | `counterparty`, `amount`, `high_risk_indicator` |
| `check_sanctions_list` | OFAC watchlist verification | `on_sanctions_list`, `list_type` |
| `submit_sar` | Terminal action with path validation | `correct_identification`, `evaluation_reason` |

### **2.3 Transitive Illicit Computation**

The environment pre-computes all nodes reachable via laundering transaction edges using BFS:

```python
def _compute_transitive_illicit(self):
    """Compute all nodes reachable via laundering transaction edges."""
    for source in laundering_sources:
        # BFS following only is_laundering=1 edges
        # Mark all reachable nodes as transitive_illicit
```

## **3. FunctionGemma & Unsloth Implementation**

### **3.1 Unsloth Model Loading & Optimization**

```python
from unsloth import FastLanguageModel

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="google/gemma-2-2b-it",
    max_seq_length=8192,        # Extended for multi-hop conversations
    dtype=None,                  # Auto-detect best dtype
    load_in_4bit=True,           # 4-bit quantization for VRAM efficiency
)

# Switch to inference mode for optimized kernels
FastLanguageModel.for_inference(model)
```

### **3.2 Internal Reasoning with `<thinking>` Tags**

**Key Implementation Change:** FunctionGemma does not support thinking by default. We implement explicit internal reasoning using `<thinking>` tags before each function call.

**Response Format:**
```xml
<thinking>
[Agent's reasoning about what to do next and why]
</thinking>
<start_function_call>call:function_name{param: value}</start_function_call>
```

**Example:**
```xml
<thinking>
The account shows transitive_illicit=True which indicates it's on a laundering path.
I should now trace outgoing transactions to follow the money flow and identify suspicious counterparties.
</thinking>
<start_function_call>call:get_recent_transactions{account_id: 123-ABC, direction: outgoing}</start_function_call>
```

This is implemented via:
- `extract_thinking()` function to parse reasoning from model output
- SFT training data includes `<thinking>` tags in all assistant responses
- Investigation prompt explicitly requires reasoning before action

### **3.3 Tool Declaration Format**

```xml
<start_function_declaration>
{"name": "get_account_summary", 
 "description": "Get account metadata and risk assessment for an account ID",
 "parameters": {"type": "object", "properties": {"account_id": {"type": "string"}}, "required": ["account_id"]}}
</start_function_declaration>
```

### **3.4 Tool Call Hardening**

To handle model hallucinations, implement fuzzy tool name correction:

```python
TOOL_NAME_CORRECTIONS = {
    "get_account": "get_account_summary",
    "get_transactions": "get_recent_transactions",
    "check_sanctions": "check_sanctions_list",
    "submit_report": "submit_sar",
}
```

### **3.5 Unsloth LoRA Configuration**

```python
model_for_training = FastLanguageModel.get_peft_model(
    model,
    r=32,                              # Higher rank for complex reasoning
    lora_alpha=64,
    target_modules=[
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj"
    ],
    lora_dropout=0.05,
    use_gradient_checkpointing="unsloth",
)
```

## **4. Training & Optimization Strategy**

### **4.1 Three-Stage Evaluation Pipeline**

The notebook implements a complete evaluation pipeline comparing performance across training stages:

| Stage | Description | Metrics Captured |
|-------|-------------|------------------|
| **Baseline** | Pre-training (base Gemma-2-2B-IT) | Success rate, steps, reward, LLM scores |
| **Post-SFT** | After Supervised Fine-Tuning | Success rate, steps, reward, LLM scores |
| **Post-GRPO** | After Reinforcement Learning | Success rate, steps, reward, LLM scores |

### **4.2 SFT: Persistence & Backtracking**

**Data Source:** `LI-Small_Patterns.txt` parsed into `LaunderingPattern` dataclass

**Training Data Generation:**
```python
def generate_sft_sample(pattern: LaunderingPattern, include_backtrack: bool = False):
    # Generate multi-turn conversation with <thinking> tags
    # Include backtracking scenarios for ~30% of samples
```

**Backtracking Example:**
```xml
<thinking>
This entity is NOT on the sanctions list. I cannot submit a SAR here.
I need to BACKTRACK and continue following the transaction trail from {wrong_account}
to find other suspicious counterparties that might be sanctioned.
</thinking>
<start_function_call>call:get_recent_transactions{account_id: {wrong_account}, direction: outgoing}</start_function_call>
```

### **4.3 GRPO: Strategy-Based Reward Shaping**

```python
def calculate_reward(tool_name, args, result, state) -> float:
    reward = -0.1  # R_Efficiency: Base step penalty
    
    if tool_name == "get_account_summary":
        if result.get("transitive_illicit"):
            reward += 0.5  # R_Discovery
    
    elif tool_name == "get_recent_transactions":
        if any("HIGH_RISK" in r for r in state['risk_indicators']):
            reward += 0.3  # R_Logic
        for txn in result:
            if txn.get("high_risk_indicator"):
                reward += 0.2  # Bonus for finding illicit
    
    elif tool_name == "check_sanctions_list":
        if result.get("on_sanctions_list"):
            reward += 0.5
    
    elif tool_name == "submit_sar":
        if result.get("correct_identification"):
            reward += 2.0  # R_Outcome
        else:
            reward -= 1.0  # Penalty for wrong SAR
    
    return reward
```

| Reward | Value | Trigger |
|--------|-------|---------|
| R_Discovery | +0.5 | Discovering nodes with `transitive_illicit` tag |
| R_Logic | +0.3 | Calling `get_recent_transactions` after identifying high-risk |
| R_Outcome | +2.0 | Correct SAR submission |
| R_Efficiency | -0.1 | Per-step penalty to prevent loops |
| R_Wrong_SAR | -1.0 | Incorrect SAR submission |

## **5. Evaluation: LLM-as-Judge**

### **5.1 Google GenAI Library**

**Important:** The `google-generativeai` library is deprecated. Use `google-genai` instead:

```python
from google import genai

gemini_client = genai.Client(api_key=GOOGLE_API_KEY)
response = gemini_client.models.generate_content(
    model="gemini-2.0-flash",
    contents=prompt
)
```

### **5.2 Evaluation Rubric**

```python
JUDGE_PROMPT = """
## Evaluation Rubric

1. **Strategy Quality** (0-10): Did the agent prioritize high-value and high-risk transfers?
2. **Decision Persistence** (0-10): Did it pivot correctly after hitting dead ends?
3. **Outcome Quality** (0-10): Did it correctly identify a sanctioned entity?

## Response Format
{
    "strategy_score": <0-10>,
    "persistence_score": <0-10>,
    "outcome_score": <0-10>,
    "overall_score": <0-10>,
    "reasoning": "<brief explanation>",
    "strengths": ["..."],
    "weaknesses": ["..."]
}
"""
```

### **5.3 Comparison Visualization**

The notebook generates comparison charts saved to `outputs/training_comparison.png`:
- Success Rate (%) across stages
- Average Reward across stages
- Average Steps (efficiency) across stages
- LLM-as-Judge Overall Score across stages

## **6. Project Structure & Paths**

### **6.1 Directory Structure**

```
project_root/
├── data/
│   └── raw/                    # IBM AML dataset (auto-downloaded)
│       ├── LI-Small_Trans.csv
│       ├── LI-Small_accounts.csv
│       └── LI-Small_Patterns.txt
├── models/                     # Training outputs
│   ├── sft_output/            # SFT training checkpoints
│   ├── sft_adapter/           # SFT LoRA adapter
│   ├── grpo_output/           # GRPO training checkpoints
│   ├── grpo_adapter/          # GRPO LoRA adapter
│   └── aml_agent_final/       # Final trained model
├── outputs/                    # Evaluation outputs
│   ├── training_comparison.png
│   └── training_comparison.csv
└── notebooks/
    └── agents/
        └── aml_investigation_agent_v2.ipynb
```

### **6.2 Path Configuration**

All paths are relative to the project root:

```python
NOTEBOOK_DIR = Path(".").resolve()
PROJECT_ROOT = NOTEBOOK_DIR.parent.parent  # Go up from notebooks/agents
DATA_DIR = PROJECT_ROOT / "data" / "raw"
MODELS_DIR = PROJECT_ROOT / "models"
OUTPUT_DIR = PROJECT_ROOT / "outputs"
```

### **6.3 Cleanup for Re-runs**

Before baseline evaluation, the notebook cleans up previous training outputs:

```python
def cleanup_training_outputs():
    cleanup_dirs = [
        MODELS_DIR / "sft_output",
        MODELS_DIR / "sft_adapter", 
        MODELS_DIR / "grpo_output",
        MODELS_DIR / "grpo_adapter",
        MODELS_DIR / "aml_agent_final",
    ]
    for dir_path in cleanup_dirs:
        if dir_path.exists():
            shutil.rmtree(dir_path)
```

## **7. Configuration Parameters**

```python
# Model Configuration
MODEL_NAME = "google/gemma-2-2b-it"
GEMINI_MODEL = "gemini-2.0-flash"

# Agent Configuration
MAX_STEPS = 25
MAX_HISTORY_TURNS = 6

# Training Configuration
SFT_EPOCHS = 3
SFT_LEARNING_RATE = 2e-4
GRPO_EPOCHS = 1
GRPO_LEARNING_RATE = 5e-6

# LoRA Configuration
LORA_R = 32
LORA_ALPHA = 64
LORA_TARGET_MODULES = [
    "q_proj", "k_proj", "v_proj", "o_proj",
    "gate_proj", "up_proj", "down_proj"
]

# Evaluation
EVAL_EPISODES = 10
```

## **8. Implementation Fixes Applied**

### **8.1 Resolved Bottlenecks**

| Issue | Solution Implemented |
|-------|---------------------|
| Brittle Parsing | `harden_tool_call()` with fuzzy name matching |
| Lack of Reasoning | Explicit `<thinking>` tags in prompt and training data |
| Path Hardcoding | Relative paths from `PROJECT_ROOT` |
| Re-run Artifacts | `cleanup_training_outputs()` before evaluation |
| Deprecated API | Migrated from `google-generativeai` to `google-genai` |

### **8.2 Key Implementation References**

- [FunctionGemma Unsloth Notebook](https://colab.research.google.com/github/unslothai/notebooks/blob/main/nb/FunctionGemma_(270M).ipynb) - Internal reasoning pattern
- [Google GenAI SDK Documentation](https://ai.google.dev/gemini-api/docs/libraries) - LLM-as-Judge API
- [TRL GRPO Documentation](https://huggingface.co/docs/trl/grpo_trainer) - Reinforcement learning

## **9. Running the Notebook**

### **9.1 Prerequisites**

```bash
pip install unsloth polars networkx mlflow trl transformers kaggle google-genai matplotlib
```

### **9.2 Environment Variables**

```bash
export GOOGLE_API_KEY="your-gemini-api-key"
export KAGGLE_USERNAME="your-kaggle-username"
export KAGGLE_KEY="your-kaggle-key"
```

### **9.3 Execution Flow**

1. **Setup & Configuration** - Load libraries, set paths
2. **Data Loading** - Download from Kaggle (if needed), process with Polars
3. **Environment Build** - Create NetworkX graph, compute transitive illicit
4. **Model Loading** - Load Gemma-2-2B-IT with Unsloth
5. **Cleanup** - Remove previous training artifacts
6. **Baseline Evaluation** - Test pre-training performance
7. **SFT Training** - Fine-tune on pattern data with backtracking
8. **Post-SFT Evaluation** - Measure improvement
9. **GRPO Training** - Reinforcement learning with reward shaping
10. **Post-GRPO Evaluation** - Final performance measurement
11. **Comparison** - Generate charts and metrics table
12. **Save Model** - Export final adapter to `/models`
