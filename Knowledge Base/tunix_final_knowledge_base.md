# TuNix: Comprehensive Knowledge Base

## Table of Contents
1. [Introduction](#introduction)
2. [Installation](#installation)
3. [SFT API Reference](#sft-api-reference)
4. [RL API Reference](#rl-api-reference)
5. [Distillation API Reference](#distillation-api-reference)
6. [Generation API Reference](#generation-api-reference)
7. [Practical Examples](#practical-examples)

---

## Introduction

**TuNix** (Tune-in-JAX) is a JAX-based library designed to streamline the post-training of Large Language Models. It provides efficient and scalable support for:

- **Supervised Fine-Tuning (SFT)**: Full weights and parameter-efficient fine-tuning with LoRA/QLoRA
- **Reinforcement Learning (RL)**: PPO, GRPO, and GSPO-token algorithms
- **Preference Fine-Tuning**: DPO and ORPO alignment methods
- **Knowledge Distillation**: Logit, attention transfer, and feature pooling strategies

TuNix leverages JAX for accelerated computation and integrates seamlessly with the **Flax NNX** framework.

### Key Features

| Feature | Description |
|---------|-------------|
| **Full Weights Fine-Tuning** | Train all model parameters for maximum flexibility |
| **PEFT with LoRA/QLoRA** | Parameter-efficient fine-tuning reducing memory by 50%+ |
| **PPO** | Proximal Policy Optimization with actor-critic architecture |
| **GRPO** | Group Relative Policy Optimization eliminating separate value models |
| **GSPO-token** | Token-level Group Sequence Policy Optimization |
| **DPO** | Direct Preference Optimization without reward models |
| **ORPO** | Odds Ratio Preference Optimization (50% more memory efficient) |
| **Logit Distillation** | Student learns teacher's output probability distribution |
| **Attention Transfer** | Align attention mechanisms between student and teacher |
| **Feature Pooling** | Match intermediate representations across different architectures |
| **Model Sharding** | Native support for DP, FSDP, and TP strategies |
| **TPU Optimization** | Designed for distributed training on accelerators |

### Current Status

TuNix is in **early development**. Upcoming features include:
- Agentic RL Training (async rollout, multi-turn support, tool usage)
- Additional state-of-the-art RL and distillation algorithms
- Multi-host distributed training
- Optimized rollout with vLLM or SGLang-Jax

---

## Installation

### Method 1: PyPI (Recommended)

```bash
pip install "google-tunix[prod]"
```

### Method 2: GitHub (Latest Main Branch)

```bash
pip install git+https://github.com/google/tunix
```

### Method 3: Editable Install (Development)

```bash
git clone https://github.com/google/tunix.git
cd tunix
pip install -e ".[dev]"
```

For TPU inference with vLLM:
```bash
pip install tpu-inference
```

Or use the Docker image: `vllm/vllm-tpu:v0.11.1`

### Installing with SGLang-Jax Rollout

```bash
# First install tunix using any method above, then:
git clone git@github.com:sgl-project/sglang-jax.git
cd sglang-jax/python
pip install -e .
```

---

## SFT API Reference

The Supervised Fine-Tuning module provides trainers, configurations, and utilities for fine-tuning LLMs.

# tunix.PeftTrainer

The `tunix.PeftTrainer` class, located at `tunix/tunix/sft/peft_trainer.py`, is the central orchestrator for executing Supervised Fine-Tuning (SFT) and Parameter-Efficient Fine-Tuning (PEFT) training loops within the Tunix framework. It manages the entire training lifecycle, encompassing model optimization, data handling, checkpointing, metrics logging, and performance tracing. Its design prioritizes flexibility, allowing extensive customization through configurable parameters, methods, and integration points like hooks.

## Constructor

### `__init__(self, model: nnx.Module, optimizer: optax.GradientTransformation, training_config: TrainingConfig, metrics_logger: Optional[MetricsLogger] = None, perf_tracer: Optional[perf_trace.Tracer] = None)`

**Description**: Initializes a new instance of the `PeftTrainer`. It sets up the core components for the training process.

**Parameters:**
- `model`: The `flax.nnx.Module` instance to be fine-tuned.
- `optimizer`: An `optax.GradientTransformation` defining the optimization algorithm. This will be wrapped by `optax.MultiSteps` if `training_config.gradient_accumulation_steps` is set, and then used to create an `nnx.Optimizer`.
- `training_config`: An instance of `tunix.TrainingConfig` containing all the hyperparameters and settings for the training run.
- `metrics_logger`: An optional `tunix.sft.metrics_logger.MetricsLogger` instance. If `None`, one is created based on `training_config.metrics_logging_options`.
- `perf_tracer`: An optional `tunix.perf.trace.Tracer` for performance tracing. If `None`, a `tunix.perf.trace.NoopTracer` is used.

**Internal Setup:**
- Initializes internal counters `_train_steps` and `_iter_steps` to 0.
- Sets up default `loss_fn` and `eval_loss_fn` to `_default_loss_fn` and `gen_model_input_fn` to a passthrough lambda.
- Creates a `tunix.sft.checkpoint_manager.CheckpointManager` instance based on `training_config.checkpoint_root_directory` and `training_config.checkpointing_options`.
- Restores the trainer state (model parameters, optimizer state, `_train_steps`, `_iter_steps`, and custom metadata) if a checkpoint is found.
- Initializes `tunix.sft.inflight_throttler.InflightThrottler` to manage computational load using `training_config.max_inflight_computations`.
- Sets up default `training_hooks` and `data_hooks` (instances of `tunix.sft.hooks.TrainingHooks` and `tunix.sft.hooks.DataHooks` respectively).

## Methods

### `with_training_hooks(self, training_hooks: hooks.TrainingHooks) -> "PeftTrainer"`

**Description**: Sets a custom `tunix.sft.hooks.TrainingHooks` object for the trainer. This allows users to inject custom logic at various points in the training lifecycle. Returns `self` for fluent API chaining.

### `with_data_hooks(self, data_hooks: hooks.DataHooks) -> "PeftTrainer"`

**Description**: Sets a custom `tunix.sft.hooks.DataHooks` object for the trainer. This enables users to inject custom data loading and preprocessing logic. Returns `self` for fluent API chaining.

### `clear_jit_cache(self) -> None`

**Description**: Clears the internal JAX JIT cache for the `_train_step` and `_eval_step` functions. This is crucial when the `loss_fn` or `gen_model_input_fn` (or other components of the compiled functions) are dynamically changed during the trainer's lifetime to ensure JAX recompiles with the new logic.

### `with_loss_fn(self, loss_fn: Callable[Concatenate[nnx.Module, P], ArrayLike | Tuple[ArrayLike, Any]], has_aux: bool = False) -> "PeftTrainer"`

**Description**: Allows the user to provide a custom loss function.

**Parameters:**
- `loss_fn`: A callable that accepts the `nnx.Module` model and preprocessed batch data, returning the loss value.
- `has_aux`: A boolean indicating if the `loss_fn` returns auxiliary outputs (e.g., additional metrics) alongside the primary loss.

**Behavior**: Automatically calls `clear_jit_cache()` and sets the internal `loss_fn` and `eval_loss_fn`. Returns `self` for fluent API chaining.

### `with_gen_model_input_fn(self, gen_model_input_fn: Callable[[Any], _ModelInputT]) -> "PeftTrainer"`

**Description**: Sets a custom function (`gen_model_input_fn`) that transforms raw input data from the dataset iterator into the specific dictionary format expected by the model's forward pass and the `loss_fn`.

**Behavior**: Automatically calls `clear_jit_cache()`. Returns `self` for fluent API chaining.

### `_train_step(self, model: nnx.Module, optimizer: nnx.Optimizer, batch: Any, is_train_step: bool)`

**Description**: An internal, non-JIT-compiled method defining one pass of the training step.

**Process:**
- Generates model inputs using `gen_model_input_fn`.
- Computes loss and gradients using `nnx.value_and_grad` (or `nnx.value_and_grad_xs` if `data_sharding_axis` is used) based on `loss_fn`.
- Applies the computed gradients to update `model` parameters via `optimizer`.
- Handles gradient accumulation if `training_config.gradient_accumulation_steps` is set.
- Buffers metrics (loss, learning rate, auxiliary outputs) and calls `training_hooks.on_train_step_end` and `_post_process_train_step`.

**Returns**: A tuple: `(loss, auxiliary_data, updated_model_state, updated_optimizer_state)`.

### `_eval_step(self, model: nnx.Module, batch: Any, is_train_step: bool)`

**Description**: An internal, non-JIT-compiled method defining one pass of the evaluation step.

**Process:**
- Generates model inputs using `gen_model_input_fn`.
- Computes loss using `eval_loss_fn`.
- **Crucially, it does NOT compute gradients or update model parameters.**
- Buffers metrics and calls `training_hooks.on_eval_step_end` and `_post_process_eval_step`.

**Returns**: A tuple: `(loss, auxiliary_data)`.

### `create_train_step_fn(self) -> Callable[..., Any]`

**Description**: Returns the underlying function responsible for a single training step. This function will be JIT-compiled by `jit_train_and_eval_step`. It wraps `_train_step` with sharding context if `data_sharding_axis` is configured.

### `create_eval_step_fn(self) -> Callable[..., Any]`

**Description**: Returns the underlying function responsible for a single evaluation step. This function will be JIT-compiled by `jit_train_and_eval_step`. It wraps `_eval_step` with sharding context if `data_sharding_axis` is configured.

### `_shard_optimizer(self, optimizer_state: nnx.State) -> nnx.State`

**Description**: An internal method responsible for sharding the optimizer's state if a JAX mesh and sharding axes are configured for the optimizer's parameters. This ensures that the optimizer state also adheres to distributed training paradigms.

### `jit_train_and_eval_step(self, skip_jit: bool = False, *, cache_nnx_graph: bool = False) -> Tuple[Callable[..., Any], Callable[..., Any]]`

**Description**: Compiles the `_train_step` and `_eval_step` functions using JAX's JIT compiler to optimize performance.

**Parameters:**
- `skip_jit`: If `True`, returns the unjitted versions of the step functions.
- `cache_nnx_graph`: If `True`, enables caching of NNX graph traversals, which can improve performance but requires the model/optimizer graph to remain static.

**Process**: Handles the initial sharding of the optimizer state using `_shard_optimizer`.

**Returns**: A tuple: `(jitted_train_step_fn, jitted_eval_step_fn)`.

### `_prepare_inputs(self, input_data: Any) -> Any`

**Description**: An internal, *overridable* method that allows for additional preprocessing of raw input data before it's passed to the model and sharded. Subclasses (e.g., `tunix.distillation.distillation_trainer.DistillationTrainer` or `tunix.sft.dpo.dpo_trainer.DPOTrainer`) often override this to handle specific input requirements or transformations.

### `_post_process_train_step(self, aux: Any) -> None`

**Description**: An internal, *overridable* method executed after each training step. Its purpose is to process and buffer any `auxiliary data` returned by the `loss_fn`, typically for additional metrics logging beyond the basic loss.

### `_post_process_eval_step(self, aux: Any) -> None`

**Description**: Similar to `_post_process_train_step`, this internal, *overridable* method is executed after each evaluation step to process and buffer auxiliary data from the `eval_loss_fn`.

### `_try_get_learning_rate(self) -> float | None`

**Description**: An internal utility method that attempts to extract the current learning rate from the `optimizer` state. It's used for logging the learning rate as a metric.

### `_log_metrics(self, mode: metrics_logger.Mode, global_step: int) -> None`

**Description**: An internal method responsible for instructing the `metrics_logger` to write all buffered metrics for the given `mode` (e.g., `train`, `eval`) at the specified `global_step`.

### `_buffer_metrics(self, metrics: dict[str, Any], mode: metrics_logger.Mode) -> None`

**Description**: An internal method that adds the provided `metrics` (a dictionary) to the `metrics_logger`'s internal buffer for the specified `mode`. These buffered metrics are later written to the log when `_log_metrics` is called.

### `_write_train_metrics(self, train_loss: float, aux: Any, global_step: int) -> None`

**Description**: An internal method specific to handling and logging training metrics. It buffers the `train_loss`, auxiliary data, and other relevant information (like learning rate) and then calls `_log_metrics`.

### `_write_metrics(self, mode: metrics_logger.Mode, loss: float, aux: Any, global_step: int) -> None`

**Description**: A generalized internal method for buffering and logging metrics (`loss` and `aux`) for a given `mode` (`train` or `eval`) at a specific `global_step`.

### `_to_np_array(self, x: Any) -> np.ndarray | Any`

**Description**: An internal helper method to convert a JAX `Array` or `DeviceArray` to a NumPy `ndarray`. If the input is not a JAX array, it's returned unchanged. This is often used before logging to ensure compatibility with various logging backends.

### `_switch_mode(self, mode: metrics_logger.Mode) -> None`

**Description**: An internal method to inform the `metrics_logger` about a change in the operational `mode` (e.g., switching from `train` to `eval`). This helps the logger organize metrics correctly.

### `_tqdm_train_metrics(self) -> dict[str, Any]` (Property)

**Description**: A property that returns a dictionary of metrics to be displayed in the `tqdm` progress bar during the training loop. It includes the current training loss, current learning rate, and any other relevant metrics from the `metrics_logger`.

### `_may_update_pbar(self, pbar: Any) -> None`

**Description**: An internal method responsible for updating the `tqdm` progress bar with current training status and metrics, potentially including custom `pbar_description`.

### `train(self, train_ds: Iterable[Any], eval_ds: Iterable[Any] | None = None, skip_jit: bool = False, *, cache_nnx_graph: bool = False) -> None`

**Description**: The main entry point for starting the training loop.

**Parameters:**
- `train_ds`: An iterable yielding batches of training data.
- `eval_ds`: An optional iterable yielding batches of evaluation data.
- `skip_jit`: If `True`, avoids JIT compilation of step functions.
- `cache_nnx_graph`: If `True`, enables NNX graph caching for performance.

**Orchestration**: This method orchestrates the entire fine-tuning process:
- Calls `training_hooks.on_train_begin`.
- Initializes/restores JIT-compiled functions for training and evaluation (`jit_train_and_eval_step`).
- Iterates through `train_ds`, performing training steps, managing `_train_steps` and `_iter_steps` counters.
- Periodically runs evaluations on `eval_ds` based on `training_config.eval_every_n_steps`.
- Manages checkpointing (saving and restoring) via `checkpoint_manager`.
- Handles metrics logging and updates the progress bar.
- Manages `max_inflight_computations` using the `inflight_throttler`.
- Calls hook methods: `on_train_step_begin`, `on_train_step_end`, `on_eval_begin`, `on_eval_end`.
- Calls `training_hooks.on_train_end` at the end.

### `_save_last_checkpoint(self, global_step: int) -> None`

**Description**: An internal method called at the end of training (or upon closing) to save the final checkpoint. It includes the current model state, optimizer state, and any custom metadata.

### `train_steps(self) -> int` (Property)

**Description**: A property that returns the current number of completed training steps (i.e., the number of times the model's parameters have been updated). This is equivalent to `self._train_steps`.

### `iter_steps(self) -> int` (Property)

**Description**: A property that returns the current number of completed data iterator steps (i.e., the total number of batches or micro-batches that have passed through the `gen_model_input_fn`). This is equivalent to `self._iter_steps`.

### `custom_checkpoint_metadata(self) -> dict[str, Any]`

**Description**: An *overridable* method that should return a dictionary of custom metadata to be saved alongside the model parameters and optimizer state in checkpoints. By default, it returns an empty dictionary. Subclasses can override this to store additional context like `global_step`.

### `close(self) -> None`

**Description**: A cleanup method that should be called at the end of the training process. It ensures that all resources are properly released, including:
- Flushing any buffered metrics to their respective logging backends.
- Saving the final checkpoint by calling `_save_last_checkpoint`.
- Closing the `checkpoint_manager`.
- Closing the `metrics_logger`.

### `_run_eval(self, eval_ds: Iterable[Any], eval_step_fn: Callable[..., Any]) -> None`

**Description**: An internal method used by the `train` method to execute the evaluation loop over the provided `eval_ds`. It iterates through the evaluation data, performs `eval_step_fn` for each batch, and logs the results using `_write_metrics`.

---

### Class: `tunix.TrainingConfig`

```python
from __future__ import annotations

import dataclasses
from typing import Any, Tuple

import orbax.checkpoint as ocp


@dataclasses.dataclass(slots=True, kw_only=True)
class TrainingConfig:
    """
    Configuration for a training run.

    This dataclass centralizes all parameters required to control training
    behavior, checkpointing, logging, profiling, sharding, and user-facing
    progress reporting.
    """

    # ---------------------------------------------------------------------
    # Core training behavior
    # ---------------------------------------------------------------------
    eval_every_n_steps: int = 100
    max_steps: int | None = None
    gradient_accumulation_steps: int | None = None

    # ---------------------------------------------------------------------
    # Checkpointing
    # ---------------------------------------------------------------------
    checkpoint_root_directory: str | None = None
    checkpointing_options: ocp.CheckpointManagerOptions | None = None

    # ---------------------------------------------------------------------
    # Logging and profiling
    # ---------------------------------------------------------------------
    metrics_logging_options: MetricsLoggerOptions = dataclasses.field(
        default_factory=MetricsLoggerOptions
    )
    profiler_options: ProfilerOptions | None = None

    # ---------------------------------------------------------------------
    # Sharding and resource management
    # ---------------------------------------------------------------------
    data_sharding_axis: Tuple[str, ...] = ("fsdp",)
    max_inflight_computations: int = 2

    # ---------------------------------------------------------------------
    # UI / UX
    # ---------------------------------------------------------------------
    metrics_prefix: str = ""
    pbar_description: str | None = "Training"

    # ---------------------------------------------------------------------
    # Utilities
    # ---------------------------------------------------------------------
    def get_with_default(self, key: str, default: Any) -> Any:
        """
        Return the value of `key` if it is not None, otherwise return `default`.

        This helper is useful when downstream logic requires a concrete value
        and should not receive None.
        """
        value = getattr(self, key)
        return default if value is None else value
```

---

# `TrainingConfig` Documentation

## Overview

`TrainingConfig` is a centralized configuration object for controlling all
aspects of a training run, including:

* Core training behavior
* Checkpointing
* Logging and profiling
* Sharding and resource management
* User-facing progress reporting

---

## Parameters (Attributes)

### Core Training

#### `eval_every_n_steps: int = 100`

Number of training steps between evaluation runs on the validation dataset.

* Smaller values provide more frequent feedback
* More frequent evaluation increases overhead

---

#### `max_steps: int | None = None`

Maximum number of training steps to execute.

* If an integer is provided, training stops after exactly that many steps
* If `None`, training continues until the dataset is exhausted

---

#### `gradient_accumulation_steps: int | None = None`

Enables gradient accumulation across multiple micro-batches.

* If set to `N > 1`, gradients are accumulated for `N` steps before one optimizer update
* If `None`, gradients are applied after every micro-batch

---

### Checkpointing

#### `checkpoint_root_directory: str | None = None`

Base directory where training checkpoints are saved.

* Typically includes model parameters and optimizer state
* Enables resuming training from saved state

---

#### `checkpointing_options: CheckpointManagerOptions | None = None`

Advanced configuration for checkpoint management, including:

* Save frequency
* Retention policies
* Backup behavior
* Asynchronous saving

If `None`, default Orbax checkpointing behavior is used.

---

### Logging and Profiling

#### `metrics_logging_options: MetricsLoggerOptions`

Configures metric logging behavior, such as:

* Log directory
* Project and run names
* Flush frequency
* Logging backends (e.g., TensorBoard, Weights & Biases)

---

#### `profiler_options: ProfilerOptions | None = None`

Optional configuration for the JAX profiler, including:

* Output directory
* Profiling start step
* Number of steps to profile

Useful for performance analysis and optimization.

---

### Sharding and Resource Management

#### `data_sharding_axis: Tuple[str, ...] = ("fsdp",)`

Specifies the sharding axes used for distributing input data across devices.

Common examples:

* `("fsdp",)` — Fully Sharded Data Parallel training
* `("data",)` — Classic data parallelism

---

#### `max_inflight_computations: int = 2`

Maximum number of training steps that may be scheduled concurrently.

* Higher values can improve throughput
* Increases peak memory usage

---

### UI / UX

#### `metrics_prefix: str = ""`

Prefix prepended to all logged metric names.

* Useful for grouping metrics from different training stages or runs

---

#### `pbar_description: str | None = "Training"`

Description displayed in the training progress bar.

* Set to `None` to disable the label

---

## Methods

### `get_with_default(key: str, default: Any) -> Any`

Safely retrieves a configuration attribute.

* Returns the attribute value if it is not `None`
* Otherwise returns the provided `default`

Useful when downstream logic requires a concrete value and should not receive
`None`.


---

### Class: `tunix.DPOTrainer`

```python
DPOTrainer(model, ref_model, optimizer, training_config, tokenizer)
```

**Purpose**: Direct Preference Optimization (DPO) and ORPO trainer for aligning LLMs with human/AI preferences without requiring a separate reward model.

**Background**: DPO is a preference tuning method that:
- Eliminates text generation in the training loop
- Bypasses reward modeling entirely
- Uses preference pairs (chosen/rejected responses) with classification-style loss
- More efficient and performant than traditional RLHF

**ORPO Variant**: Odds Ratio Preference Optimization combines supervised fine-tuning with preference alignment without requiring a reference model, making it ~50% more memory-efficient.

**Constructor Parameters**:

| Parameter | Type | Description |
|-----------|------|-------------|
| `model` | `Module` | The policy model to train (typically with LoRA applied) |
| `ref_model` | `Module` | Reference model for KL divergence computation (frozen during training) |
| `optimizer` | `GradientTransformation` | Optax optimizer for weight updates |
| `training_config` | `DPOTrainingConfig` | DPO-specific configuration |
| `tokenizer` | `Tokenizer` | Tokenizer for processing text inputs |

**References**:
- DPO Paper: https://arxiv.org/abs/2305.18290
- ORPO Paper: https://arxiv.org/abs/2403.07691

---

### Class: `tunix.DPOTrainingConfig`

```python
DPOTrainingConfig(*, eval_every_n_steps, ...)
```

**Purpose**: Configuration for DPO/ORPO training. Extends TrainingConfig with preference-specific parameters.

**Attributes**:

| Attribute | Type | Default | Description |
|-----------|------|---------|-------------|
| `algorithm` | `str` | `"dpo"` | Algorithm variant: `"dpo"` or `"orpo"`. ORPO doesn't require a reference model. |
| `beta` | `float` | `0.1` | Temperature parameter controlling how much to trust preference data. Higher values make the model more confident in learned preferences. Typical range: 0.05-0.5. |
| `label_smoothing` | `float` | `0.0` | Smoothing factor for preference labels. Helps prevent overconfident predictions. Range: 0.0-0.3. |
| `lambda_orpo` | `float` | `0.1` | Weighting factor for ORPO loss component. Only used when algorithm="orpo". |
| `max_prompt_length` | `int` | `512` | Maximum tokens for the prompt portion. Sequences are truncated if longer. |
| `max_response_length` | `int` | `512` | Maximum tokens for response (chosen/rejected). |

---

### Class: `tunix.MetricsLogger`

```python
MetricsLogger(metrics_logger_options=None)
```

**Purpose**: Simple metrics logger supporting multiple backends. Handles logging, aggregation, and persistence of training metrics.

**Backends**: TensorBoard, Weights & Biases, custom backends via protocol

**Methods**:

#### `log(metric_name, value, step, mode="train")`
Logs a scalar metric value to local history and via jax.monitoring.

**Parameters**:
- `metric_name`: Name of the metric (e.g., "loss", "accuracy")
- `value`: Scalar value to log
- `step`: Training step number
- `mode`: Either "train" or "eval"

#### `get_metric(metric_name, mode="train")`
Returns the mean metric value for the given metric name and mode.

#### `get_metric_history(metric_name, mode="train")`
Returns all past metric values as a list.

#### `metric_exists(metric_name, mode="train")`
Checks if the metric exists for the given name and mode.

**Returns**: `bool`

#### `close()`
Closes all registered logging backends. Always call this when training completes.

---

### Class: `tunix.MetricsLoggerOptions`

```python
MetricsLoggerOptions(log_dir, ...)
```

**Purpose**: Configuration for metrics logging backends and behavior.

**Attributes**:

| Attribute | Type | Default | Description |
|-----------|------|---------|-------------|
| `log_dir` | `str` | Required | Directory for saving logs (TensorBoard events, etc.) |
| `project_name` | `str` | `None` | Project name for WandB or similar services |
| `run_name` | `str` | `None` | Unique run identifier |
| `flush_every_n_steps` | `int` | `100` | How often to flush metrics to disk |
| `backend_factories` | `List[Callable]` | `[]` | List of factory functions creating LoggingBackend instances |

**Methods**:

#### `create_backends()`
Factory method creating fresh backend instances. Called internally by MetricsLogger.

---

## RL API Reference

The Reinforcement Learning module provides algorithms and infrastructure for training LLMs with reward signals.

### Class: `tunix.GRPOConfig`

```python
GRPOConfig(*, algo_variant="grpo", ...)
```

**Purpose**: Configuration for Group Relative Policy Optimization algorithms. GRPO is an RL algorithm that enhances LLM reasoning abilities by generating multiple responses per prompt and computing relative advantages.

**Key Insight**: GRPO eliminates the need for a separate value function model (unlike PPO), reducing memory usage significantly while maintaining training effectiveness.

**Attributes**:

| Attribute | Type | Default | Description |
|-----------|------|---------|-------------|
| `algo_variant` | `str` | `"grpo"` | Core algorithm variant to use. |
| `advantage_estimator` | `str` | `"grpo"` | Method for computing advantages. Determines how rewards are transformed into policy gradients. |
| `policy_loss_fn` | `str` | `"clip"` | Policy loss function type. "clip" uses PPO-style clipping for stable updates. |
| `loss_agg_mode` | `str` | `"mean"` | Aggregation mode for loss: "mean", "sum". Affects gradient scaling. |
| `loss_algo` | `str` | `"grpo"` | Loss algorithm: "grpo" (per-batch normalized) or "gspo" (gspo-token, more flexible). **Deprecated**: Use algo_variant instead. |
| `num_generations` | `int` | `4` | Number of responses generated per prompt (G in Algorithm 1). Higher values provide better advantage estimates but increase computation. Typical range: 2-16. |
| `num_iterations` | `int` | `1` | Policy update iterations per batch (μ in GRPO). Higher values mean more updates per data batch. |
| `beta` | `float` | `0.01` | KL divergence penalty coefficient (β). Prevents policy from deviating too far from reference model. 0.0 disables KL penalty. Typical range: 0.01-0.1. |
| `epsilon` | `float` | `0.2` | Clipping epsilon (ε) for ratio clipping. Ensures stable updates like PPO. Typical range: 0.1-0.3. |
| `epsilon_high` | `float` | `None` | Upper bound for asymmetric clipping. If None, uses epsilon. |

**References**:
- GRPO Paper: https://arxiv.org/abs/2402.03300
- GSPO Paper: https://arxiv.org/abs/2507.18071

---

### Class: `tunix.GRPOLearner`

```python
GRPOLearner(rl_cluster, algo_config, reward_fns)
```

**Purpose**: GRPO learner implementing the training loop. Handles generation, reward computation, advantage estimation, and policy updates.

**How GRPO Works**:
1. For each prompt, generate G responses using current policy
2. Compute rewards for each response using reward functions
3. Calculate relative advantages within the group
4. Update policy using clipped objective with KL penalty

**Algorithm** (from paper):
```
Input: initial policy πθ, reward models rφ, prompts D, hyperparameters ε, β, μ
for iteration = 1, ..., I do
    reference model πref ← πθ
    for step = 1, ..., M do
        Sample batch D♭ from D
        Update old policy πθold ← πθ
        Sample G outputs {oi}ᴳᵢ₌₁ ~ πθold(· | q) for each q ∈ D♭
        Compute rewards {ri}ᴳᵢ₌₁ for each output
        Compute advantages via group relative estimation
        for GRPO iteration = 1, ..., μ do
            Update policy πθ by maximizing GRPO objective
Output: πθ
```

**Constructor Parameters**:

| Parameter | Type | Description |
|-----------|------|-------------|
| `rl_cluster` | `RLCluster` | Cluster containing actor, reference, and rollout components |
| `algo_config` | `GRPOConfig` | GRPO hyperparameters and settings |
| `reward_fns` | `List[RewardFn]` | List of reward functions. Each function returns rewards for response quality. Rewards are summed. |

**Methods**:

#### `train(train_ds, eval_ds=None, skip_jit=False)`
GRPO training loop implementation.

**Parameters**:
- `train_ds`: Iterable of dicts with key 'prompts'
- `eval_ds`: Optional evaluation dataset
- `skip_jit`: Disable JIT for debugging

**Notes**:
- The outer loop (I) for updating reference model is not implemented yet
- Sample and train share the same model reference

---

### Type: `tunix.RewardFn`

```python
RewardFn = Callable[[...], List[float]]
```

**Purpose**: Type alias for reward functions. Functions that evaluate response quality and return scalar rewards.

**Signature**: `(prompts, completions, **kwargs) -> List[float]`

**Parameters**:
- `prompts`: List of input prompts
- `completions`: List of generated responses
- `**kwargs`: Additional context (e.g., ground truth answers)

**Returns**: List of float rewards, one per completion

**Example Reward Functions**:

```python
def format_reward(prompts, completions, **kwargs):
    """Reward for following output format instructions."""
    return [3.0 if matches_format(c) else 0.0 for c in completions]

def correctness_reward(prompts, completions, answer, **kwargs):
    """Reward for correct answers."""
    return [1.0 if extract_answer(c) == a else 0.0 
            for c, a in zip(completions, answer)]
```

---

### Class: `tunix.PPOConfig`

```python
PPOConfig(*, algo_variant="ppo", ...)
```

**Purpose**: Configuration for Proximal Policy Optimization. PPO uses an actor-critic architecture with clipped surrogate objectives for stable policy updates.

**Attributes**:

| Attribute | Type | Default | Description |
|-----------|------|---------|-------------|
| `num_iterations` | `int` | `4` | Optimization epochs per rollout batch. More iterations = more updates per data. Typical: 1-10. |
| `mini_batch_size` | `int` | `None` | Batch size for actual updates. Rollout batch is split into mini-batches. |
| `gamma` | `float` | `1.0` | Discount factor for future rewards in GAE. 1.0 = no discounting. For episodic tasks, use 0.99. |
| `gae_lambda` | `float` | `0.95` | Lambda parameter for Generalized Advantage Estimation. Balances bias-variance. Range: 0.9-1.0. |
| `beta` | `float` | `0.01` | KL divergence penalty coefficient. Controls policy constraint strength. |
| `epsilon` | `float` | `0.2` | Clipping range for policy ratio. Standard PPO value. |
| `epsilon_low` | `float` | `None` | Lower clipping bound. Defaults to epsilon if not set. |
| `epsilon_high` | `float` | `None` | Upper clipping bound. Defaults to epsilon if not set. |
| `epsilon_c` | `float` | `None` | Dual-clip PPO lower bound (reference: https://arxiv.org/abs/1912.09729). None disables dual-clip. |
| `entropy_coef` | `float` | `0.01` | Entropy bonus coefficient. Encourages exploration. None or 0.0 disables. |
| `clip_range_value` | `float` | `0.2` | Clipping range for value function loss. Stabilizes value learning. |
| `kl_method` | `str` | `"low_var_kl"` | KL divergence method: "low_var_kl", "kl", or "mse_kl". |

**Reference**: PPO Paper: https://arxiv.org/abs/1707.06347

---

### Class: `tunix.PPOLearner`

```python
PPOLearner(rl_cluster, ppo_config, ...)
```

**Purpose**: PPO learner using actor-critic architecture. The actor (policy) learns actions, while the critic (value model) estimates state values for advantage calculation.

**Key Features**:
- Clipped surrogate objective prevents destructive updates
- Balances exploration and exploitation
- Robust across diverse RL tasks

**Methods**:

#### `train(train_ds, eval_ds=None)`
PPO training loop with rollout generation and mini-batch updates.

---

### Class: `tunix.RLCluster`

```python
RLCluster(*, actor, critic=None, reference=None, tokenizer, cluster_config)
```

**Purpose**: Central orchestrator for RL training components. Manages actor, critic, reference models and handles generation, log probability computation, and weight synchronization.

**Constructor Parameters**:

| Parameter | Type | Description |
|-----------|------|-------------|
| `actor` | `Module` | Policy model (typically with LoRA). This model is trained. |
| `critic` | `Module` | Value model for PPO (optional for GRPO). |
| `reference` | `Module` | Reference model for KL computation. Frozen during training. |
| `tokenizer` | `Tokenizer` | Text tokenizer for input processing. |
| `cluster_config` | `ClusterConfig` | Cluster configuration including sharding, rollout, and training settings. |

**Attributes**:

| Attribute | Type | Description |
|-----------|------|-------------|
| `actor_trainer` | `PeftTrainer` | Trainer instance for the actor model. |
| `critic_trainer` | `PeftTrainer` | Trainer for critic model (if applicable). |
| `inference_worker` | `InferenceWorker` | Handles model inference operations. |
| `rollout` | `BaseRollout` | Rollout engine for text generation. |
| `perf` | `PerformanceTracker` | Training performance metrics. |

**Methods**:

#### `generate(prompts, apply_chat_template=True, mode=Mode.TRAIN, micro_batch_size=None)`
Generates text from prompts using the current policy.

**Parameters**:
- `prompts`: List of prompts (strings or conversation dicts)
- `apply_chat_template`: Whether to apply chat formatting
- `mode`: TRAIN or EVAL mode
- `micro_batch_size`: Batch size for generation (None = no micro-batching)

**Returns**: Generated text completions

#### `get_old_per_token_logps(input_ids, attention_mask)`
Computes per-token log probabilities from the old policy (before update).

#### `get_ref_per_token_logps(input_ids, attention_mask)`
Computes per-token log probabilities from the reference model.

#### `get_rewards(prompts, completions, reward_fns)`
Evaluates completions using reward functions.

#### `get_values(input_ids, attention_mask)`
Gets value estimates from the critic model (PPO only).

#### `update_actor(batch)`
Performs one update step on the actor model.

#### `update_critic(batch)`
Performs one update step on the critic model.

#### `sync_weights()`
Synchronizes weights between models (e.g., after updates).

#### `buffer_metrics(metrics, mode)`
Buffers metrics for logging when step increments.

**Parameters**:
- `metrics`: Dict mapping metric names to (value, aggregation_fn) tuples
- `mode`: TRAIN or EVAL

#### `buffer_metrics_async(metrics, mode, step)`
Async version for asynchronous training pipelines.

#### `with_external_metrics_logger(logger)`
Sets an external metrics logger.

#### `close()`
Releases all resources.

---

### Class: `tunix.ClusterConfig`

```python
ClusterConfig(*, role_to_mesh, ...)
```

**Purpose**: Configuration for RL cluster setup including model placement, sharding, and rollout settings.

**Attributes**:

| Attribute | Type | Description |
|-----------|------|-------------|
| `role_to_mesh` | `Dict[Role, Mesh]` | Maps model roles to JAX meshes. Key config for colocated vs disaggregated setup. |
| `role_to_logical_axis_rule` | `Dict[Role, Sequence]` | Logical to physical axis mapping for sharded models. |
| `rollout_engine` | `str | Type[BaseRollout]` | Rollout engine: "vanilla", "vllm", "sglang_jax", or custom class. |
| `offload_to_cpu` | `bool` | Whether to offload models to CPU between steps. Saves HBM but adds transfer overhead. |
| `training_config` | `RLTrainingConfig` | Training-specific configuration. |
| `rollout_config` | `RolloutConfig | Dict[Mode, RolloutConfig]` | Rollout settings. Can differ for TRAIN vs EVAL modes. |
| `rollout_vllm_model_version` | `str` | Model version string for vLLM engine. |
| `rollout_vllm_lora_config` | `Dict` | LoRA configuration for vLLM. |
| `rollout_vllm_hbm_utilization` | `float` | Percentage of TPU/GPU HBM for vLLM. |
| `rollout_vllm_init_with_random_weights` | `bool` | Initialize vLLM with random weights (for testing). |
| `rollout_vllm_tpu_backend_type` | `str` | TPU backend: "jax", "torchax", or "pytorch_xla". |
| `rollout_vllm_swap_space_size_gb` | `float` | CPU swap space (GiB) for KV cache overflow. |

---

### Class: `tunix.RLTrainingConfig`

```python
RLTrainingConfig(*, eval_every_n_steps, ...)
```

**Purpose**: Training configuration specific to RL algorithms.

**Attributes**:

| Attribute | Type | Description |
|-----------|------|-------------|
| `actor_optimizer` | `GradientTransformation` | Optimizer for actor model. |
| `critic_optimizer` | `GradientTransformation` | Optimizer for critic model (PPO). |
| `mini_batch_size` | `int` | Batch size for gradient updates. |
| `train_micro_batch_size` | `int` | Micro-batch size for training forward pass. |
| `rollout_micro_batch_size` | `int` | Micro-batch size for generation. |
| `compute_logps_micro_batch_size` | `int` | Micro-batch size for log probability computation. |

---

### Enum: `tunix.Role`

```python
class Role(Enum):
    ACTOR = "actor"
    CRITIC = "critic"
    REFERENCE = "reference"
    REWARD = "reward"
    ROLLOUT = "rollout"
```

**Purpose**: Enumeration of model roles in RL training.

| Role | Description |
|------|-------------|
| `ACTOR` | Policy model being trained |
| `CRITIC` | Value estimation model (PPO) |
| `REFERENCE` | Frozen reference for KL divergence |
| `REWARD` | Reward model (if using learned rewards) |
| `ROLLOUT` | Model used for text generation |

---

### Class: `tunix.RolloutConfig`

```python
RolloutConfig(max_tokens_to_generate=256, ...)
```

**Purpose**: Configuration for text generation during RL rollouts.

**Attributes**:

| Attribute | Type | Default | Description |
|-----------|------|---------|-------------|
| `max_tokens_to_generate` | `int` | `256` | Maximum tokens to generate per response. |
| `max_prompt_length` | `int` | `512` | Maximum input prompt length. |
| `kv_cache_size` | `int` | `None` | Size of KV cache. Auto-calculated if None. |
| `temperature` | `float` | `1.0` | Sampling temperature. Higher = more random. |
| `top_k` | `int` | `None` | Top-K sampling. None = disabled. |
| `top_p` | `float` | `None` | Nucleus sampling threshold. |
| `eos_tokens` | `List[int]` | `[]` | Token IDs that stop generation. |
| `seed` | `int` | `None` | Random seed for reproducibility. |
| `data_type` | `DType` | `float32` | Data type for computation. |
| `data_parallel_size` | `int` | `1` | Data parallelism degree. |
| `tensor_parallel_size` | `int` | `1` | Tensor parallelism degree. |
| `rollout_mapping_config` | `Dict` | `None` | Custom weight mapping configuration. |

**SGLang-Jax Specific**:
| Attribute | Description |
|-----------|-------------|
| `rollout_sglang_jax_chunked_prefill_size` | Chunked prefill size |
| `rollout_sglang_jax_context_length` | Context length limit |
| `rollout_sglang_jax_disable_radix_cache` | Disable radix cache |
| `rollout_sglang_jax_enable_deterministic_sampling` | Enable deterministic sampling |
| `rollout_sglang_jax_mem_fraction_static` | Static memory fraction |
| `rollout_sglang_jax_page_size` | Page size for paged attention |

**vLLM Specific**:
| Attribute | Description |
|-----------|-------------|
| `rollout_vllm_additional_config` | Additional vLLM config dict |
| `rollout_vllm_async_scheduling` | Enable async scheduling |
| `rollout_vllm_hbm_utilization` | HBM utilization percentage |
| `rollout_vllm_server_mode` | Run vLLM as server |

---

## Distillation API Reference

Knowledge distillation enables training smaller "student" models to mimic larger "teacher" models.

### Class: `tunix.DistillationTrainer`

```python
DistillationTrainer(student_model, teacher_model, optimizer, training_config)
```

**Purpose**: Trainer for knowledge distillation from a teacher model to a student model. Supports various distillation strategies including logit matching, attention transfer, and feature projection.

**Constructor Parameters**:

| Parameter | Type | Description |
|-----------|------|-------------|
| `student_model` | `Module` | The smaller model being trained. Receives gradient updates. |
| `teacher_model` | `Module` | The larger model providing knowledge. Frozen during training. |
| `optimizer` | `GradientTransformation` | Optax optimizer for student model updates. |
| `training_config` | `TrainingConfig` | Training configuration (alias for `DistillationTrainingConfig`). |

**Methods**:

#### `close()`
Closes the trainer and releases resources. Writes buffered metrics, saves final checkpoint, and closes managers.

#### `get_train_loss(batch)`
Computes training loss for a given batch.

**Returns**: `jax.Array` loss value

#### `get_eval_loss(batch)`
Computes evaluation loss.

**Returns**: `jax.Array` loss value

#### `with_gen_model_input_fn(gen_model_input_fn)`
Sets the function that generates model input from training data.

**Important**: The output of this function is passed to the loss function, so argument types must match what the loss function expects.

**Parameters**:
- `gen_model_input_fn`: Function with signature `(batch) -> model_inputs`

**Returns**: `DistillationTrainer` (self)

#### `with_loss_fn(loss_fn)`
Sets a custom distillation loss function.

**Returns**: `DistillationTrainer` (self)

### Type: `tunix.DistillationTrainingConfig`

Alias for `TrainingConfig`. See [TrainingConfig](#class-tunixtrainingconfig) for attributes.

---

## Generation API Reference

The Generation module provides utilities for text generation with transformer models.

### Class: `tunix.Sampler`

```python
Sampler(transformer, tokenizer, cache_config)
```

**Purpose**: Sampler for generating text from transformer models. Handles tokenization, KV caching, and various sampling strategies.

**Constructor Parameters**:

| Parameter | Type | Description |
|-----------|------|-------------|
| `transformer` | `Module` | The transformer model for generation. |
| `tokenizer` | `Tokenizer` | Text tokenizer for encoding/decoding. |
| `cache_config` | `CacheConfig` | Configuration for the KV cache. |

**Properties**:

| Property | Type | Description |
|----------|------|-------------|
| `dtype` | `jnp.dtype` | Data type used for computation (e.g., float32, bfloat16). |
| `transformer` | `Module` | The underlying transformer module. |
| `transformer_state` | `State` | The Flax NNX state of the transformer. |

**Methods**:

#### `__call__(input_strings, max_generation_steps, temperature=1.0, top_k=None, top_p=None, echo=False, seed=None, eos_tokens=[])`
Main generation method. Generates text completions for input strings.

**Parameters**:
- `input_strings`: List of prompt strings
- `max_generation_steps`: Maximum tokens to generate
- `temperature`: Sampling temperature (None = greedy)
- `top_k`: Top-K sampling (None = disabled)
- `top_p`: Nucleus sampling (None = disabled)
- `echo`: Include prompt in output
- `seed`: Random seed
- `eos_tokens`: Token IDs that stop generation

**Returns**: `SamplerOutput` with `.text` attribute containing generated strings

#### `init_sample_state(input_ids, positions)`
Initializes the sampling state for given prompts.

**Parameters**:
- `input_ids`: Tokenized input (batch_size, seq_len)
- `positions`: Position indices

**Returns**: `SampleState` for iterative generation

#### `tokenize(input_string)`
Tokenizes an input string.

**Parameters**:
- `input_string`: Text to tokenize

**Returns**: `jax.Array` of token IDs

---

### Class: `tunix.CacheConfig`

```python
CacheConfig(cache_size, num_layers, num_kv_heads, head_dim)
```

**Purpose**: Configuration for the KV cache used during autoregressive generation.

**Attributes**:

| Attribute | Type | Description |
|-----------|------|-------------|
| `cache_size` | `int` | Maximum sequence length the cache can hold. Should be `max_prompt_length + max_generation_steps + buffer`. |
| `num_layers` | `int` | Number of transformer layers. Must match model architecture. |
| `num_kv_heads` | `int` | Number of key-value heads. For MQA/GQA, this differs from query heads. |
| `head_dim` | `int` | Dimension of each attention head. |

**Example**:
```python
cache_config = CacheConfig(
    cache_size=1024,  # 256 prompt + 768 generation + buffer
    num_layers=model_config.num_layers,
    num_kv_heads=model_config.num_kv_heads,
    head_dim=model_config.head_dim,
)
```

---

## Practical Examples

### Example 1: DPO Training with Gemma

Fine-tune Gemma 3 1B-IT on math problems using Direct Preference Optimization.

**Setup**:
```python
# Install packages
pip install git+https://github.com/google/tunix
pip install git+https://github.com/google/qwix
pip install git+https://github.com/google/flax
```

**Load Model with LoRA**:
```python
from flax import nnx
from huggingface_hub import snapshot_download
import qwix
from tunix.models.gemma3 import model as gemma_lib
from tunix.models.gemma3 import params_safetensors as params_lib

# Download model
model_id = "google/gemma-3-1b-it"
local_path = snapshot_download(repo_id=model_id)

# Create model
model_config = gemma_lib.ModelConfig.gemma3_1b()
mesh = jax.make_mesh((1, 1), ("fsdp", "tp"))
with mesh:
    gemma = params_lib.create_model_from_safe_tensors(local_path, model_config, mesh)

# Apply LoRA
lora_provider = qwix.LoraProvider(
    module_path=".*q_einsum|.*kv_einsum|.*gate_proj|.*down_proj|.*up_proj|.*attn_vec_einsum",
    rank=32,
    alpha=16.0,
)
lora_model = qwix.apply_lora_to_model(gemma, lora_provider, **gemma.get_model_input())
```

**Configure DPO Training**:
```python
from tunix.sft.dpo.dpo_trainer import DPOTrainer, DPOTrainingConfig
import optax

optimizer = optax.chain(
    optax.clip_by_global_norm(max_norm=0.1),
    optax.adamw(
        learning_rate=optax.schedules.warmup_cosine_decay_schedule(
            init_value=0.0, peak_value=3e-5, warmup_steps=50, decay_steps=1024
        ),
        b1=0.9, b2=0.99, weight_decay=0.1,
    ),
)

dpo_config = DPOTrainingConfig(
    beta=0.1,
    max_prompt_length=192,
    max_response_length=192,
    max_steps=1024,
    eval_every_n_steps=256,
)

trainer = DPOTrainer(
    model=lora_model,
    ref_model=gemma,
    optimizer=optimizer,
    training_config=dpo_config,
    tokenizer=tokenizer,
)
```

**Prepare Data**:
```python
from datasets import load_dataset
import grain

# DPO format: prompts, chosen_responses, rejected_responses
dataset = load_dataset("argilla/distilabel-intel-orca-dpo-pairs")
train_data = grain.MapDataset.source(dataset["train"]).map(
    lambda x: {
        "prompts": f"<start_of_turn>user\n{x['input']}<end_of_turn>\n<start_of_turn>model\n",
        "chosen_responses": x["chosen"],
        "rejected_responses": x["rejected"],
    }
).batch(2)

# Train
with mesh:
    trainer.train(train_data)
```

---

### Example 2: GRPO Training for Math Reasoning

Train Gemma on GSM8K using Group Relative Policy Optimization.

**Define Reward Functions**:
```python
import re

reasoning_start = "<reasoning>"
reasoning_end = "</reasoning>"
solution_start = "<answer>"
solution_end = "</answer>"

match_format = re.compile(
    rf"^[\s]{{0,}}{reasoning_start}.+?{reasoning_end}.*?"
    rf"{solution_start}(.+?){solution_end}[\s]{{0,}}$",
    flags=re.MULTILINE | re.DOTALL,
)

def reward_format(prompts, completions, **kwargs):
    """Reward for correct output format."""
    return [3.0 if match_format.search(c) else 0.0 for c in completions]

def reward_correctness(prompts, completions, answer, **kwargs):
    """Reward for correct numerical answer."""
    scores = []
    for completion, true_answer in zip(completions, answer):
        match = match_format.search(completion)
        if match and match.group(1).strip() == true_answer.strip():
            scores.append(3.0)
        else:
            scores.append(0.0)
    return scores
```

**Configure GRPO**:
```python
from tunix.rl import rl_cluster as rl_lib
from tunix.rl.grpo.grpo_learner import GRPOConfig, GRPOLearner
from tunix.rl.rollout import base_rollout

cluster_config = rl_lib.ClusterConfig(
    role_to_mesh={
        rl_lib.Role.ACTOR: mesh,
        rl_lib.Role.REFERENCE: mesh,
        rl_lib.Role.ROLLOUT: mesh,
    },
    rollout_engine='vanilla',
    offload_to_cpu=False,
    training_config=rl_lib.RLTrainingConfig(
        actor_optimizer=optimizer,
        max_steps=3000,
        mini_batch_size=1,
    ),
    rollout_config=base_rollout.RolloutConfig(
        max_tokens_to_generate=768,
        max_prompt_length=256,
        temperature=0.9,
        top_k=50,
    ),
)

grpo_config = GRPOConfig(
    num_generations=2,  # Generate 2 responses per prompt
    num_iterations=1,
    beta=0.08,          # KL penalty
    epsilon=0.2,        # Clipping
)

rl_cluster = rl_lib.RLCluster(
    actor=lora_model,
    reference=gemma,
    tokenizer=tokenizer,
    cluster_config=cluster_config,
)

trainer = GRPOLearner(
    rl_cluster=rl_cluster,
    reward_fns=[reward_format, reward_correctness],
    algo_config=grpo_config,
)

trainer.train(train_dataset)
```

---

### Example 3: Knowledge Distillation (Gemma 7B → 2B)

Transfer knowledge from a larger Gemma model to a smaller one.

```python
from tunix.distillation import DistillationTrainer

# Load teacher (7B) and student (2B) models
teacher = load_gemma_model("google/gemma-7b")
student = load_gemma_model("google/gemma-2b")

# Simple logit distillation loss
def distillation_loss(student_logits, teacher_logits, labels, temperature=2.0):
    # Soft targets from teacher
    teacher_probs = jax.nn.softmax(teacher_logits / temperature, axis=-1)
    student_log_probs = jax.nn.log_softmax(student_logits / temperature, axis=-1)
    
    # KL divergence
    kl_loss = -jnp.sum(teacher_probs * student_log_probs, axis=-1).mean()
    
    # Hard label loss
    ce_loss = optax.softmax_cross_entropy_with_integer_labels(student_logits, labels).mean()
    
    return 0.5 * kl_loss + 0.5 * ce_loss

trainer = DistillationTrainer(
    student_model=student,
    teacher_model=teacher,
    optimizer=optax.adamw(1e-4),
    training_config=TrainingConfig(max_steps=10000),
)
trainer.with_loss_fn(distillation_loss)
trainer.train(dataset)
```

---

### Example 4: LoRA/QLoRA Fine-tuning

Parameter-efficient fine-tuning with quantized LoRA.

```python
import qwix
from tunix.sft import PeftTrainer, TrainingConfig

# Apply QLoRA (quantized base weights + LoRA adapters)
qlora_provider = qwix.LoraProvider(
    module_path=".*q_einsum|.*kv_einsum|.*gate_proj|.*down_proj|.*up_proj",
    rank=64,
    alpha=64.0,
)

# Quantize base weights to int4
quantized_model = qwix.quantize_model(gemma, bits=4)

# Apply LoRA on top
lora_model = qwix.apply_lora_to_model(
    quantized_model, qlora_provider, **quantized_model.get_model_input()
)

# Configure trainer
config = TrainingConfig(
    max_steps=5000,
    eval_every_n_steps=500,
    checkpoint_root_directory="/tmp/qlora_ckpts",
)

trainer = PeftTrainer(
    model=lora_model,
    optimizer=optax.adamw(5e-5),
    training_config=config,
)

# Standard SFT training
trainer.train(sft_dataset)
```

---

## Best Practices

### Hyperparameter Guidelines

| Algorithm | Key Parameters | Typical Values |
|-----------|---------------|----------------|
| **LoRA** | rank, alpha | rank=32-128, alpha=16-64 |
| **DPO** | beta, label_smoothing | beta=0.1-0.5, smoothing=0.0-0.1 |
| **GRPO** | num_generations, beta, epsilon | G=2-8, β=0.01-0.1, ε=0.1-0.3 |
| **PPO** | num_iterations, gamma, gae_lambda | epochs=1-10, γ=0.99, λ=0.95 |

### Memory Optimization

1. **Use gradient checkpointing** for large models
2. **Offload to CPU** between inference and training (`offload_to_cpu=True`)
3. **Use FSDP sharding** for multi-device training
4. **Enable micro-batching** for large sequences

### Sharding Configuration

```python
# Single TPU v6e-1
mesh = jax.make_mesh((1, 1), ("fsdp", "tp"))

# TPU v6e-8 (8 chips)
mesh = jax.make_mesh((1, 4), ("fsdp", "tp"))

# Multi-host (adjust based on topology)
mesh = jax.make_mesh((num_hosts, devices_per_host), ("fsdp", "tp"))
```

---

## Citation

```bibtex
@software{tunix2025,
  title = {TuNix: JAX-native LLM Post-Training Library},
  author = {TuNix Developers},
  year = {2025},
  url = {https://github.com/google/tunix}
}
```

---

## Resources

- **Documentation**: https://tunix.readthedocs.io/
- **GitHub Repository**: https://github.com/google/tunix
- **Discussion Forum**: https://github.com/google/tunix/discussions
- **Contributing Guide**: https://tunix.readthedocs.io/en/latest/contributing.html

### Related Projects
- **Flax NNX**: https://flax.readthedocs.io/
- **Qwix**: https://github.com/google/qwix (LoRA/QLoRA)
- **Optax**: https://optax.readthedocs.io/ (Optimizers)
- **Orbax**: https://orbax.readthedocs.io/ (Checkpointing)

