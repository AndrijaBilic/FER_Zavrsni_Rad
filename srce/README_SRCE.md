# Running the Enumerability Steering Experiments on Supek

This is the SRCE/Supek version of the Colab workflow. It runs the same
`phase2_activation_steering_enumerability.py` code as a PBS batch job.

## 1. Copy the project to your SRCE directory

Use your own Supek workspace, for example an `andrija` project directory:

```bash
cd ~/andrija
git clone <your-private-repo-url> enumerability-steering
cd enumerability-steering
```

If you are not using git yet, copy at least these files/directories:

```text
phase2_activation_steering_enumerability.py
srce/
```

## 2. Create the Python environment with uv

Run this once:

```bash
bash srce/setup_uv_env.sh
```

This creates `.venv`, installs CUDA-enabled PyTorch, and then installs the
project Python dependencies.

If `uv` is missing:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

Then open a new shell or source your shell profile and rerun the setup script.

## 3. Hugging Face access token for gated models

Llama and Gemma checkpoints are gated on Hugging Face. After your account has
accepted the model license, create a local token file in the repo root on SRCE:

```bash
cd ~/andrija/enumerability-steering
printf '%s\n' 'hf_your_token_here' > .hf_token
chmod 600 .hf_token
```

`.hf_token` is ignored by git. The setup and PBS scripts automatically export it
as `HF_TOKEN`/`HUGGING_FACE_HUB_TOKEN` when it exists.

You can also avoid a file and pass the token only for one command:

```bash
HF_TOKEN=hf_your_token_here MODEL_CHOICE=llama31_8b_it bash srce/prepare_hf_assets.sh
```

## 4. First debug run: gpu-test queue

Before submitting a PBS job, prepare Hugging Face assets from a node with
internet access:

```bash
MODEL_CHOICE=mistral bash srce/prepare_hf_assets.sh
```

This creates:

```text
outputs/df_webq_balanced.csv
.cache/huggingface/
```

The PBS scripts run in offline mode, because compute nodes may not have
outbound network access.

Submit the shorter test job:

```bash
qsub srce/run_phase2_activation_mistral_gpu_test.pbs
```

Check output while it runs:

```bash
qstat -u "$USER"
qtail <job_id>
qtail -e <job_id>
```

PBS starts jobs in `$HOME` by default, so the script explicitly runs:

```bash
cd "$PBS_O_WORKDIR"
```

That keeps outputs inside this project directory.

## 5. Main run: gpu queue

Once the test job works, submit:

```bash
qsub srce/run_phase2_activation_mistral_gpu.pbs
```

## 4b. Gemma Judge

After the steering generation files exist, cache Gemma from a node with internet
access:

```bash
JUDGE_MODEL=google/gemma-4-31B-it MODEL_CHOICE=mistral bash srce/prepare_hf_assets.sh
```

Then test the judge on 12 rows:

```bash
qsub srce/run_gemma_judge_test.pbs
```

For a non-default steering model, pass the same model choice used for
generation:

```bash
qsub -v MODEL_CHOICE=llama31_8b_it srce/run_gemma_judge_test.pbs
qsub -v MODEL_CHOICE=qwen3_8b srce/run_gemma_judge_test.pbs
qsub -v MODEL_CHOICE=gemma3_12b_it srce/run_gemma_judge_test.pbs
```

If the test succeeds, judge all current curated and WebQuestions rows:

```bash
qsub srce/run_gemma_judge_full.pbs
```

or, for a specific steering model:

```bash
qsub -v MODEL_CHOICE=llama31_8b_it srce/run_gemma_judge_full.pbs
```

The judge PBS scripts use INT8 quantization by default:

```bash
JUDGE_QUANTIZATION=int8
```

For smaller GPUs you can switch this to:

```bash
JUDGE_QUANTIZATION=4bit
```

Judge outputs go to:

```text
outputs/phase2_activation_steering_enumerability/mistral_7b/gemma_judge/
```

Important judge files:

```text
judged_generations.csv
judge_alpha_summary.csv
curated_judge_alpha_summary.csv
webq_judge_alpha_summary.csv
raw_judge_outputs.json
```

Outputs go to:

```text
outputs/phase2_activation_steering_enumerability/mistral_7b/
```

The most important files are:

```text
curated_manual_review_steered_generations.csv
curated_steering_alpha_summary.csv
webq_manual_review_steered_generations.csv
webq_steering_alpha_summary.csv
ood_manual_review_steered_generations.csv
ood_steering_alpha_summary.csv
manual_review_steered_generations.csv
steering_alpha_summary.csv
direction_selection_scores.csv
best_direction_metadata.json
projection_classifier_results.json
```

## 5. Current experiment settings

The PBS scripts set experiment options through environment variables, so you do
not need to edit the Python file for normal runs.

Useful variables:

```bash
MODEL_CHOICE=mistral
MODEL_CHOICE=llama31_8b_it
MODEL_CHOICE=qwen3_8b
MODEL_CHOICE=gemma3_12b_it
MODEL_QUANTIZATION=4bit
MODEL_QUANTIZATION=int8
MODEL_QUANTIZATION=none
TORCH_DTYPE=float16
TORCH_DTYPE=bfloat16
MAX_TRAIN_PER_CLASS=600
MAX_VAL_PER_CLASS=200
MAX_TEST_PER_CLASS=200
MAX_MANUAL_QUESTIONS=10
WEBQ_EVAL_PER_CLASS=50
MAX_OOD_QUESTIONS=12
ALPHAS="-1.5,-1.0,-0.5,0.0,0.5,1.0,1.5"
```

`WEBQ_EVAL_PER_CLASS=50` means evaluate steering on 50 held-out single
questions and 50 held-out multiple questions. Set it to `-1` only if you want
the whole held-out WebQuestions test split.

To cache a model before submitting an offline PBS job:

```bash
MODEL_CHOICE=llama31_8b_it bash srce/prepare_hf_assets.sh
MODEL_CHOICE=qwen3_8b bash srce/prepare_hf_assets.sh
MODEL_CHOICE=gemma3_12b_it bash srce/prepare_hf_assets.sh
```

To submit a specific model on the regular GPU queue:

```bash
qsub -v MODEL_CHOICE=llama31_8b_it srce/run_phase2_activation_mistral_gpu.pbs
qsub -v MODEL_CHOICE=qwen3_8b srce/run_phase2_activation_mistral_gpu.pbs
```

For Gemma 3 12B, prefer the 96GB queue:

```bash
qsub -v MODEL_CHOICE=gemma3_12b_it srce/run_phase2_activation_gpu_bigmem.pbs
```

The 4-bit Gemma 3 12B run can be numerically unstable. Use the bf16 smoke test
first:

```bash
qsub srce/run_phase2_gemma_bf16_smoke.pbs
```

If the smoke-test log has finite AUC values and non-empty generations, run the
full bf16 job and then judge that separate artifact folder:

```bash
act_job=$(qsub srce/run_phase2_gemma_bf16_full.pbs)
qsub -v MODEL_CHOICE=gemma3_12b_it,ARTIFACT_SUBDIR=phase2_activation_steering_enumerability_gemma_bf16 -W depend=afterok:${act_job} srce/run_gemma_judge_full.pbs
```

Llama 3.1 may require accepting Meta's model license and having a Hugging Face
token available in the cache/preparation environment.

## 6. Notes from the Supek docs

- Supek uses PBS Pro, and jobs are submitted with `qsub`.
- GPU jobs must use the `gpu` or `gpu-test` queues.
- The docs say GPU queues default to one GPU and about 120 GiB RAM, but the
  scripts request these explicitly.
- PBS does not automatically run inside the submit directory, so `cd
  "$PBS_O_WORKDIR"` is required.
- Standard output/error are written by PBS; with `#PBS -j oe`, they are merged.

## 7. Things to fill in if SRCE requires them

Some projects require an explicit project/group option. If your account needs
that, add one of these to the PBS header, depending on SRCE's instruction for
your allocation:

```bash
#PBS -P <project_code>
```

or:

```bash
#PBS -W group_list=<project_code>
```

Your mentor or the SRCE allocation page should tell you the exact project code.
