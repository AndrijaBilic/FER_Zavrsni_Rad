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

## 3. First debug run: gpu-test queue

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

## 4. Main run: gpu queue

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

If the test succeeds, judge all current curated and WebQuestions rows:

```bash
qsub srce/run_gemma_judge_full.pbs
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
MAX_TRAIN_PER_CLASS=600
MAX_VAL_PER_CLASS=200
MAX_TEST_PER_CLASS=200
MAX_MANUAL_QUESTIONS=10
WEBQ_EVAL_PER_CLASS=50
ALPHAS="-1.5,-1.0,-0.5,0.0,0.5,1.0,1.5"
CANDIDATE_LAYERS="16,17,18,19,20,21,22,23,24,25,26,27,28,29,30"
```

For Qwen, use:

```bash
MODEL_CHOICE=qwen
```

but keep in mind Qwen has not shown useful behavioral steering so far.

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
