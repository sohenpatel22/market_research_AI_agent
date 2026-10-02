# ADR 0006: Hosting: what was built, and the constraint that paused the public deploy

## Context
The goal was a free, public demo: a Docker image on a Hugging Face Space talking to a free Neon
Postgres, deployed from GitHub Actions.

## What was built and verified
- Runtime image (multi-stage, non-root, CPU-only PyTorch, models baked in, healthcheck), compose
  stack, and a CI job that builds it and smoke-tests the running container against Postgres.
- Production database on Neon (PostgreSQL 18, pgvector 0.8) seeded by a workflow that runs the real
  ingestion pipeline on a GitHub runner, then refreshed weekly.
- A deploy workflow/module that copies secrets into the Space, uploads a minimal build context,
  waits for the build and smoke-tests the live URL.

## What blocked it
The deploy reached Hugging Face and failed with `402 Payment Required`: *"Static Spaces are free for
everyone, but hosting Gradio and Docker Spaces on free cpu-basic requires a PRO subscription."* The
Space that already existed had been created on ZeroGPU hardware, which also cannot be moved to CPU
without PRO.

## Decision
Do not make a paid subscription a hidden dependency of the repo. The deploy workflow is manual-only,
its failure mode is documented, and the project runs end to end locally with one command
(`make docker-up`). The deploy automation is complete and re-enabling it is a one-line trigger change
once the account can host a Docker Space (or the same image can be pushed to another host).

## Alternatives considered
Google Cloud Run (scale to zero, needs a billing account), Fly.io / Render (small always-free tiers
do not fit the ~1.5 GB memory of PyTorch plus two models), AWS App Runner / ECS (paid).
