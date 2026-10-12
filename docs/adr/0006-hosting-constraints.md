# ADR 0006: Hosting platform and deployment status

## Status
Accepted. The app is hosted on a Hugging Face Gradio Space; the Docker image is kept as the portable artifact.

## Context
The target was a public demo: a Docker image running on a managed container host, backed by a managed
PostgreSQL with pgvector, deployed from GitHub Actions.

## What was implemented and validated
- A multi-stage runtime image (non-root, CPU-only PyTorch, models included, healthcheck), a compose stack, and
  a CI job that builds the image and smoke-tests the running container against Postgres.
- A production database on Neon (PostgreSQL 18, pgvector 0.8), seeded by a workflow that runs the ingestion
  pipeline and refreshed weekly.
- A deployment workflow and module for a Hugging Face Docker Space: secrets are copied into the Space, a
  minimal build context is uploaded, and the live URL is smoke-tested after the build.

## Constraint
Docker Spaces on managed CPU hardware are not available on the account's current plan, so the build step of
the deployment could not be completed. The workflow reaches the platform and uploads the build context; the
Space cannot be started.

## Decision
Keep the deployment workflow in the repository, triggered manually, and do not make hosting a dependency of
the application. The project is fully runnable locally with `make docker-up`. Public hosting will be enabled
once a platform is selected; no application change is needed because the same image is used.

## Options for hosting
- Hugging Face Docker Space, on an account plan that includes managed CPU hardware.
- A serverless container service (for example Google Cloud Run), which scales to zero and needs a billing account.
- Oracle Cloud Always Free (Ampere A1, up to 4 OCPU / 24 GB). It needs a card for verification but no charge
  on Always Free resources. A deployment workflow for it is in `deploy-oracle.yml` ([guide](../oracle-deploy.md)).
- A container service on AWS (App Runner or ECS), which suits the AWS experience listed on the resume.
- Small always-free tiers (Render, Fly.io) were considered but do not provide the roughly 1.5 GB of memory
  that PyTorch and the two models need.

## Outcome
The Space on ZeroGPU hardware runs the app on CPU. Four adjustments were needed and are covered by the deploy
module: torch pinned to a release ZeroGPU accepts, pydantic left to the resolver (Gradio caps it), Gradio's SSR
proxy disabled so uvicorn can bind the port, and a small GPU-decorated probe app so the platform's startup
check passes. The embedding model and reranker are loaded with an explicit CPU device because ZeroGPU emulates CUDA.
`deploy.yml` redeploys on demand and smoke-tests `/health`, the UI and a real `/chat` question.
