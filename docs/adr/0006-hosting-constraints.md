# ADR 0006: Hosting platform and deployment status

## Status
Deployment automation implemented; public hosting deferred pending the choice of a platform.

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
