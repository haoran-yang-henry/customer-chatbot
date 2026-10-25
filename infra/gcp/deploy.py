"""Deploy the backend to Google Cloud (infra/gcp/README.md).

    uv run python infra/gcp/deploy.py plan | deploy | pause | resume | destroy

Configuration and secrets come from .env: GCP_PROJECT_ID, GCP_BILLING_ACCOUNT, optionally
GCP_REGION and GCP_SECRETS_VERSION, the model keys and settings. Secrets go to Terraform as
environment variables and are never printed.

deploy: creates the project if needed, builds and pushes the three images (tag: the commit),
applies the Terraform, then runs the setup job (migrations, demo orders, knowledge index).
"""

import json
import os
import subprocess
import sys
from pathlib import Path

from dotenv import dotenv_values

ROOT = Path(__file__).resolve().parents[2]
TERRAFORM_DIR = ROOT / "infra" / "gcp"
DOCKER_CONFIG = ROOT / ".cache" / "gcp-docker"  # not the user's: no Docker Desktop helper
# Settings the hosted API takes over from .env as they are; the rest is set in run.tf.
MODEL_SETTINGS = [
    "CA_LLM_BASE_URL",
    "CA_LLM_MODEL",
    "CA_LLM_SESSION_HEADER",
    "CA_LLM_USER_AGENT",
    "CA_LLM_REASONING_EFFORT",
    "CA_ROUTER_LLM_MODEL",
    "CA_ROUTER_REASONING_EFFORT",
    "CA_ANSWER_LLM_MODEL",
    "CA_ANSWER_REASONING_EFFORT",
    "CA_OPENAI_BASE_URL",
    "CA_EMBEDDING_MODEL",
    "CA_EMBEDDING_DIMENSIONS",
    "LANGFUSE_PUBLIC_KEY",
    "LANGFUSE_BASE_URL",
]
# What goes into the images: a change here without a commit gets a "-dirty" tag.
IMAGE_SOURCES = [
    "src",
    "prompts",
    "migrations",
    "data",
    "infra",
    "pyproject.toml",
    "uv.lock",
    "Dockerfile",
]


def run(*args: str, env: dict[str, str] | None = None, stdin: str | None = None) -> str:
    result = subprocess.run(
        args, cwd=ROOT, env=env, input=stdin, text=True, check=True, stdout=subprocess.PIPE
    )
    return result.stdout.strip()


def show(*args: str, env: dict[str, str] | None = None) -> None:
    subprocess.run(args, cwd=ROOT, env=env, check=True)


class Deployment:
    def __init__(self) -> None:
        values = {k: v for k, v in dotenv_values(ROOT / ".env").items() if v}
        missing = [
            key
            for key in (
                "GCP_PROJECT_ID",
                "GCP_BILLING_ACCOUNT",
                "CA_LLM_API_KEY",
                "CA_OPENAI_API_KEY",
            )
            if key not in values
        ]
        if missing:
            sys.exit(f"Set {', '.join(missing)} in .env first.")
        self.values = values
        self.project = values["GCP_PROJECT_ID"]
        self.region = values.get("GCP_REGION", "europe-west1")
        self.registry = f"{self.region}-docker.pkg.dev/{self.project}/images"

    def tag(self) -> str:
        commit = run("git", "rev-parse", "--short=12", "HEAD")
        changed = run("git", "status", "--porcelain", "--untracked-files=no", "--", *IMAGE_SOURCES)
        return f"{commit}-dirty" if changed else commit

    def terraform_env(self) -> dict[str, str]:
        langfuse_key = self.values.get("LANGFUSE_SECRET_KEY", "")
        settings = {key: self.values[key] for key in MODEL_SETTINGS if key in self.values}
        return {
            **os.environ,
            "TF_IN_AUTOMATION": "1",
            "TF_VAR_project_id": self.project,
            "TF_VAR_billing_account": self.values["GCP_BILLING_ACCOUNT"],
            "TF_VAR_region": self.region,
            "TF_VAR_image_tag": self.tag(),
            "TF_VAR_model_settings": json.dumps(settings),
            "TF_VAR_llm_api_key": self.values["CA_LLM_API_KEY"],
            "TF_VAR_openai_api_key": self.values["CA_OPENAI_API_KEY"],
            "TF_VAR_langfuse_secret_key": langfuse_key,
            "TF_VAR_langfuse_enabled": "true" if langfuse_key else "false",
            "TF_VAR_secrets_version": self.values.get("GCP_SECRETS_VERSION", "1"),
        }

    def terraform(self, *args: str, env: dict[str, str]) -> None:
        # A fresh token per run: it expires after an hour and is not refreshed.
        token = run("gcloud", "auth", "print-access-token")
        show(
            "terraform",
            f"-chdir={TERRAFORM_DIR}",
            *args,
            env={**env, "GOOGLE_OAUTH_ACCESS_TOKEN": token},
        )
        state = TERRAFORM_DIR / "terraform.tfstate"
        if state.exists():
            state.chmod(0o600)  # holds the generated database and demo passwords

    def ensure_project(self) -> None:
        exists = subprocess.run(
            ["gcloud", "projects", "describe", self.project, "--format=value(projectId)"],
            capture_output=True,
            text=True,
        )
        if exists.returncode != 0:
            print(f"Creating project {self.project} …")
            show("gcloud", "projects", "create", self.project, "--name=NordElektro customer agent")
            show(
                "gcloud",
                "billing",
                "projects",
                "link",
                self.project,
                f"--billing-account={self.values['GCP_BILLING_ACCOUNT']}",
            )
        # Terraform reads the project and its services through these; it cannot enable them.
        show(
            "gcloud",
            "services",
            "enable",
            "cloudresourcemanager.googleapis.com",
            "serviceusage.googleapis.com",
            f"--project={self.project}",
        )

    def push_images(self, tag: str) -> None:
        DOCKER_CONFIG.mkdir(parents=True, exist_ok=True)
        (DOCKER_CONFIG / "config.json").write_text("{}")
        docker = {**os.environ, "DOCKER_CONFIG": str(DOCKER_CONFIG)}
        run(
            "docker",
            "login",
            "--username=oauth2accesstoken",
            "--password-stdin",
            f"https://{self.region}-docker.pkg.dev",
            env=docker,
            stdin=run("gcloud", "auth", "print-access-token"),
        )
        images = {
            "app": ["."],
            "keycloak": ["-f", "infra/gcp/keycloak/Dockerfile", "infra"],
            "proxy": ["infra/gcp/proxy"],
        }
        for name, context in images.items():
            image = f"{self.registry}/{name}:{tag}"
            print(f"Building {image} …")
            show("docker", "build", "--platform=linux/amd64", "-t", image, *context, env=docker)
            show("docker", "push", "--quiet", image, env=docker)

    def plan(self) -> None:
        self.ensure_project()
        env = self.terraform_env()
        self.terraform("init", "-input=false", env=env)
        self.terraform("plan", "-input=false", env=env)

    def deploy(self) -> None:
        self.ensure_project()
        env = self.terraform_env()
        self.terraform("init", "-input=false", env=env)
        # The registry first: the service needs the images in it.
        self.terraform(
            "apply",
            "-input=false",
            "-auto-approve",
            "-target=google_artifact_registry_repository.images",
            env=env,
        )
        self.push_images(env["TF_VAR_image_tag"])
        self.terraform("apply", "-input=false", "-auto-approve", env=env)
        print("Running the setup job (migrations, demo orders, knowledge index) …")
        show(
            "gcloud",
            "run",
            "jobs",
            "execute",
            "setup",
            "--wait",
            f"--region={self.region}",
            f"--project={self.project}",
        )
        self.terraform("output", env=env)

    def activation(self, policy: str) -> None:
        show(
            "gcloud",
            "sql",
            "instances",
            "patch",
            "agent-db",
            f"--activation-policy={policy}",
            f"--project={self.project}",
            "--quiet",
        )

    def destroy(self) -> None:
        env = self.terraform_env()
        self.terraform("destroy", "-input=true", env=env)
        print(f"The project stays (empty). Delete it with: gcloud projects delete {self.project}")


def main() -> None:
    command = sys.argv[1] if len(sys.argv) > 1 else ""
    deployment = Deployment()
    match command:
        case "plan":
            deployment.plan()
        case "deploy":
            deployment.deploy()
        case "pause":
            deployment.activation("NEVER")
        case "resume":
            deployment.activation("ALWAYS")
        case "destroy":
            deployment.destroy()
        case _:
            sys.exit(__doc__)


if __name__ == "__main__":
    main()
