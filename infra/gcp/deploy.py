"""Deploy the backend to Google Cloud (infra/gcp/README.md).

    uv run python infra/gcp/deploy.py plan | deploy | release | secrets | pause | resume | destroy

deploy (your gcloud login, settings from .env): creates the project and the state bucket if
needed, adds missing model keys to Secret Manager, saves the settings for CI, then releases.
release (the deploy job in CI, as the deployer; also the end of deploy): builds and pushes the
three images (tag: the commit), applies the Terraform, runs the setup job (migrations, demo
orders, knowledge index) and waits until the backend is ready. It needs no .env and no key:
GCP_PROJECT_ID comes from the environment, the rest from the settings deploy saved.
secrets: adds the model and Langfuse keys from .env as new versions (a new key).

.env holds GCP_PROJECT_ID, GCP_BILLING_ACCOUNT, optionally GCP_REGION, the model keys and
settings. The keys go to Secret Manager through gcloud, never to Terraform, and are never
printed.
"""

import json
import os
import subprocess
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from urllib.error import URLError
from urllib.request import urlopen

from dotenv import dotenv_values

ROOT = Path(__file__).resolve().parents[2]
TERRAFORM_DIR = ROOT / "infra" / "gcp"
LOCAL_STATE = TERRAFORM_DIR / "terraform.tfstate"  # before the state moved to the bucket
DOCKER_CONFIG = ROOT / ".cache" / "gcp-docker"  # not the user's: no Docker Desktop helper
SETTINGS_OBJECT = "deploy-settings.json"  # beside the state: what release needs from .env
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
# Secret Manager secret ← .env key. Terraform creates the secrets; the values bypass it.
KEYS = {
    "llm-api-key": "CA_LLM_API_KEY",
    "openai-api-key": "CA_OPENAI_API_KEY",
    "langfuse-secret-key": "LANGFUSE_SECRET_KEY",
}
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
READY_TIMEOUT = 180  # seconds; a cold start waits about 15 s for Keycloak


def run(*args: str, env: dict[str, str] | None = None, stdin: str | None = None) -> str:
    result = subprocess.run(
        args, cwd=ROOT, env=env, input=stdin, text=True, check=True, stdout=subprocess.PIPE
    )
    return result.stdout.strip()


def show(*args: str, env: dict[str, str] | None = None) -> None:
    subprocess.run(args, cwd=ROOT, env=env, check=True)


def state_bucket(project: str) -> str:
    return f"{project}-tfstate"


@dataclass(frozen=True)
class Settings:
    """What a release needs: no key among it, so it can be saved for CI."""

    project: str
    billing_account: str
    region: str
    model_settings: dict[str, str]
    langfuse_enabled: bool

    @classmethod
    def from_env_file(cls, values: dict[str, str]) -> "Settings":
        missing = [key for key in ("GCP_PROJECT_ID", "GCP_BILLING_ACCOUNT") if key not in values]
        if missing:
            sys.exit(f"Set {', '.join(missing)} in .env first.")
        return cls(
            project=values["GCP_PROJECT_ID"],
            billing_account=values["GCP_BILLING_ACCOUNT"],
            region=values.get("GCP_REGION", "europe-west1"),
            model_settings={key: values[key] for key in MODEL_SETTINGS if key in values},
            langfuse_enabled="LANGFUSE_SECRET_KEY" in values,
        )

    @classmethod
    def saved(cls, project: str) -> "Settings":
        saved = run("gcloud", "storage", "cat", f"gs://{state_bucket(project)}/{SETTINGS_OBJECT}")
        return cls(project=project, **json.loads(saved))

    def save(self) -> None:
        settings = asdict(self)
        del settings["project"]  # the object's address
        run(
            "gcloud",
            "storage",
            "cp",
            "-",
            f"gs://{state_bucket(self.project)}/{SETTINGS_OBJECT}",
            stdin=json.dumps(settings, indent=2),
        )


class Deployment:
    def __init__(self, settings: Settings, keys: dict[str, str]) -> None:
        self.settings = settings
        self.keys = keys  # secret → value; empty in CI
        self.project = settings.project
        self.region = settings.region
        self.bucket = state_bucket(settings.project)
        self.registry = f"{self.region}-docker.pkg.dev/{self.project}/images"
        self.env = self.terraform_env()

    def tag(self) -> str:
        commit = run("git", "rev-parse", "--short=12", "HEAD")
        changed = run("git", "status", "--porcelain", "--untracked-files=no", "--", *IMAGE_SOURCES)
        return f"{commit}-dirty" if changed else commit

    def terraform_env(self) -> dict[str, str]:
        return {
            **os.environ,
            "TF_IN_AUTOMATION": "1",
            "TF_VAR_project_id": self.project,
            "TF_VAR_billing_account": self.settings.billing_account,
            "TF_VAR_region": self.region,
            "TF_VAR_image_tag": self.tag(),
            "TF_VAR_model_settings": json.dumps(self.settings.model_settings),
            "TF_VAR_langfuse_enabled": "true" if self.settings.langfuse_enabled else "false",
            "TF_VAR_state_bucket": self.bucket,
        }

    def authorized(self) -> dict[str, str]:
        # A fresh token per run: it expires after an hour and is not refreshed.
        return {
            **self.env,
            "GOOGLE_OAUTH_ACCESS_TOKEN": run("gcloud", "auth", "print-access-token"),
        }

    def terraform(self, *args: str) -> None:
        show("terraform", f"-chdir={TERRAFORM_DIR}", *args, env=self.authorized())

    def output(self, name: str) -> str:
        return run(
            "terraform", f"-chdir={TERRAFORM_DIR}", "output", "-raw", name, env=self.authorized()
        )

    def init(self) -> None:
        args = ["init", "-input=false", f"-backend-config=bucket={self.bucket}"]
        migrate = LOCAL_STATE.exists() and LOCAL_STATE.stat().st_size > 0
        if migrate:
            args += ["-migrate-state", "-force-copy"]  # once: the local state into the bucket
        self.terraform(*args)
        if migrate:
            # Terraform empties the local file and keeps the old state as the backup.
            LOCAL_STATE.unlink()
            backup = LOCAL_STATE.with_name("terraform.tfstate.backup")
            backup.chmod(0o600)
            print(
                f"The state is now in gs://{self.bucket}. {backup.name} still holds the old "
                "state with the generated passwords: delete it once a plan works."
            )

    def apply(self, *targets: str) -> None:
        self.terraform("apply", "-input=false", "-auto-approve", *(f"-target={t}" for t in targets))

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
                f"--billing-account={self.settings.billing_account}",
            )
        # Terraform reads the project, its services and its billing account through these, with
        # the project as quota project; it cannot enable them.
        show(
            "gcloud",
            "services",
            "enable",
            "cloudbilling.googleapis.com",
            "cloudresourcemanager.googleapis.com",
            "serviceusage.googleapis.com",
            "storage.googleapis.com",
            f"--project={self.project}",
        )
        self.ensure_state_bucket()

    def ensure_state_bucket(self) -> None:
        exists = subprocess.run(
            ["gcloud", "storage", "buckets", "describe", f"gs://{self.bucket}"],
            capture_output=True,
            text=True,
        )
        if exists.returncode == 0:
            return
        print(f"Creating the state bucket gs://{self.bucket} …")
        show(
            "gcloud",
            "storage",
            "buckets",
            "create",
            f"gs://{self.bucket}",
            f"--project={self.project}",
            f"--location={self.region}",
            "--uniform-bucket-level-access",
            "--public-access-prevention",
        )
        # Every state version is kept: an apply that went wrong can be rolled back.
        show("gcloud", "storage", "buckets", "update", f"gs://{self.bucket}", "--versioning")

    def has_version(self, secret: str) -> bool:
        return bool(
            run(
                "gcloud",
                "secrets",
                "versions",
                "list",
                secret,
                "--filter=state=ENABLED",
                "--limit=1",
                "--format=value(name)",
                f"--project={self.project}",
            )
        )

    def add_keys(self, *, only_missing: bool) -> None:
        secrets = ["llm-api-key", "openai-api-key"]
        if self.settings.langfuse_enabled:
            secrets.append("langfuse-secret-key")
        for secret in secrets:
            if only_missing and self.has_version(secret):
                continue
            if secret not in self.keys:
                sys.exit(f"Set {KEYS[secret]} in .env first.")
            run(
                "gcloud",
                "secrets",
                "versions",
                "add",
                secret,
                "--data-file=-",
                f"--project={self.project}",
                stdin=self.keys[secret],
            )

    def push_images(self) -> None:
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
            image = f"{self.registry}/{name}:{self.env['TF_VAR_image_tag']}"
            print(f"Building {image} …")
            show("docker", "build", "--platform=linux/amd64", "-t", image, *context, env=docker)
            show("docker", "push", "--quiet", image, env=docker)

    def wait_until_ready(self, url: str) -> None:
        ready = f"{url}/health/ready"  # url: the https address from the Terraform output
        deadline = time.monotonic() + READY_TIMEOUT
        while time.monotonic() < deadline:
            try:
                with urlopen(ready, timeout=30) as response:  # noqa: S310
                    if response.status == 200:
                        print(f"Ready: {url}")
                        return
            except (URLError, TimeoutError):
                pass  # 503 until the database answers, or the instance is still starting
            time.sleep(5)
        sys.exit(f"{ready} did not answer 200 within {READY_TIMEOUT} s.")

    def plan(self) -> None:
        self.ensure_project()
        self.init()
        self.terraform("plan", "-input=false")

    def deploy(self) -> None:
        self.ensure_project()
        self.init()
        # Before the images and the first revision: the registry, the secrets and their keys.
        self.apply(
            "google_artifact_registry_repository.images", "google_secret_manager_secret.secrets"
        )
        self.add_keys(only_missing=True)
        self.settings.save()
        self.release()

    def release(self) -> None:
        self.push_images()
        self.apply()
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
        self.wait_until_ready(self.output("url"))
        self.terraform("output")

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
        self.init()
        self.terraform("destroy", "-input=true")
        print(
            f"The project stays (empty), and so does gs://{self.bucket}. Delete both with: "
            f"gcloud projects delete {self.project}"
        )


def release_in_ci() -> None:
    project = os.environ.get("GCP_PROJECT_ID")
    if not project:
        sys.exit("Set GCP_PROJECT_ID.")
    deployment = Deployment(Settings.saved(project), keys={})
    deployment.init()
    deployment.release()


def main() -> None:
    command = sys.argv[1] if len(sys.argv) > 1 else ""
    if command == "release":
        release_in_ci()
        return
    values = {k: v for k, v in dotenv_values(ROOT / ".env").items() if v}
    keys = {secret: values[key] for secret, key in KEYS.items() if key in values}
    deployment = Deployment(Settings.from_env_file(values), keys)
    match command:
        case "plan":
            deployment.plan()
        case "deploy":
            deployment.deploy()
        case "secrets":
            deployment.add_keys(only_missing=False)
            print("New versions added. A running instance keeps the old keys; new ones read these.")
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
