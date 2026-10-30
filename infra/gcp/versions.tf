terraform {
  # 1.11: write-only arguments, so the model API keys never enter the state.
  required_version = ">= 1.11"

  # In Cloud Storage, shared by `make gcp-deploy` and the deploy job in CI. deploy.py creates
  # the bucket (<project>-tfstate) before the first init and passes its name.
  backend "gcs" {
    prefix = "terraform"
  }

  required_providers {
    google = {
      source  = "hashicorp/google"
      version = "~> 7.46"
    }
    random = {
      source  = "hashicorp/random"
      version = "~> 3.9"
    }
  }
}

provider "google" {
  project = var.project_id
  region  = var.region
  # The budget API needs a quota project when called with user credentials.
  user_project_override = true
  billing_project       = var.project_id
}
