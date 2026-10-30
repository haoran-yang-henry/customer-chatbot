terraform {
  # 1.11: write-only arguments, so the model API keys never enter the state.
  required_version = ">= 1.11"

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
