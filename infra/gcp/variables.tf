variable "project_id" {
  description = "The project this backend gets to itself (created by deploy.py)."
  type        = string
}

variable "billing_account" {
  description = "Billing account of the project; the budget alert is set on it."
  type        = string
}

variable "region" {
  type    = string
  default = "europe-west1"
}

variable "monthly_budget" {
  description = "Budget alert in the billing account's currency: e-mails at 50, 90 and 100 %."
  type        = number
  default     = 20
}

variable "image_tag" {
  description = "Tag of the three images (git commit), pushed by deploy.py before the apply."
  type        = string
}

variable "pages_origin" {
  description = "The browser demo on GitHub Pages: the only origin allowed to call the API."
  type        = string
  default     = "https://haoran-yang-henry.github.io"
}

variable "model_settings" {
  description = "Model and Langfuse settings without secrets (CA_LLM_MODEL, …), taken from .env."
  type        = map(string)
  default     = {}
}

variable "langfuse_enabled" {
  description = "Whether .env has a Langfuse secret key: then the secret exists and is read."
  type        = bool
  default     = false
}

variable "state_bucket" {
  description = "The Terraform state's bucket (created by deploy.py); the deployer writes to it."
  type        = string
}

variable "github_repository" {
  description = "The repository whose main branch deploys (owner/name)."
  type        = string
  default     = "haoran-yang-henry/customer-chatbot"
}

variable "github_repository_id" {
  description = "Its numeric id: a deleted and re-created repository of the same name gets a new one."
  type        = string
  default     = "1416107290"
}
