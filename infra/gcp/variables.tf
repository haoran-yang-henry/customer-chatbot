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

# The secrets come from .env through deploy.py. Write-only: Secret Manager stores them, the
# Terraform state does not. Bump secrets_version to send changed values.
variable "llm_api_key" {
  type      = string
  sensitive = true
  ephemeral = true
}

variable "openai_api_key" {
  type      = string
  sensitive = true
  ephemeral = true
}

variable "langfuse_secret_key" {
  description = "Empty: no Langfuse traces from the hosted backend."
  type        = string
  default     = ""
  sensitive   = true
  ephemeral   = true
}

variable "langfuse_enabled" {
  description = "Whether a Langfuse secret key is given (an ephemeral value cannot decide that)."
  type        = bool
  default     = false
}

variable "secrets_version" {
  type    = number
  default = 1
}
