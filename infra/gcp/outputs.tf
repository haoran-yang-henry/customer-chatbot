output "url" {
  description = "The backend's address: the frontend's backend, Keycloak's hostname."
  value       = local.url
}

output "registry" {
  value = local.registry
}

output "workload_identity_provider" {
  description = "The deploy job's workload_identity_provider (.github/workflows/ci.yml)."
  value       = google_iam_workload_identity_pool_provider.github.name
}

output "deployer" {
  description = "The deploy job's service_account (.github/workflows/ci.yml)."
  value       = google_service_account.deployer.email
}

output "demo_password_command" {
  description = "Prints the demo users' password (in your terminal only)."
  value       = "gcloud secrets versions access latest --secret demo-password --project ${var.project_id}"
}

output "owner_password_command" {
  description = "Prints the password of the owner's account, owner (in your terminal only)."
  value       = "gcloud secrets versions access latest --secret owner-password --project ${var.project_id}"
}

output "realm_sha" {
  description = "The realm the database holds; deploy.py re-imports when its sources differ."
  value       = var.realm_sha
}
