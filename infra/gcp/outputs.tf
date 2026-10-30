output "url" {
  description = "The backend's address: the frontend's backend, Keycloak's hostname."
  value       = local.url
}

output "registry" {
  value = local.registry
}

output "demo_password_command" {
  description = "Prints the demo users' password (in your terminal only)."
  value       = "gcloud secrets versions access latest --secret demo-password --project ${var.project_id}"
}
