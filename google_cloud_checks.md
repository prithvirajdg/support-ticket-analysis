# Google Cloud Access Checks

Run these checks in Cloud Shell before starting the pipeline to catch any
access or configuration issues early.

Run checks 1–5 first (fast, no API calls), then check 6 last to validate
end-to-end. If check 6 fails, the error message will directly tell you which
of the above issues it is.

---

## 1. Auth — confirm you're logged in as the right account

```bash
gcloud auth list
```

Look for an `*` next to your account. If it's a service account or the wrong
email, that's the issue.

---

## 2. Project — confirm you're pointing at the right project

```bash
gcloud config get-value project
```

If blank or wrong:

```bash
gcloud config set project YOUR_PROJECT_ID
```

---

## 3. Billing — confirm billing is enabled (Vertex AI won't work without it)

```bash
gcloud beta billing projects describe $(gcloud config get-value project)
```

Look for `billingEnabled: true`. If `false`, billing needs to be attached to the
project — this requires org admin action.

---

## 4. Vertex AI API — confirm it's enabled

```bash
gcloud services list --enabled --filter="name:aiplatform.googleapis.com"
```

If no output, enable it:

```bash
gcloud services enable aiplatform.googleapis.com
```

---

## 5. IAM — confirm your account has the right role on the project

```bash
gcloud projects get-iam-policy $(gcloud config get-value project) \
  --flatten="bindings[].members" \
  --format="table(bindings.role,bindings.members)" \
  --filter="bindings.members:$(gcloud config get-value account)"
```

You need at least one of these roles in the output:

| Role | Level |
|------|-------|
| `roles/aiplatform.user` | Minimum needed |
| `roles/aiplatform.admin` | Full Vertex AI access |
| `roles/editor` | Project-wide editor |
| `roles/owner` | Project owner |

If none of these appear, you don't have Vertex AI access and need to ask your
org admin to grant `roles/aiplatform.user`.

---

## 6. Gemini model — confirm end-to-end access with a test call

```bash
python3 - <<'EOF'
import vertexai
from vertexai.generative_models import GenerativeModel, GenerationConfig
import os

project = os.popen("gcloud config get-value project").read().strip()
print(f"Project: {project}")

vertexai.init(project=project, location="us-central1")

model = GenerativeModel("gemini-2.0-flash-001")
response = model.generate_content(
    "Say hello in one word.",
    generation_config=GenerationConfig(max_output_tokens=10, temperature=0.0)
)
print("SUCCESS:", response.text)
EOF
```

If this prints `SUCCESS: Hello` (or similar), everything is wired up correctly
and the pipeline is ready to run.

---

## What each failure means

| Error message | Cause | Fix |
|--------------|-------|-----|
| `PERMISSION_DENIED: Vertex AI API has not been enabled` | API not enabled | Run `gcloud services enable aiplatform.googleapis.com` |
| `PERMISSION_DENIED: caller does not have permission` | Missing IAM role | Ask org admin for `roles/aiplatform.user` |
| `FAILED_PRECONDITION: billing not enabled` | No billing on project | Org admin needs to attach a billing account |
| `NOT_FOUND: model not found` | Model not available in region | Try `--region us-east4` or `asia-south1` |
| `RESOURCE_EXHAUSTED: quota exceeded` | API quota hit | Reduce `--workers` to `2`, or request quota increase |
| `google.auth.exceptions.DefaultCredentialsError` | Not authenticated | Run `gcloud auth application-default login` |
