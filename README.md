# Lumina

**Lumina is an AI-powered chat workspace for image understanding, image editing, image generation, and structured technical answers.** It combines a React interface with a Flask/Python backend, computer-vision utilities, and configurable AI services.

<p align="center">
  <a href="https://lumina-fbhi.onrender.com">Live demo</a> ·
  <a href="https://github.com/Mohan2618/Lumina">Source code</a>
</p>

> **Deployment note:** The live demo is hosted on Render and may take time to wake up. AI features require valid provider credentials and may be affected by provider quotas, network availability, and service configuration.

## Contents

- [Highlights](#highlights)
- [Technology stack](#technology-stack)
- [How Lumina works](#how-lumina-works)
- [Repository structure](#repository-structure)
- [Requirements](#requirements)
- [Local setup](#local-setup)
- [Environment configuration](#environment-configuration)
- [Run Lumina](#run-lumina)
- [Docker and deployment](#docker-and-deployment)
- [Testing and quality checks](#testing-and-quality-checks)
- [Security and privacy notes](#security-and-privacy-notes)
- [Troubleshooting](#troubleshooting)
- [Contributing](#contributing)
- [Author](#author)

## Highlights

### AI chat and technical responses

- Natural-language conversations through the Lumina chat interface.
- Structured Markdown rendering for headings, lists, links, blockquotes, tables, and fenced code.
- Code blocks with language labels, syntax highlighting, copy and save actions, and browser-sandboxed previews for supported web formats.
- Table-copy support for moving tabular output into spreadsheets and editors.
- Task-specific progress indicators for chat, coding, structured responses, image editing, and image generation.

Code execution is deliberately limited: browser previews are intended for supported HTML/SVG/CSS/JavaScript content in a sandbox. Lumina does not provide a general-purpose server-side terminal for arbitrary generated code.

### Image understanding and editing

- Image upload and image analysis through natural-language prompts.
- Common deterministic image operations, including grayscale, invert, sepia, blur, sharpen, flip, and rotation, alongside additional computer-vision filters and effects.
- Color analysis, detection and edge-processing utilities.
- Creative effects and selected restoration/enhancement operations implemented in the image-processing modules.
- AI image generation through the configured image-generation provider.
- Multi-step natural-language editing where supported by the backend.
- Image result preview, download/copy/save actions, before-and-after comparison, and undo/redo editing history in the interface.
- 4K desktop wallpaper fitting: creates a 3840 × 2160 canvas using the source image with a blurred background fill, rather than stretching the source to fill the screen.

**Medical-image caution:** Image-processing or AI descriptions of medical images are for informational and educational use only. They are not a diagnosis and must not replace a qualified clinician's assessment.

### Accounts and saved work

- Sign-up and sign-in.
- Password reset using a one-time verification code when email delivery is configured.
- Authenticated conversation history and saved-image gallery.
- Pinned saved conversations and profile/account controls.
- Database-backed account data when PostgreSQL is configured, with SQLite available for local development.
- Guest message limits and browser-local state for selected guest experiences.

Account, history, and gallery features depend on the relevant backend configuration and the user's authentication state.

## Technology stack

| Area | Technologies |
|---|---|
| Frontend | React 18, Vite, JavaScript, CSS, Lucide icons |
| Backend | Python 3.10, Flask, Werkzeug, Gunicorn |
| Image processing | OpenCV, OpenCV-contrib, Pillow, NumPy |
| AI integrations | Google Gemini; optional Anthropic Claude integration |
| Image generation | Configurable provider integration; Cloudflare Workers AI is the configured example in `.env.example`, with Hugging Face support in the service |
| Authentication and storage | Flask sessions, PostgreSQL via `psycopg2`, SQLite fallback |
| Email | Brevo API for OTP email delivery |
| Deployment | Docker, Render |
| Checks | Python `unittest`, frontend Vite production build, GitHub Actions |

## How Lumina works

1. The user sends a message and may attach one or more images.
2. The React interface sends the request to Flask API routes.
3. The backend determines whether to answer conversationally, call a configured AI service, generate an image, or apply an image-processing operation.
4. Flask returns text and any available image or conversation state.
5. The frontend renders the result as a message, code artifact, table, image result, or other supported structured content.
6. Authenticated history and gallery operations are persisted through the configured backend storage; selected guest state may remain in the browser.

Provider calls can fail because of missing API keys, quota limits, rate limits, or network issues. A successful frontend build does not validate external provider credentials.

## Repository structure

```text
Lumina/
├── app.py                         # Flask application entry point
├── lumina/
│   ├── core.py                    # Flask app and shared service setup
│   ├── config.py                  # AI clients and application settings
│   ├── routes.py                  # Chat, image, health, auth and history routes
│   ├── image_processing/
│   │   ├── basic.py               # Basic image transformations
│   │   ├── color.py               # Color utilities
│   │   ├── creative.py            # Creative image effects
│   │   ├── detection.py           # Detection and analysis operations
│   │   ├── filters.py             # Image filters
│   │   ├── medical.py             # Medical-image processing helpers
│   │   ├── advanced.py            # Advanced processing helpers
│   │   └── processor.py           # Processing dispatcher
│   ├── prompts/
│   │   └── system_prompt.py       # Assistant instructions
│   ├── services/
│   │   ├── ai_service.py          # AI response orchestration
│   │   ├── image_generation.py    # Image generation provider integration
│   │   ├── auth_db.py             # Account, conversation and gallery persistence
│   │   └── email_service.py       # OTP email delivery
│   └── utils/                     # Authentication and image utilities
├── frontend/
│   ├── src/
│   │   ├── App.jsx                # React application and chat workspace
│   │   ├── AuthModal.jsx          # Authentication UI
│   │   ├── main.jsx               # React entry point
│   │   └── styles.css             # Frontend styling
│   ├── package.json
│   ├── vite.config.js
│   └── README.md                  # Frontend-specific instructions
├── templates/                     # Legacy/fallback Flask templates
├── static/                        # Static assets and response enhancements
├── tests/                         # Backend unit tests
├── docs/REACT_FEATURE_PARITY.md   # Frontend feature audit and known gaps
├── Dockerfile                     # Builds React and packages Flask
├── requirements.txt               # Python dependencies
├── .env.example                   # Environment variable template
└── README.md
```

## Requirements

- Python 3.10 (matches the Docker image).
- Node.js 20 and npm for frontend development/builds.
- Git.
- API credentials for the AI or image-generation features you want to use.
- PostgreSQL for persistent production account data (recommended for deployment); SQLite is supported as a local fallback.

## Local setup

### 1. Clone the repository

```bash
git clone https://github.com/Mohan2618/Lumina.git
cd Lumina
```

### 2. Create and activate a Python virtual environment

**Windows PowerShell**

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

If PowerShell blocks activation due to its execution policy, follow your machine's Python/PowerShell guidance or activate the environment from Command Prompt instead.

**macOS/Linux**

```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 3. Install backend dependencies

```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### 4. Install frontend dependencies

Open a second terminal or continue after activating the Python environment:

```bash
cd frontend
npm install
cd ..
```

### 5. Configure environment variables

Copy `.env.example` to `.env` in the repository root, then fill in only the credentials for services you intend to use. See [Environment configuration](#environment-configuration) below.

### 6. Start the application

Start Flask from the repository root:

```powershell
python app.py
```

In a second terminal, start Vite:

```powershell
cd frontend
npm run dev
```

Open the development URL printed by Vite (typically `http://localhost:5173`). The Vite configuration proxies supported backend paths, including `/api`, `/process`, and `/health`, to Flask on port `7860`.

For a production-style frontend build:

```powershell
cd frontend
npm run build
cd ..
```

The generated assets are written to `frontend/dist/`. The Dockerfile builds these assets and copies them into the final image. For local production-style serving, ensure the frontend build exists before starting the backend.

## Environment configuration

The application reads environment variables from the process environment; `app.py` also loads a root `.env` file for local development. Never commit real credentials.

| Variable | Purpose | Required when |
|---|---|---|
| `GEMINI_API_KEY` | Google Gemini client used by the AI service | Using Gemini-powered features |
| `ANTHROPIC_API_KEY` | Credentials for optional Claude integration | Claude is enabled |
| `ENABLE_CLAUDE` | Enables optional Claude client initialization when set to `true` | Using Claude |
| `IMAGE_PROVIDER` | Selects the image-generation provider | Image generation |
| `CLOUDFLARE_ACCOUNT_ID` | Cloudflare account for Workers AI | `IMAGE_PROVIDER=cloudflare` |
| `CLOUDFLARE_API_TOKEN` | Cloudflare API token | `IMAGE_PROVIDER=cloudflare` |
| `IMAGE_GENERATION_STEPS` | Configures generation steps where supported by the provider | Optional; example default is `4` |
| `HF_TOKEN` | Hugging Face token for the supported Hugging Face path | When that provider/model requires it |
| `IMAGE_GENERATION_MODEL` | Image-generation model identifier | Optional/provider-specific |
| `IMAGE_GENERATION_GUIDANCE` | Provider-specific generation guidance value | Optional/provider-specific |
| `BREVO_API_KEY` | Brevo API key for transactional email | Sending OTP emails through Brevo |
| `BREVO_FROM_EMAIL` | Verified sender address | Sending OTP emails through Brevo |
| `BREVO_FROM_NAME` | Sender display name | Optional email branding |
| `DATABASE_URL` | PostgreSQL connection string | Persistent PostgreSQL storage |
| `FLASK_SECRET_KEY` | Stable Flask session-signing secret | **Required in production** |
| `GEMINI_TIMEOUT` | Gemini request timeout in seconds | Optional; defaults to `45` |

The exact provider options and defaults are defined in `lumina/config.py`, `lumina/services/image_generation.py`, and `.env.example`. Do not assume a provider is available until its credentials and deployment configuration have been tested.

### Production secrets and database

- Set `FLASK_SECRET_KEY` to a random secret of at least 32 characters. The production application refuses to start if this requirement is not met. Use the same secret across instances and restarts so sessions remain valid.
- Set `DATABASE_URL` to a PostgreSQL connection string for persistent deployed account, conversation, and gallery data. Without it, the application uses its SQLite fallback, which may not be suitable for ephemeral container storage.
- Existing records in a previous SQLite database are not automatically migrated to PostgreSQL. Plan and verify any migration explicitly.
- Keep API keys, email credentials, database passwords, and session secrets out of source control and logs.

## Docker and deployment

The root `Dockerfile` uses a multi-stage build:

1. Node.js 20 installs frontend dependencies and runs `npm run build`.
2. Python 3.10 installs `requirements.txt`.
3. The React `dist/` output is copied into the application image.
4. Gunicorn starts the Flask application on port `7860`.

Build and run locally:

```bash
docker build -t lumina .
docker run --rm -p 7860:7860 --env-file .env lumina
```

Configure production environment variables in the hosting provider's dashboard. For Render, connect the GitHub repository, use the Docker runtime, and provide the required secrets and persistent database configuration. After a deployment, inspect the service logs and test the live site; a successful image build alone does not guarantee valid runtime configuration or AI-provider access.

Live demo: https://lumina-fbhi.onrender.com

## Testing and quality checks

### Frontend production build

```bash
cd frontend
npm install
npm run build
cd ..
```

### Backend tests

From the repository root with the virtual environment activated:

```bash
python -m unittest discover -s tests -p "test_*.py" -v
```

### Python syntax check

```bash
python -m compileall -q app.py lumina
```

The repository includes tests for image processing, image validation, guest limits, and response UI behavior. GitHub Actions also runs frontend build, Python syntax, and backend unit-test checks on configured branch/pull-request events. Run these checks after changes and manually test integrations that require credentials.

### Suggested manual smoke test

- [ ] Homepage loads without a server error.
- [ ] Send a plain-text message and verify the response.
- [ ] Render code in Python, Java, JavaScript/TypeScript, SQL, and another language; test copy and save.
- [ ] Verify wide code can still be scrolled when its scrollbar is hidden.
- [ ] Upload a supported image and test a basic edit such as grayscale or rotate.
- [ ] Test image generation with valid provider credentials and check provider quota/error handling.
- [ ] Sign up, sign in, sign out, and refresh the page.
- [ ] Verify saved conversation history and active-conversation restoration.
- [ ] Test gallery save/download/delete and image undo/redo/compare.
- [ ] Test password reset only when Brevo and a verified sender are configured.
- [ ] Check Render logs for missing packages, missing environment variables, or provider errors.

## Security and privacy notes

- Treat all AI-generated content as untrusted input.
- Browser code previews should remain sandboxed; do not enable arbitrary server-side execution of generated code.
- Do not commit `.env`, access tokens, secrets, or real database connection strings.
- Validate uploaded images and keep request-size limits in place.
- Medical-image features are not a substitute for professional medical advice.
- Review authentication, session, rate-limit, and sharing behavior before exposing a deployment to sensitive or high-volume use.

## Troubleshooting

| Symptom | What to check |
|---|---|
| `ModuleNotFoundError` during startup | Confirm the dependency is in root `requirements.txt` and the deployment installs that file; trigger a fresh build if necessary. |
| Flask refuses to start in production | Set `FLASK_SECRET_KEY` to a stable value at least 32 characters long. |
| AI response or image generation fails | Check the relevant API key, selected provider, model configuration, network logs, and provider quota/rate limits. |
| Password-reset email does not arrive | Verify `BREVO_API_KEY`, the verified sender address, and email-service logs. |
| Saved account history is missing | Confirm the user is signed in, `DATABASE_URL` points to the expected database, and the history API request succeeds. |
| Frontend changes do not appear | Confirm the latest commit deployed, inspect Render build logs, and hard-refresh the browser to bypass cached assets. |
| Homepage returns HTTP 500 | Read the newest server traceback in Render logs; the browser's generic error page does not identify the underlying exception. |

## Contributing

1. Create a branch for your change.
2. Keep secrets and local environment files out of commits.
3. Run the frontend build, Python syntax check, and backend tests.
4. Test the relevant user workflow manually, especially when changing authentication, history, image handling, or deployment behavior.
5. Update this README or the relevant documentation when commands, configuration, or user-facing features change.

For details about the React frontend and feature-parity tracking, see [`frontend/README.md`](frontend/README.md) and [`docs/REACT_FEATURE_PARITY.md`](docs/REACT_FEATURE_PARITY.md).

## Author

**Mohan Lingabathina**

- GitHub: https://github.com/Mohan2618
- Project: https://github.com/Mohan2618/Lumina

## License

No explicit open-source license is currently documented in this repository. Unless a license is added, reuse and redistribution are subject to the default copyright rules.
