---
title: Lumina
emoji: 🖼️
colorFrom: blue
colorTo: indigo
sdk: docker
pinned: false
---

## Links

**Live Demo:** https://lumina-fbhi.onrender.com

**Source Code:** https://github.com/Mohan2618/Lumina

# Lumina - AI-Powered Image Processing Chatbot

## Overview

Lumina is an AI-powered image processing chatbot that combines computer vision techniques with generative AI to provide an interactive platform for image analysis, enhancement, editing, and generation. Users can communicate with the application using natural language and perform image processing operations through a web interface.

The application is developed using Flask, OpenCV, Pillow, and NumPy and integrates Google Gemini for intelligent image understanding and conversational responses. It is deployed on Render using Docker.

---

## User-Facing Branding

Lumina presents a consistent **Lumina AI** identity throughout the application. User-facing status messages, notifications, assistant responses, and service errors do not expose the names of underlying AI models or providers. Provider integrations remain implementation details and do not change the Lumina user experience.

---

## Features

- AI-powered conversational image assistant
- Image analysis and description
- AI image generation
- Image enhancement and restoration
- Face detection
- Background removal
- Edge detection
- Color analysis
- Image filtering
- Cartoon, sketch, and watercolor effects
- Noise reduction
- Super resolution
- Histogram equalization
- Medical image analysis with appropriate disclaimer
- Session-based conversation management
- Email OTP delivery for password reset
- Natural-language multi-step image editing
- Local image-edit undo/redo history
- Before/after image comparison
- 4K desktop wallpaper fitting
- Structured code blocks with **Copy** and **Save** controls
- Safe client-side **Run** controls for HTML, SVG, CSS, and JavaScript response blocks
- Markdown table rendering with **Copy table** support
- Project/folder structures and diagrams preserved as copyable text artifacts
- Task-specific progress states for chat, coding, structured information, image editing, and image generation
- Responsive response artifacts for desktop and mobile layouts

---

## Structured AI Responses

Lumina now treats assistant responses as structured content instead of displaying every response as plain text.

### Code

Fenced Markdown code blocks such as:

```python
print("Hello from Lumina")
```

are displayed as dedicated code artifacts with:

- language label
- syntax-preserving formatting
- Copy button
- Save button
- Run button for supported browser-safe languages

The browser Run action is sandboxed and is currently available for HTML, SVG, CSS, and JavaScript. It does not execute Python, shell commands, PowerShell, or server-side code.

### Tables

Standard Markdown tables are converted into responsive table cards with a **Copy table** action. The copied table uses tab-separated values so it can be pasted into spreadsheets and editors.

### Project Structures and Diagrams

Folder structures, ASCII diagrams, Mermaid source, architecture sketches, and other structured text inside fenced code blocks are preserved as copyable artifacts instead of being flattened into normal paragraphs.

### Security

Assistant-generated HTML is not trusted as executable page content. Code artifacts are escaped before rendering. The optional Run action executes only the selected browser-safe artifact inside a sandboxed iframe.

---

## Task-Specific Response Status

Lumina uses separate user-facing progress states based on the type of request rather than using the same generic status for every operation.

| Request type | Example progress states |
|---|---|
| Normal chat | Understanding → Forming → Polishing |
| Coding / project help | Understanding → Structuring → Checking |
| Tables / diagrams / comparisons | Organizing → Formatting → Polishing |
| Image editing | Understanding → Processing → Finalizing |
| Image generation | Planning → Creating → Preparing |

These are interface-level progress indicators. They do not claim to expose hidden model reasoning or private chain-of-thought.

---

## Technology Stack

### Backend

- Python
- Flask

### Computer Vision

- OpenCV
- Pillow (PIL)
- NumPy

### Artificial Intelligence

- Google Gemini API
- Anthropic Claude API
- Brevo Transactional Email API for password-reset OTP delivery

### Frontend

- HTML
- CSS
- JavaScript
- Responsive structured-response rendering

### Deployment

- Render
- Docker
- PostgreSQL for production authentication storage
- SQLite for local-development fallback

---

## Project Structure

```text
Lumina/
│
├── app.py
├── lumina/
│   ├── core.py
│   ├── routes.py
│   ├── image_processing/
│   ├── prompts/
│   ├── services/
│   └── utils/
├── templates/
│   └── index.html
├── static/
│   ├── logo.jpeg
│   └── response_enhancements.js
├── tests/
├── Dockerfile
├── requirements.txt
├── README.md
└── ...
```

---

## Installation

### Clone the Repository

```bash
git clone https://github.com/Mohan2618/Lumina.git
cd Lumina
```

### Create a Virtual Environment

```bash
python -m venv venv
```

Windows PowerShell:

```powershell
venv\Scripts\activate
```

Linux/macOS:

```bash
source venv/bin/activate
```

### Install Dependencies

```bash
pip install -r requirements.txt
```

### Configure Environment Variables

Create a `.env` file in the project root.

For Render, configure these environment variables in the service settings:

```text
GOOGLE_API_KEY=your_google_api_key
ANTHROPIC_API_KEY=your_anthropic_api_key
BREVO_API_KEY=your_brevo_api_key
BREVO_FROM_EMAIL=your_verified_sender_email
BREVO_FROM_NAME=Lumina
DATABASE_URL=your_postgresql_connection_string
```

Never commit real API keys, database connection strings, passwords, or other credentials to GitHub.

### Persistent Authentication Database

Accounts created in an older Render-hosted SQLite database are not automatically migrated into PostgreSQL. After configuring `DATABASE_URL`, create/test an account in the new database or perform a deliberate data migration before testing password reset.

Lumina uses PostgreSQL in production when `DATABASE_URL` is configured. SQLite remains the local-development fallback when it is not set. The application creates the required authentication tables on startup.

Do not commit `DATABASE_URL` or any database password to the repository.

### Run the Application

```bash
python app.py
```

The application will be available at:

```text
http://localhost:7860
```

---

## Deployment

The application is deployed on Render using Docker.

Live Demo:

https://lumina-fbhi.onrender.com

---

## How It Works

1. The user enters a natural-language request and optionally uploads an image.
2. Lumina determines whether the request is conversational, analytical, an image edit, an image-generation task, or another supported operation.
3. AI-assisted requests are sent through Lumina's configured AI service, while deterministic image operations are executed through the image-processing layer.
4. The backend returns the assistant message, optional processed/generated image, conversation history, and current image state.
5. The frontend converts supported Markdown structures into dedicated response artifacts such as code blocks and tables.
6. Code artifacts provide copy/save controls and browser-safe Run controls where supported.
7. Image operations can use local undo/redo history and before/after comparison.
8. Password-reset requests use one-time OTPs stored in the configured authentication database and delivered through Brevo.

---

## Advanced Editing

Lumina supports natural-language multi-step image editing. Users can request several supported edits in a single message, and Lumina can preserve the requested order and apply up to eight operations as one workflow.

The web interface provides local undo/redo controls for image-edit states, with up to 20 states retained per active chat, plus an interactive before/after comparison with a draggable split slider.

Lumina also supports **4K desktop wallpaper fitting**. A request such as "make this image into a desktop 4K wallpaper" preserves the complete source image as the foreground and builds the remaining 16:9 canvas from a softly blurred, enlarged version of the same image. This avoids stretching or cutting the original subject while producing a 3840×2160 result. Center, top, and bottom placement are supported.

---

## Response UI Architecture

The existing Flask template remains the primary application interface. A small response-enhancement layer is loaded from `static/response_enhancements.js` through the Flask response pipeline in `app.py`.

This keeps the existing chat implementation intact while adding:

```text
AI response
    ↓
Markdown / structured-content detection
    ↓
┌───────────────┬───────────────┬────────────────┐
│ Code artifact │ Table artifact│ Normal message │
└───────┬───────┴───────┬───────┴────────────────┘
        ↓               ↓
 Copy / Save       Copy table
        ↓
 Optional sandbox Run
```

The response enhancement layer is intentionally client-side for presentation. It does not change the server's image-processing pipeline.

---

## v1.1 Stability & Reliability

The v1.1 stability work hardens the existing application without removing its current image-processing or response features. It includes safer image upload validation, explicit upload size limits, session-cookie security defaults, removal of the obsolete client-side password-hashing endpoint, and clearer image-generation failure handling. Images are validated by their decoded format and dimensions rather than trusting the uploaded filename or MIME type.

The backend rejects oversized image payloads, malformed/truncated files, unsupported formats, and images exceeding the configured pixel/dimension limits before normal processing. Image-generation failures now return an explicit service-unavailable response instead of pretending that a result was produced.

## Future Enhancements

- Streaming assistant responses
- Rich visual graph and chart rendering
- Mermaid diagram rendering with export support
- Additional AI image-generation models
- Specialized vision models for segmentation and super-resolution
- Model/provider routing and fallback
- Batch image processing
- Cloud image storage
- REST API support
- Additional authentication hardening
- Performance optimization

---

## Learning Outcomes

This project demonstrates practical experience in:

- Flask application development
- Computer vision using OpenCV
- AI model integration
- Prompt engineering
- Structured AI response rendering
- Frontend JavaScript and responsive UI design
- Image processing techniques
- API integration
- Docker-based deployment
- Render deployment

---

## Author

**Mohan Lingabathina**

GitHub: https://github.com/Mohan2618/Lumina

---

## License

This project is intended for educational and research purposes.
