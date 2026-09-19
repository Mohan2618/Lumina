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

Lumina is an AI-powered image processing chatbot that combines computer vision techniques with generative AI to provide an interactive platform for image analysis, enhancement, editing, and generation. Users can communicate with the application using natural language and perform various image processing operations through a simple web interface.

The application is developed using Flask and OpenCV and integrates Google Gemini for intelligent image understanding and conversational responses. It is deployed on Render for public access.

---

## User-Facing Branding

Lumina presents a consistent **Lumina AI** identity throughout the application. User-facing status messages, notifications, assistant responses, and service errors do not expose the names of underlying AI models or providers. Internal provider integrations remain implementation details and do not change the Lumina user experience.

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
- Web-based user interface
- Email OTP delivery for password reset

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

### Deployment

- Render
- Docker
- PostgreSQL (production authentication storage)
- SQLite (local-development fallback)

---

## Project Structure

```
Lumina/
│
├── app.py
├── templates/
├── static/
├── uploads/
├── processed/
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

Windows

```powershell
venv\Scripts\activate
```

Linux/macOS

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

**Important:** accounts created in the old Render-hosted SQLite database are not automatically migrated into the new PostgreSQL database. After configuring `DATABASE_URL`, create/test an account in the new database (or perform a deliberate data migration) before testing password reset. The PostgreSQL authentication layer uses a database cursor for each query so result fetching works correctly with both user and OTP operations.

Lumina uses **PostgreSQL in production** when the `DATABASE_URL` environment variable is configured. This keeps user accounts, password hashes, and password-reset OTP records persistent across Render deployments and restarts. SQLite remains the local-development fallback when `DATABASE_URL` is not set.

For a hosted deployment, create a PostgreSQL database (for example, through Supabase), copy its PostgreSQL connection string into Render as `DATABASE_URL`, and redeploy. The application automatically creates the required `users` and `otps` tables on startup; no manual SQL setup is required.

Do not commit `DATABASE_URL` or any database password to the repository.

### Run the Application

```bash
python app.py
```

The application will be available at

```
http://localhost:5000
```

---

## Deployment

The application is deployed on Render using Docker.

Live Demo:

https://lumina-fbhi.onrender.com

---

## How It Works

1. The user uploads an image.
2. The user provides a prompt or selects an image processing task.
3. The chatbot interprets the request using Google Gemini.
4. OpenCV performs the requested image processing operation.
5. The processed image and AI-generated response are returned to the user.
6. For password reset, Lumina generates a one-time OTP, stores it in the configured authentication database, and sends it through the Brevo HTTPS API.
7. In production, PostgreSQL keeps user accounts and OTP records persistent across Render restarts and deployments.

**Forgot-password UI:** The Send OTP control uses an explicit JavaScript click listener instead of relying on the inline `onclick` handler, while keeping the existing `sendOTP()` request flow unchanged.

---

## Applications

- Image enhancement
- Educational demonstrations
- Computer vision experimentation
- AI-assisted image editing
- Medical image assistance (non-diagnostic)
- Image analysis and understanding

---

## Future Enhancements

- Additional authentication hardening
- Image history management
- Batch image processing
- Additional AI image generation models
- Cloud storage integration
- REST API support
- Mobile-responsive interface
- Performance optimization

---

## Learning Outcomes

This project demonstrates practical experience in:

- Flask application development
- Computer vision using OpenCV
- AI model integration
- Prompt engineering
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


## Advanced Editing

Lumina supports natural-language multi-step image editing. Users can request several supported edits in a single message, and Lumina can preserve the requested order and apply up to eight operations as one workflow. The web interface also provides local undo/redo controls for image-edit states, with up to 20 states retained per active chat, plus an interactive before/after comparison with a draggable split slider.

Lumina also supports **4K desktop wallpaper fitting**. A request such as "make this image into a desktop 4K wallpaper" uses a 16:9 cover crop and high-quality resize to produce a 3840×2160 result without stretching the image. The crop can preserve the center by default and supports top/bottom positioning when explicitly requested.

The interface displays Lumina activity statuses while work is in progress, such as **Thinking**, **Analyzing**, **Working/Editing**, **Creating**, and **One last touch**. These are user-facing progress indicators and do not expose underlying model or provider implementation details.

Example: "Make it grayscale, sharpen it, and resize it to 1024x1024."

The advanced pipeline keeps provider/model implementation details internal and presents the workflow as a Lumina feature.

