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
```

Never commit real API keys or other credentials to GitHub.

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
6. For password reset, Lumina generates a one-time OTP and sends it through the Brevo HTTPS API.

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

- User authentication improvements
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
