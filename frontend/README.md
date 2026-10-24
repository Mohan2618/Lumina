# Lumina React frontend

This folder is the first, isolated phase of migrating Lumina's UI from the large
Jinja/vanilla-JavaScript template to React. The existing Flask backend and
`templates/index.html` are intentionally unchanged in this phase.

## Run locally

1. Start the existing Flask app on port 7860 using the repository's normal
   Python setup.
2. In a second terminal:

   ```bash
   cd frontend
   npm install
   npm run dev
   ```

3. Open the local URL printed by Vite (usually http://localhost:5173).

Vite forwards `/api`, `/process`, and `/health` requests to Flask on port 7860.

## Migration status

- [x] Add a React + Vite entry point and isolated workspace.
- [x] Add the first React conversation workspace and image-upload interaction.
- [x] Preserve Flask's `history`, `last_image`, `image`, and `images` response fields in the React chat flow.
- [x] Add React sign-in/sign-up UI using the existing `/api/auth/me`, `/api/auth/verify-password`, `/api/auth/signup`, and `/api/auth/logout` endpoints.
- [ ] Map the exact response schema for every image-generation/editing operation.
- [x] Add React sign-in/sign-up and OTP password-reset screens using existing Flask auth endpoints.
- [x] Add session-aware account controls.
- [x] Add browser-local saved conversation text and saved-image gallery with delete/download controls.
- [ ] Migrate the original app's full history restore, image variations, compare/lightbox, attachments, undo/redo, and share flows.
- [ ] Configure the production build to be served by Flask/Render.
- [ ] Run end-to-end regression checks against the existing backend.
- [ ] Switch the production route only after parity checks pass.

Saved conversations and gallery entries in this first React phase are stored in the current browser's local storage; they are not synced to the Flask database. Restoring a saved conversation restores text only, not the backend image-edit context. Password-reset email delivery depends on the existing Flask email configuration.

The React app is deliberately not wired into the production Flask route yet.
This avoids replacing a working interface before the remaining features and
backend response formats have been migrated and verified.
