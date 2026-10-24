# React Frontend Feature-Parity Audit

Baseline: `templates/index.html` and `static/response_enhancements.js` on `feature/react-frontend-migration`.
Target: `frontend/src/App.jsx`, `frontend/src/AuthModal.jsx`, and React styles.
Rule: preserve Flask routes and existing backend behavior; do not merge/deploy until parity checks pass.

## Confirmed working locally
- [x] React/Vite app boots and builds.
- [x] Flask API is reached through the Vite proxy.
- [x] Text chat and generated images render in chat (user-tested).
- [x] Basic image upload, generated-image download/save/copy, and image preview.
- [x] Basic sign-in/sign-up UI and password-reset UI exist (full workflow still needs testing).
- [x] Local browser storage for saved chats/images exists (not equivalent to original full chat history).

## Original frontend features found in `templates/index.html`

### Chat and generation
- [ ] Send/stop generation and explicit cancellation behavior.
- [ ] Original activity/status stages during processing.
- [ ] Multi-image attachments and previews; remove individual attachments.
- [ ] Paste images from clipboard, including paste into prompt area.
- [ ] Voice input via browser speech recognition, with graceful unsupported-browser handling.
- [ ] New chat and full conversation history save/list/load/delete, scoped by user.
- [ ] Restore message history and relevant image/edit context, not only a text snapshot.
- [ ] Share chat: create share link, copy link, and shared-chat behavior.
- [ ] Message copy controls and complete Markdown formatting.
- [ ] Code/table response enhancements: copy/save code, run supported snippets in sandbox, copy tables.
- [ ] Generated image variation display and per-image actions.
- [ ] Use a generated image as the next edit input with proper binary/data URL handling.
- [ ] Image lightbox actions: download, copy, save, close.
- [ ] Before/after comparison slider.
- [ ] Image edit undo/redo history and button enabled/disabled states.

### Image gallery and tools
- [ ] Saved gallery with persistence, badge/count, preview, download, delete, and save from lightbox.
- [ ] Operations panel and all original quick operations:
  - Medical: X-ray, MRI, CT, fundus/retina, skin, wound, CLAHE, denoise, segmentation.
  - Creative: cartoon, watercolor, sketch, oil painting, neon, glitch, halftone, pop art, stained glass, pointillism, thermal, vintage, duotone, double exposure, tilt-shift, bokeh, HDR, fisheye, ASCII, mosaic, cross-process.
  - Analysis: face detection, color palette, Canny, Sobel, quality check, color analysis, histogram equalization.
  - Restoration: super-resolution, deblur, colorize B&W, restore old photo.
  - Basic edits: rotate, flip, grayscale, invert, sepia, sharpen, blur, contrast, brightness, saturation, vignette, grain, warm/cool white balance, image info.
  - Generate from text shortcut.
- [ ] API status check from profile menu.
- [ ] Image compression/resizing behavior before upload (original max dimension 1024 and JPEG quality 0.85).

### Account and profile
- [ ] Sign-in and sign-up fully wired to existing Flask endpoints.
- [ ] Sign-up fields/validation: first/last name, email, username, date of birth/age gate, password strength, confirm password, terms agreement.
- [ ] Password visibility controls.
- [ ] Login identifier supports email/username and saved-account dropdown behavior.
- [ ] Failed-login rate-limit messaging/timer.
- [ ] Forgot password flow: request OTP, countdown, resend, six-digit OTP input navigation, verify, set new password.
- [ ] Change password workflow.
- [ ] Profile menu, user/plan display, logout, and session refresh.
- [ ] User-scoped saved data and safe behavior for guest sessions.

## React migration gaps visible from current implementation
- Saved chats currently restore text but explicitly reset API history/edit context.
- React saved gallery is capped at 12 and saved chats at 20; original storage behavior needs comparison and compatibility decisions.
- React UI currently lacks the original operations panel, voice input, chat sharing, change-password/profile menu, API status, complete auth validation/rate-limit UX, full code/table artifact actions, and full clipboard-paste/multi-attachment behavior.
- Some features may exist in backend or shared scripts, but they are not considered migrated until wired into React and tested end-to-end.

## Implementation and verification order
1. Preserve a baseline and finish the inventory above.
2. Migrate high-value interaction parity: full message rendering/actions, multi-attachment + paste, stop/status, image history/compare, gallery/history.
3. Migrate operations panel and all quick-operation prompts.
4. Complete account/profile flows using existing API endpoints only.
5. Migrate sharing and code/table artifact enhancements with appropriate security boundaries.
6. Add regression tests and run `npm run build` plus existing Python tests.
7. Manually test each checklist item against the original UI. Keep PR open until every item is implemented or explicitly documented as intentionally retired with user approval.
