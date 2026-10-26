# React Frontend Feature-Parity Audit

Baseline: `templates/index.html` and `static/response_enhancements.js` on `feature/react-frontend-migration`.
Target: `frontend/src/App.jsx`, `frontend/src/AuthModal.jsx`, and React styles.
Rule: preserve Flask routes and existing backend behavior; do not merge/deploy until parity checks pass.

## Confirmed working locally
- [x] React/Vite app boots and builds.
- [x] Flask API is reached through the Vite proxy.
- [x] Text chat and generated images render in chat (user-tested).
- [x] Basic image upload, generated-image download/save/copy, and image preview.
- [x] Sign-in/sign-up UI and password-reset UI exist; sign-up now validates email format, username format, password strength, age 13+, and terms acceptance in the client (server-side age/terms enforcement is not implemented).
- [x] Local browser storage for saved chats/images exists (not equivalent to original full chat history).

## Original frontend features found in `templates/index.html`

### Chat and generation
- [x] Send/stop generation and explicit client-side request cancellation (server-side work may continue if already running).
- [x] Task-specific progress labels for image generation, image editing, coding, structured information, and normal chat (sequential stage animation still differs from the original).
- [x] Multi-image attachments and previews; remove individual attachments.
- [x] Paste images from clipboard into the prompt area.
- [x] Voice input via browser speech recognition, with graceful unsupported-browser handling (manual browser test still required).
- [x] New chat and conversation history save/list/load/delete, with authenticated server sync.
- [x] Restore messages, API history, image state, and edit versions.
- [x] Create/copy expiring share links; shared conversations load read-only and expire after 7 days.
- [ ] Message copy controls exist; full Markdown parity (nested lists, blockquotes, richer inline syntax) still needs work.
- [x] Code copy/save, sandboxed browser-safe run previews, and table copy controls.
- [x] Display returned image variations and provide per-image actions.
- [x] Use generated images as the next edit input through the existing `last_image` flow.
- [x] Image lightbox actions: download, copy, save to gallery, close.
- [x] Draggable before/after comparison slider.
- [x] Image edit undo/redo history and button enabled/disabled states.

### Image gallery and tools
- [x] Saved gallery with browser persistence, badge/count, display, download, delete, and save from lightbox; signed-in saves use the authenticated `/api/gallery` endpoints, while guest images stay in guest browser storage (legacy guest items are not automatically imported into accounts).
- [ ] Operations panel and all original quick operations:
  - Medical: X-ray, MRI, CT, fundus/retina, skin, wound, CLAHE, denoise, segmentation.
  - Creative: cartoon, watercolor, sketch, oil painting, neon, glitch, halftone, pop art, stained glass, pointillism, thermal, vintage, duotone, double exposure, tilt-shift, bokeh, HDR, fisheye, ASCII, mosaic, cross-process.
  - Analysis: face detection, color palette, Canny, Sobel, quality check, color analysis, histogram equalization.
  - Restoration: super-resolution, deblur, colorize B&W, restore old photo.
  - Basic edits: rotate, flip, grayscale, invert, sepia, sharpen, blur, contrast, brightness, saturation, vignette, grain, warm/cool white balance, image info.
  - Generate from text shortcut.
- [x] API status check from the workspace navigation.
- [ ] Image compression/resizing behavior before upload (original max dimension 1024 and JPEG quality 0.85).

### Account and profile
- [x] Sign-in and sign-up wired to existing Flask endpoints.
- [x] Sign-up fields/validation: first/last name, email, username, date of birth/age gate, password strength, confirm password, terms agreement (age/terms still need server-side enforcement).
- [ ] Password visibility controls.
- [ ] Login identifier supports email/username and saved-account dropdown behavior.
- [ ] Failed-login rate-limit messaging/timer.
- [x] Forgot password request, 60-second resend countdown, resend, six separate OTP digit fields with keyboard navigation/paste, and OTP/password reset UI are implemented; end-to-end email verification still needs manual testing.
- [x] Authenticated change-password workflow using the current password.
- [x] User display, logout, Settings, and Change password are accessible through the profile dropdown; dedicated plan display and saved-account dropdown remain unimplemented.
- [x] User-scoped conversation and gallery API endpoints; guest gallery storage is separated from authenticated gallery state. Verify all sign-out/login edge cases manually.

## Latest branch changes
- Added authenticated saved-gallery GET/POST/DELETE endpoints and frontend sync for signed-in saved images.
- Added client and server validation for signup email, username, password strength, date of birth/age 13+, and terms acceptance.
- Added a compact profile dropdown containing Settings, Change password, and Sign out (Sign in for guests).
- Fixed account history/gallery hydration after in-page login.
- CI passed on `5ab78e8`. Profile-menu commits `8d4d622` and `1da8341` are awaiting CI results.

## Latest user test results
- [x] Operations panel visible and opens.
- [x] API status check reports the backend reachable.
- [ ] Voice input not tested yet.

## React migration gaps visible from current implementation
- Saved chats now restore API history and edit context; confirm with an authenticated end-to-end test.
- React saved gallery is capped at 12 and saved chats at 20; original storage behavior needs comparison and compatibility decisions.
- Remaining known gaps: full Markdown parity; saved-account dropdown; failed-login rate-limit timer; dedicated plan display; image compression/resizing parity; voice-input verification; and final end-to-end/manual regression tests.
- Some features may exist in backend or shared scripts, but they are not considered migrated until wired into React and tested end-to-end.

## Implementation and verification order
1. Preserve a baseline and finish the inventory above.
2. Migrate high-value interaction parity: full message rendering/actions, multi-attachment + paste, stop/status, image history/compare, gallery/history.
3. Migrate operations panel and all quick-operation prompts.
4. Complete account/profile flows using existing API endpoints only.
5. Migrate sharing and code/table artifact enhancements with appropriate security boundaries.
6. Add regression tests and run `npm run build` plus existing Python tests.
7. Manually test each checklist item against the original UI. Keep PR open until every item is implemented or explicitly documented as intentionally retired with user approval.
