import { useEffect, useRef, useState } from "react";
import AuthModal from "./AuthModal.jsx";
import {
  ArrowUp,
  ImagePlus,
  Sparkles,
  X,
  LoaderCircle,
  WandSparkles,
  PanelLeft,
  Images,
  History,
  Trash2,
  BookmarkPlus,
} from "lucide-react";

const WELCOME = {
  title: "What would you like to create?",
  subtitle: "Describe an image, upload one to edit, or ask Lumina about an image.",
};

export default function App() {
  const [prompt, setPrompt] = useState("");
  const [file, setFile] = useState(null);
  const [preview, setPreview] = useState("");
  const [messages, setMessages] = useState([]);
  const [apiHistory, setApiHistory] = useState([]);
  const [lastImage, setLastImage] = useState("");
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState("");
  const [user, setUser] = useState(null);
  const [authOpen, setAuthOpen] = useState(false);
  const [panel, setPanel] = useState("");
  const [lightbox, setLightbox] = useState("");
  const [savedGallery, setSavedGallery] = useState(() => { try { return JSON.parse(localStorage.getItem("lumina_react_gallery") || "[]"); } catch { return []; } });
  const [savedChats, setSavedChats] = useState(() => { try { return JSON.parse(localStorage.getItem("lumina_react_chats") || "[]"); } catch { return []; } });
  const fileInput = useRef(null);

  useEffect(() => {
    let active = true;
    fetch("/api/auth/me", { credentials: "same-origin" })
      .then((response) => response.ok ? response.json() : null)
      .then((data) => { if (active && data?.authenticated) setUser(data.user); })
      .catch(() => {});
    return () => { active = false; };
  }, []);

  async function signOut() {
    try { await fetch("/api/auth/logout", { method: "POST", credentials: "same-origin" }); }
    finally { setUser(null); }
  }

  useEffect(() => {
    try { localStorage.setItem("lumina_react_gallery", JSON.stringify(savedGallery.slice(0, 12))); }
    catch { setError("Browser storage is full. Remove saved images and try again."); }
  }, [savedGallery]);

  useEffect(() => {
    try { localStorage.setItem("lumina_react_chats", JSON.stringify(savedChats.slice(0, 20))); } catch {}
  }, [savedChats]);

  function saveCurrentChat() {
    if (!messages.length) { setError("Start a conversation before saving it."); return; }
    const firstPrompt = messages.find((item) => item.role === "user")?.text || "Lumina conversation";
    const chat = { id: crypto.randomUUID(), title: firstPrompt.slice(0, 52), savedAt: new Date().toISOString(), messages: messages.map(({ role, text }) => ({ role, text })) };
    setSavedChats((current) => [chat, ...current].slice(0, 20));
    setPanel("history");
    setError("");
  }

  function restoreChat(chat) {
    setMessages(chat.messages.map((item) => ({ ...item, id: crypto.randomUUID() })));
    setApiHistory([]);
    setLastImage("");
    setPanel("");
    setError("Saved text restored. Image-edit context starts fresh.");
  }

  function saveImage(src) {
    const image = src.startsWith("data:") || src.startsWith("http") ? src : "data:image/png;base64," + src;
    setSavedGallery((current) => [{ id: crypto.randomUUID(), src: image, savedAt: new Date().toISOString(), prompt: messages.filter((m) => m.role === "user").at(-1)?.text || "Lumina image" }, ...current.filter((item) => item.src !== image)].slice(0, 12));
    setPanel("gallery");
    setError("");
  }

  async function copyImage(src) {
    try {
      const normalized = src.startsWith("data:") || src.startsWith("http") ? src : "data:image/png;base64," + src;
      const response = await fetch(normalized);
      const blob = await response.blob();
      await navigator.clipboard.write([new ClipboardItem({ [blob.type || "image/png"]: blob })]);
      setError("Image copied to clipboard.");
    } catch {
      setError("Copying images isn't supported by this browser. Use Download instead.");
    }
  }

  function chooseFile(nextFile) {
    if (!nextFile) return;
    const allowedTypes = ["image/jpeg", "image/png", "image/webp", "image/gif", "image/bmp"];
    if (!allowedTypes.includes(nextFile.type)) {
      setError("Use a JPG, PNG, WEBP, GIF, or BMP image.");
      return;
    }
    if (nextFile.size > 10 * 1024 * 1024) {
      setError("Images must be 10 MB or smaller.");
      return;
    }
    if (preview.startsWith("blob:")) URL.revokeObjectURL(preview);
    setError("");
    setFile(nextFile);
    setPreview(URL.createObjectURL(nextFile));
  }

  function clearFile(revoke = true) {
    if (preview && revoke) URL.revokeObjectURL(preview);
    setFile(null);
    setPreview("");
    if (fileInput.current) fileInput.current.value = "";
  }

  async function submitPrompt(event) {
    event.preventDefault();
    const text = prompt.trim();
    if ((!text && !file) || busy) return;

    const userMessage = {
      id: crypto.randomUUID(),
      role: "user",
      text,
      image: preview || "",
    };
    setMessages((current) => [...current, userMessage]);
    setPrompt("");
    setError("");
    setBusy(true);

    try {
      const body = new FormData();
      body.append("prompt", text);
      body.append("history", JSON.stringify(apiHistory));
      if (lastImage) body.append("last_image", lastImage);
      if (file) body.append("image", file);

      const response = await fetch("/process", { method: "POST", body });
      const data = await response.json().catch(() => ({}));
      if (!response.ok) {
        throw new Error(data.message || data.error || "Lumina couldn't complete that request.");
      }

      const answer = data.message || data.reply || data.response || data.text ||
        "Lumina completed the request.";
      const images = Array.isArray(data.images) ? data.images :
        Array.isArray(data.result_images) ? data.result_images : [];
      setApiHistory(Array.isArray(data.history) ? data.history : apiHistory);
      setLastImage(typeof data.last_image === "string" ? data.last_image : lastImage);
      const singleImage = data.image || "";
      setMessages((current) => [
        ...current,
        { id: crypto.randomUUID(), role: "assistant", text: answer, images: images.length ? images : (singleImage ? [singleImage] : []) },
      ]);
      clearFile(false);
    } catch (err) {
      const message = err instanceof Error ? err.message : "Something went wrong.";
      setError(message);
      setMessages((current) => [
        ...current,
        { id: crypto.randomUUID(), role: "assistant", text: message, failed: true },
      ]);
    } finally {
      setBusy(false);
    }
  }

  return (
    <main className="app-shell">
      <aside className="sidebar">
        <a className="brand" href="/" aria-label="Lumina home">
          <span className="brand-mark"><Sparkles size={19} /></span>
          <span>Lumina</span>
        </a>
        <button className="new-chat" onClick={() => { setMessages([]); setApiHistory([]); setLastImage(""); setError(""); setPanel(""); }}>
          <PanelLeft size={17} /> New conversation
        </button>
        <button className="side-action" onClick={() => setPanel(panel === "history" ? "" : "history")}><History size={16} /> Saved conversations <span>{savedChats.length}</span></button>
        <button className="side-action" onClick={() => setPanel(panel === "gallery" ? "" : "gallery")}><Images size={16} /> Saved images <span>{savedGallery.length}</span></button>
        <button className="side-action" onClick={saveCurrentChat}><BookmarkPlus size={16} /> Save conversation</button>
        <div className="sidebar-note">
          <span className="status-dot" />
          <span>AI workspace</span>
        </div>
        <div className="sidebar-bottom">
          <div className="avatar">{user ? (user.firstName || user.username || "U").slice(0, 1).toUpperCase() : "L"}</div>
          <div className="account-details"><strong>{user ? (user.firstName || user.username || "Account") : "Lumina"}</strong><small>{user ? (user.email || user.username) : "Image assistant"}</small></div>
          {user ? <button className="account-action" onClick={signOut}>Sign out</button> : <button className="account-action" onClick={() => setAuthOpen(true)}>Sign in</button>}
        </div>
      </aside>

      <section className="workspace">
        <header className="topbar">
          <div className="mobile-brand"><span className="brand-mark"><Sparkles size={17} /></span> Lumina</div>
          <span className="model-label"><span className="status-dot" /> AI Image Studio</span>
          <div className="topbar-actions">
            {user ? <span className="signed-in-label">{user.firstName || user.username}</span> : <button className="topbar-signin" onClick={() => setAuthOpen(true)}>Sign in</button>}
            <span className="topbar-caption">Create · Edit · Explore</span>
          </div>
        </header>

        <div className="conversation" aria-live="polite">
          {messages.length === 0 ? (
            <div className="welcome">
              <div className="welcome-icon"><WandSparkles size={25} /></div>
              <p className="eyebrow">YOUR CREATIVE SPACE</p>
              <h1>{WELCOME.title}</h1>
              <p className="welcome-subtitle">{WELCOME.subtitle}</p>
              <div className="suggestions">
                {[
                  ["Create", "A cinematic landscape at golden hour"],
                  ["Edit", "Make my photo look like a film still"],
                  ["Explore", "Describe the image I upload"],
                ].map(([label, suggestion]) => (
                  <button className="suggestion" key={label} onClick={() => setPrompt(suggestion)}>
                    <span>{label}</span><p>{suggestion}</p>
                  </button>
                ))}
              </div>
            </div>
          ) : (
            <div className="message-list">
              {messages.map((message) => (
                <article className={`message message-${message.role}`} key={message.id}>
                  <div className="message-avatar">{message.role === "user" ? "Y" : <Sparkles size={16} />}</div>
                  <div className="message-content">
                    <strong>{message.role === "user" ? "You" : "Lumina"}</strong>
                    {message.text && <p>{message.text}</p>}
                    {message.image && <img className="message-image" src={message.image} alt="Uploaded input" />}
                    {message.images?.map((src, index) => {
                      const imageSrc = src.startsWith("data:") || src.startsWith("http") ? src : "data:image/png;base64," + src;
                      return <div className="result-image-wrap" key={index}>
                        <button className="image-open-button" onClick={() => setLightbox(imageSrc)} aria-label="Open image preview"><img className="message-image" src={imageSrc} alt={"Generated or edited result " + (index + 1)} /></button>
                        <div className="image-action-row"><button className="save-image-button" onClick={() => saveImage(src)}><BookmarkPlus size={14} /> Save image</button><a className="image-download-link" href={imageSrc} download="lumina-image.png">Download</a><button className="image-download-link" onClick={() => copyImage(src)}>Copy image</button><button className="image-download-link" onClick={() => { setLastImage(src.startsWith("data:") ? src.split(",", 2)[1] : src); setPrompt("Edit this image: "); }}>Use for edit</button></div>
                      </div>;
                    })}
                  </div>
                </article>
              ))}
              {busy && <div className="loading-line"><LoaderCircle className="spin" size={17} /> Lumina is thinking…</div>}
            </div>
          )}
        </div>

        <div className="composer-area">
          {error && <div className="error-banner" role="alert">{error}</div>}
          <form className="composer" onSubmit={submitPrompt}>
            {preview && (
              <div className="attachment">
                <img src={preview} alt="Selected upload preview" />
                <button type="button" aria-label="Remove image" onClick={clearFile}><X size={15} /></button>
              </div>
            )}
            <textarea
              value={prompt}
              onPaste={(event) => {
                const imageItem = Array.from(event.clipboardData?.items || []).find((item) => item.type.startsWith("image/"));
                const pastedImage = imageItem?.getAsFile();
                if (pastedImage) {
                  event.preventDefault();
                  chooseFile(new File([pastedImage], "pasted-image.png", { type: pastedImage.type || "image/png" }));
                }
              }}
              onChange={(event) => setPrompt(event.target.value)
              onKeyDown={(event) => {
                if (event.key === "Enter" && !event.shiftKey) {
                  event.preventDefault();
                  event.currentTarget.form?.requestSubmit();
                }
              }}
              placeholder="Message Lumina…"
              rows={2}
              aria-label="Message Lumina"
            />
            <div className="composer-footer">
              <div className="composer-tools">
                <input
                  ref={fileInput}
                  type="file"
                  accept="image/jpeg,image/png,image/webp,image/gif,image/bmp"
                  hidden
                  onChange={(event) => chooseFile(event.target.files?.[0])}
                />
                <button className="tool-button" type="button" onClick={() => fileInput.current?.click()}>
                  <ImagePlus size={17} /><span>Add image</span>
                </button>
                <span className="hint">Shift + Enter for a new line</span>
              </div>
              <button className="send-button" type="submit" disabled={busy || (!prompt.trim() && !file)} aria-label="Send message">
                {busy ? <LoaderCircle className="spin" size={17} /> : <ArrowUp size={18} />}
              </button>
            </div>
          </form>
          <p className="disclaimer">Lumina can make mistakes. Review generated results before using them.</p>
        </div>
      </section>
      {lightbox && <div className="lightbox-backdrop" onMouseDown={(event) => { if (event.target === event.currentTarget) setLightbox(""); }} role="presentation">
        <button className="lightbox-close" onClick={() => setLightbox("")} aria-label="Close image preview"><X size={20} /></button>
        <img src={lightbox} alt="Expanded Lumina result" />
        <a href={lightbox} download="lumina-image.png">Download image</a>
      </div>}
      {panel && <div className="drawer-backdrop" onMouseDown={(event) => { if (event.target === event.currentTarget) setPanel(""); }}>
        <section className="drawer">
          <header className="drawer-header"><div><h2>{panel === "gallery" ? "Saved images" : "Saved conversations"}</h2><p>Saved locally in this browser</p></div><button className="drawer-close" onClick={() => setPanel("")} aria-label="Close panel"><X size={18} /></button></header>
          {panel === "gallery" ? (savedGallery.length ? <div className="saved-gallery">{savedGallery.map((item) => <article className="saved-image-card" key={item.id}><img src={item.src} alt={item.prompt} /><p title={item.prompt}>{item.prompt}</p><div className="saved-image-actions"><a href={item.src} download="lumina-image.png">Download</a><button onClick={() => setSavedGallery((current) => current.filter((saved) => saved.id !== item.id))} aria-label="Remove saved image"><Trash2 size={14} /></button></div></article>)}</div> : <p className="drawer-empty">No saved images yet. Generate an image and select Save image.</p>) : (savedChats.length ? <div className="saved-chat-list">{savedChats.map((chat) => <article className="saved-chat-card" key={chat.id}><button className="saved-chat-open" onClick={() => restoreChat(chat)}><strong>{chat.title}</strong><small>{new Date(chat.savedAt).toLocaleString()} · {chat.messages.length} messages</small></button><button className="saved-chat-delete" onClick={() => setSavedChats((current) => current.filter((item) => item.id !== chat.id))} aria-label="Delete saved conversation"><Trash2 size={14} /></button></article>)}</div> : <p className="drawer-empty">No saved conversations yet. Use Save conversation to keep a text copy.</p>)}
        </section>
      </div>}
      {authOpen && <AuthModal onClose={() => setAuthOpen(false)} onAuthenticated={(nextUser) => { setUser(nextUser); setAuthOpen(false); }} />}
    </main>
  );
}
