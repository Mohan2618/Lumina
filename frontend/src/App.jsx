import { useRef, useState } from "react";
import {
  ArrowUp,
  ImagePlus,
  Sparkles,
  X,
  LoaderCircle,
  WandSparkles,
  PanelLeft,
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
  const fileInput = useRef(null);

  function chooseFile(nextFile) {
    if (!nextFile) return;
    if (!nextFile.type.startsWith("image/")) {
      setError("Please choose an image file.");
      return;
    }
    setError("");
    setFile(nextFile);
    setPreview(URL.createObjectURL(nextFile));
  }

  function clearFile() {
    if (preview) URL.revokeObjectURL(preview);
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
      clearFile();
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
        <button className="new-chat" onClick={() => { setMessages([]); setApiHistory([]); setLastImage(""); setError(""); }}>
          <PanelLeft size={17} /> New conversation
        </button>
        <div className="sidebar-note">
          <span className="status-dot" />
          <span>AI workspace</span>
        </div>
        <div className="sidebar-bottom">
          <div className="avatar">L</div>
          <div><strong>Lumina</strong><small>Image assistant</small></div>
        </div>
      </aside>

      <section className="workspace">
        <header className="topbar">
          <div className="mobile-brand"><span className="brand-mark"><Sparkles size={17} /></span> Lumina</div>
          <span className="model-label"><span className="status-dot" /> AI Image Studio</span>
          <span className="topbar-caption">Create · Edit · Explore</span>
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
                    {message.images?.map((src, index) => (
                      <img className="message-image" key={index} src={src.startsWith("data:") || src.startsWith("http") ? src : `data:image/png;base64,${src}`} alt={`Generated result ${index + 1}`} />
                    ))}
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
              onChange={(event) => setPrompt(event.target.value)}
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
                  accept="image/*"
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
    </main>
  );
}
