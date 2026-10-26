import { useEffect, useRef, useState } from "react";
import AuthModal from "./AuthModal.jsx";
import {
  ArrowUp,
  Copy,
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
  Mic,
  Activity,
  Share2,
} from "lucide-react";

const WELCOME = {
  title: "What would you like to create?",
  subtitle: "Describe an image, upload one to edit, or ask Lumina about an image.",
};

const OPERATION_GROUPS = [
  ["Medical", [
    ["X-Ray Enhance", "enhance this xray"], ["MRI Enhance", "enhance MRI scan"],
    ["CT Enhance", "enhance ct scan"], ["Fundus/Retina", "fundus enhance"],
    ["Skin Analysis", "analyze skin"], ["Wound Analysis", "analyze wound"],
    ["CLAHE", "clahe enhance"], ["Denoise", "denoise image"], ["Segmentation", "segment image"],
  ]],
  ["Creative effects", [
    ["Cartoon", "apply cartoon effect"], ["Watercolor", "apply watercolor effect"],
    ["Pencil Sketch", "apply pencil sketch"], ["Oil Painting", "apply oil painting"],
    ["Neon Glow", "apply neon glow"], ["Glitch Art", "apply glitch effect"],
    ["Halftone", "apply halftone effect"], ["Pop Art", "apply pop art effect"],
    ["Stained Glass", "apply stained glass effect"], ["Pointillism", "apply pointillism effect"],
    ["Thermal Vision", "apply thermal vision"], ["Vintage", "apply vintage filter"],
    ["Duotone", "apply duotone effect"], ["Double Exposure", "double exposure effect"],
    ["Tilt-Shift", "tilt shift miniature"], ["Bokeh", "apply bokeh blur background"],
    ["HDR", "apply HDR effect"], ["Fisheye", "apply fisheye lens effect"],
    ["ASCII Art", "ascii art"], ["Mosaic", "apply mosaic effect"], ["Cross Process", "cross process effect"],
  ]],
  ["Analysis", [
    ["Face Detect", "detect faces"], ["Color Palette", "extract color palette"],
    ["Canny Edges", "canny edge detection"], ["Sobel", "sobel gradient"],
    ["Quality Check", "check image quality"], ["Color Analysis", "analyze colors"],
    ["Histogram Equalization", "histogram equalization"],
  ]],
  ["Restoration", [
    ["Super Resolution", "super resolution upscale"], ["Deblur", "deblur image"],
    ["Colorize B&W", "colorize black and white"], ["Restore Old Photo", "restore old photo"],
  ]],
  ["Basic edits", [
    ["Rotate 90°", "rotate 90 degrees"], ["Flip Horizontal", "flip horizontally"],
    ["Flip Vertical", "flip vertically"], ["Grayscale", "make grayscale"],
    ["Invert", "invert colors"], ["Sepia", "apply sepia"], ["Sharpen", "sharpen image"],
    ["Blur", "blur image"], ["Increase Contrast", "increase contrast"],
    ["Increase Brightness", "increase brightness"], ["Increase Saturation", "increase saturation"],
    ["Vignette", "add vignette"], ["Film Grain", "add film grain"],
    ["Warm Tone", "warm white balance"], ["Cool Tone", "cool white balance"],
    ["Image Info", "image info"],
  ]],
];

function InlineMarkdown({ text }) {
  const source = String(text ?? "");
  const tokens = source.split(/(\x60{1,2}[^\\n\x60]+\x60{1,2}|\*\*[^*]+\*\*|__[^_]+__|\*[^*\n]+\*|_[^_\n]+_|\[[^\]]+\]\([^)]+\))/g);
  return tokens.map((token, index) => {
    if (/^\x60{1,2}[^\\n]+\x60{1,2}$/.test(token)) return <code className="inline-code" key={index}>{token.replace(/^\x60{1,2}|\x60{1,2}$/g, "")}</code>;
    if (token.startsWith("**") && token.endsWith("**")) return <strong key={index}>{token.slice(2, -2)}</strong>;
    if (token.startsWith("__") && token.endsWith("__")) return <strong key={index}>{token.slice(2, -2)}</strong>;
    if ((token.startsWith("*") && token.endsWith("*")) || (token.startsWith("_") && token.endsWith("_"))) return <em key={index}>{token.slice(1, -1)}</em>;
    const link = token.match(/^\[([^\]]+)\]\(([^)]+)\)$/);
    if (link) return <a key={index} href={link[2]} target="_blank" rel="noreferrer">{link[1]}</a>;
    return token;
  });
}
function HighlightedCode({ code, language }) {
  const lang = String(language || "").toLowerCase();
  if (!["python", "py", "javascript", "js", "typescript", "ts", "jsx", "tsx", "json", "bash", "sh"].includes(lang)) return <code>{code}</code>;
  const tokenPattern = /(#[^\n]*|\/\/[^\n]*|"(?:\\.|[^"\\])*"|\'(?:\\.|[^\'\\])*\'|\b(?:def|return|if|elif|else|for|while|in|not|and|or|is|class|import|from|as|try|except|finally|with|lambda|yield|async|await|pass|break|continue|raise|True|False|None|const|let|var|function|new|throw|catch|typeof|export|default|interface|type|public|private|static)\b|\b\d+(?:\.\d+)?\b|\b(?:print|len|range|enumerate|int|str|list|dict|set|sum|min|max|abs|map|filter|input|JSON|console|Math)\b|[=+\-*/%<>!]+|[{}()[\].,:])/g;
  const parts = []; let cursor = 0, match;
  while ((match = tokenPattern.exec(code)) !== null) {
    if (match.index > cursor) parts.push({ text: code.slice(cursor, match.index), type: "" });
    const token = match[0]; let type = "punctuation";
    if (/^(#|\/\/)/.test(token)) type = "comment";
    else if (/^["\']/.test(token)) type = "string";
    else if (/^\d/.test(token)) type = "number";
    else if (/^(def|return|if|elif|else|for|while|in|not|and|or|is|class|import|from|as|try|except|finally|with|lambda|yield|async|await|pass|break|continue|raise|True|False|None|const|let|var|function|new|throw|catch|typeof|export|default|interface|type|public|private|static)$/.test(token)) type = "keyword";
    else if (/^(print|len|range|enumerate|int|str|list|dict|set|sum|min|max|abs|map|filter|input|JSON|console|Math)$/.test(token)) type = "builtin";
    else if (/^[=+\-*/%<>!]+$/.test(token)) type = "operator";
    parts.push({ text: token, type }); cursor = tokenPattern.lastIndex;
  }
  if (cursor < code.length) parts.push({ text: code.slice(cursor), type: "" });
  return <code>{parts.map((part, i) => <span key={i} className={part.type ? "syntax-" + part.type : undefined}>{part.text}</span>)}</code>;
}
function CodeSnippet({ code, language, onCopy, copied }) {
  const [wrapped, setWrapped] = useState(false);
  const [previewOpen, setPreviewOpen] = useState(false);
  const normalizedLanguage = String(language || "").toLowerCase();
  const canRun = ["html", "svg", "css", "javascript", "js"].includes(normalizedLanguage);
  const extension = ({ javascript: "js", js: "js", html: "html", svg: "svg", css: "css", python: "py", py: "py", typescript: "ts", ts: "ts", jsx: "jsx", tsx: "tsx", json: "json", bash: "sh", sh: "sh" })[normalizedLanguage] || "txt";
  const previewDocument = normalizedLanguage === "css"
    ? `<!doctype html><html><head><meta charset="utf-8"><style>${code}</style></head><body><h2>CSS preview</h2><p>Sample text</p><button>Sample button</button></body></html>`
    : normalizedLanguage === "js" || normalizedLanguage === "javascript"
      ? `<!doctype html><html><head><meta charset="utf-8"></head><body><h2>JavaScript preview</h2><pre id="output"></pre><script>const originalLog = console.log; console.log = (...args) => { document.getElementById('output').textContent += args.join(' ') + '\\n'; originalLog(...args); }; try { ${code} } catch (error) { document.getElementById('output').textContent += String(error); }</script></body></html>`
      : code;
  function saveCode() {
    const blob = new Blob([code], { type: "text/plain;charset=utf-8" });
    const url = URL.createObjectURL(blob);
    const anchor = document.createElement("a");
    anchor.href = url;
    anchor.download = "lumina-code." + extension;
    document.body.appendChild(anchor);
    anchor.click();
    anchor.remove();
    URL.revokeObjectURL(url);
  }
  return <>
    <section className="response-code">
      <header className="response-code-header"><span className="code-language"><span className="code-language-dot" />{language || "Code"}</span><div className="code-header-actions"><button type="button" onClick={() => setWrapped(!wrapped)}>{wrapped ? "No wrap" : "Wrap"}</button><button type="button" onClick={saveCode}>Save</button>{canRun && <button type="button" onClick={() => setPreviewOpen(true)}>Run preview</button>}<button type="button" onClick={() => onCopy(code)} aria-label="Copy code snippet"><Copy size={14} />{copied ? "Copied" : "Copy code"}</button></div></header>
      <pre className={wrapped ? "code-pre wrapped" : "code-pre"}><HighlightedCode code={code} language={language} /></pre>
    </section>
    {previewOpen && <div className="artifact-preview-backdrop" role="presentation" onMouseDown={(event) => { if (event.target === event.currentTarget) setPreviewOpen(false); }}>
      <section className="artifact-preview-dialog" role="dialog" aria-modal="true" aria-label="Code preview">
        <header><strong>Sandboxed preview · {normalizedLanguage.toUpperCase()}</strong><button type="button" onClick={() => setPreviewOpen(false)} aria-label="Close code preview">Close</button></header>
        <iframe title="Sandboxed code preview" sandbox="allow-scripts" referrerPolicy="no-referrer" srcDoc={previewDocument} />
        <p>Preview runs in an isolated sandbox. It cannot access Lumina's page storage or authenticated session.</p>
      </section>
    </div>}
  </>;
}
function StructuredResponse({ text, onCopy }) {
  const [copiedCode, setCopiedCode] = useState("");
  const source = String(text || "").replace(/\r\n?/g, "\n");
  const fence = /\x60{3,}([^\n\x60]*)\n([\s\S]*?)(?:\n?\x60{3,}(?=\n|$)|$)/g;
  const blocks = [];
  let cursor = 0, match;
  while ((match = fence.exec(source)) !== null) {
    if (match.index > cursor) blocks.push({ type: "text", text: source.slice(cursor, match.index) });
    blocks.push({ type: "code", language: match[1].trim().replace(/^(python|py)$/i, "python"), code: match[2].replace(/\n$/, "") });
    cursor = fence.lastIndex;
    if (match[0].length === 0) fence.lastIndex++;
  }
  if (cursor < source.length) blocks.push({ type: "text", text: source.slice(cursor) });
  function copyCode(code) {
    navigator.clipboard.writeText(code).then(() => { setCopiedCode(code); window.setTimeout(() => setCopiedCode(""), 1800); }).catch(() => {});
  }
  return <div className="structured-response">
    {blocks.map((block, index) => block.type === "code"
      ? <CodeSnippet key={index} code={block.code} language={block.language} onCopy={copyCode} copied={copiedCode === block.code} />
      : <MarkdownText key={index} text={block.text} onCopyTable={(csv) => navigator.clipboard.writeText(csv).catch(() => {})} />)}
    <div className="response-actions"><button type="button" onClick={() => onCopy(text)}><Copy size={14} /> Copy response</button></div>
  </div>;
}
function MarkdownText({ text, onCopyTable }) {
  const lines = String(text || "").replace(/\r\n?/g, "\n").split("\n");
  const output = [];
  let paragraph = [], listItems = [], listType = "", listIndent = 0;
  const flushParagraph = () => {
    if (paragraph.length) {
      output.push(<p key={"p" + output.length}><InlineMarkdown text={paragraph.join(" ").trim()} /></p>);
      paragraph = [];
    }
  };
  const flushList = () => {
    if (listItems.length) {
      const Tag = listType === "ol" ? "ol" : "ul";
      output.push(<Tag key={"l" + output.length}>{listItems.map((item, i) => <li key={i}><InlineMarkdown text={item} /></li>)}</Tag>);
      listItems = []; listType = ""; listIndent = 0;
    }
  };
  const isTableRow = (line) => line.trim().startsWith("|") && line.trim().endsWith("|");
  const parseCells = (line) => line.trim().replace(/^\|/, "").replace(/\|$/, "").split(/(?<!\\)\|/).map((cell) => cell.trim().replace(/\\\|/g, "|"));
  const isDivider = (line) => parseCells(line).every((cell) => /^:?-{3,}:?$/.test(cell));
  for (let index = 0; index < lines.length; index++) {
    const raw = lines[index], line = raw.trim();
    const bullet = raw.match(/^(\s*)[-*+]\s+(.+)$/), numbered = raw.match(/^(\s*)\d+[.)]\s+(.+)$/);
    if (isTableRow(line)) {
      flushParagraph(); flushList();
      const tableLines = [];
      let cursor = index;
      while (cursor < lines.length && isTableRow(lines[cursor].trim())) { tableLines.push(lines[cursor].trim()); cursor++; }
      if (tableLines.length >= 2 && isDivider(tableLines[1])) {
        const headers = parseCells(tableLines[0]), rows = tableLines.slice(2).map(parseCells);
        const csv = [headers, ...rows].map((row) => row.map((cell) => '"' + cell.replace(/"/g, '""') + '"').join(",")).join("\n");
        output.push(<div className="response-table-wrap" key={"t" + output.length}><div className="response-table-toolbar"><span>Table</span><button type="button" onClick={() => onCopyTable(csv)}><Copy size={13} /> Copy table</button></div><div className="response-table-scroll"><table className="response-table"><thead><tr>{headers.map((cell, i) => <th key={i}><InlineMarkdown text={cell} /></th>)}</tr></thead><tbody>{rows.map((row, ri) => <tr key={ri}>{headers.map((_, ci) => <td key={ci}><InlineMarkdown text={row[ci] || ""} /></td>)}</tr>)}</tbody></table></div></div>);
        index = cursor - 1; continue;
      }
    }
    if (!line) { flushParagraph(); flushList(); continue; }
    const heading = line.match(/^(#{1,6})\s+(.+)$/);
    const boldHeading = line.match(/^\*\*(.+?)\*\*:?\s*$/);
    if (heading || boldHeading) {
      flushParagraph(); flushList();
      const level = heading ? Math.min(heading[1].length + 2, 6) : 4;
      const Tag = "h" + level;
      output.push(<Tag key={"h" + output.length}><InlineMarkdown text={heading ? heading[2] : boldHeading[1].replace(/:$/, "")} /></Tag>);
      continue;
    }
    if (bullet || numbered) {
      flushParagraph();
      const nextType = numbered ? "ol" : "ul";
      const indent = (numbered || bullet)[1].length;
      if (listType && (listType !== nextType || indent !== listIndent)) flushList();
      listType = nextType; listIndent = indent;
      listItems.push((numbered || bullet)[2]);
      continue;
    }
    flushList();
    paragraph.push(line.replace(/ {2,}/g, " "));
  }
  flushParagraph(); flushList();
  return <div className="markdown-text">{output}</div>;
}

export default function App() {
  const [prompt, setPrompt] = useState("");
  const [attachments, setAttachments] = useState([]);
  const [messages, setMessages] = useState([]);
  const [apiHistory, setApiHistory] = useState([]);
  const [lastImage, setLastImage] = useState("");
  const [editVersions, setEditVersions] = useState([]);
  const [editVersionIndex, setEditVersionIndex] = useState(-1);
  const [compareImages, setCompareImages] = useState(null);
  const [comparePosition, setComparePosition] = useState(50);
  const [busy, setBusy] = useState(false);
  const requestController = useRef(null);
  const activeConversationId = useRef("");
  const [error, setError] = useState("");
  const [user, setUser] = useState(null);
  const [authOpen, setAuthOpen] = useState(false);
  const [passwordModalOpen, setPasswordModalOpen] = useState(false);
  const [passwordFields, setPasswordFields] = useState({ currentPassword: "", newPassword: "", confirmPassword: "" });
  const [passwordError, setPasswordError] = useState("");
  const [passwordNotice, setPasswordNotice] = useState("");
  const [passwordBusy, setPasswordBusy] = useState(false);
  const [panel, setPanel] = useState("");
  const [lightbox, setLightbox] = useState("");
  const [apiStatus, setApiStatus] = useState("");
  const [progressLabel, setProgressLabel] = useState("Lumina is thinking…");
  const [sharedReadOnly, setSharedReadOnly] = useState(false);
  const [listening, setListening] = useState(false);
  const recognitionRef = useRef(null);
  const [savedGallery, setSavedGallery] = useState(() => { try { return JSON.parse(localStorage.getItem("lumina_react_gallery") || "[]"); } catch { return []; } });
  const [savedChats, setSavedChats] = useState(() => { try { return JSON.parse(localStorage.getItem("lumina_react_chats") || "[]"); } catch { return []; } });
  const fileInput = useRef(null);

  useEffect(() => {
    let active = true;
    fetch("/api/auth/me", { credentials: "same-origin" })
      .then((response) => response.ok ? response.json() : null)
      .then(async (data) => {
        if (!active || !data?.authenticated) return;
        setUser(data.user);
        // Restore the last active conversation only for an authenticated session.
        // A public/guest visit must always start with a clean chat.
        if (!new URLSearchParams(window.location.search).has("share")) {
          try {
            const activeId = localStorage.getItem("lumina_react_active_chat");
            const activeChat = savedChats.find((chat) => chat.id === activeId);
            if (activeChat && Array.isArray(activeChat.messages) && activeChat.messages.length) {
              activeConversationId.current = activeId;
              setMessages(activeChat.messages.map((item) => ({ ...item, id: crypto.randomUUID() })));
              setApiHistory(Array.isArray(activeChat.apiHistory) ? activeChat.apiHistory : []);
              setLastImage(activeChat.lastImage || "");
              setEditVersions(Array.isArray(activeChat.editVersions) ? activeChat.editVersions : []);
              setEditVersionIndex(Number.isInteger(activeChat.editVersionIndex) ? activeChat.editVersionIndex : -1);
            } else {
              activeConversationId.current = "";
              localStorage.removeItem("lumina_react_active_chat");
            }
          } catch { /* Ignore invalid browser cache and start with a clean chat. */ }
        }
        try {
          const response = await fetch("/api/conversations", { credentials: "same-origin" });
          if (!response.ok) return;
          const result = await response.json();
          if (active && Array.isArray(result.conversations) && result.conversations.length) {
            setSavedChats((current) => {
              const merged = new Map(current.map((chat) => [chat.id, chat]));
              result.conversations.forEach((chat) => {
                const savedAt = typeof chat.savedAt === "number" ? new Date(chat.savedAt * 1000).toISOString() : chat.savedAt;
                merged.set(chat.id, { ...chat, savedAt, messages: Array.isArray(chat.messages) ? chat.messages : [] });
              });
              return [...merged.values()].sort((a, b) => new Date(b.savedAt || 0) - new Date(a.savedAt || 0)).slice(0, 100);
            });
          }
        } catch { /* Keep the browser cache available when sync is offline. */ }
      })
      .catch(() => {});
    return () => { active = false; };
  }, []);

  useEffect(() => {
    const token = new URLSearchParams(window.location.search).get("share");
    if (!token) return;
    let active = true;
    setSharedReadOnly(true);
    fetch("/api/shared/" + encodeURIComponent(token), { credentials: "same-origin" })
      .then(async (response) => {
        const data = await response.json().catch(() => ({}));
        if (!response.ok) throw new Error(data.error || "This share link is unavailable.");
        if (!active) return;
        const payload = data.payload || {};
        setMessages(Array.isArray(payload.messages) ? payload.messages.map((item) => ({ ...item, id: crypto.randomUUID() })) : []);
        setApiHistory(Array.isArray(payload.apiHistory) ? payload.apiHistory : []);
        setLastImage(payload.lastImage || "");
        setEditVersions(Array.isArray(payload.editVersions) ? payload.editVersions : []);
        setEditVersionIndex(Number.isInteger(payload.editVersionIndex) ? payload.editVersionIndex : -1);
        setError("Read-only shared conversation. Shared links expire after 7 days.");
      })
      .catch((error) => { if (active) setError(error instanceof Error ? error.message : "Unable to load shared conversation."); });
    return () => { active = false; };
  }, []);

  async function submitPasswordChange(event) {
    event.preventDefault();
    setPasswordError("");
    setPasswordNotice("");
    if (!passwordFields.currentPassword || !passwordFields.newPassword || !passwordFields.confirmPassword) {
      setPasswordError("Fill in all password fields.");
      return;
    }
    if (passwordFields.newPassword !== passwordFields.confirmPassword) {
      setPasswordError("New passwords do not match.");
      return;
    }
    setPasswordBusy(true);
    try {
      const response = await fetch("/api/auth/change-password", {
        method: "POST",
        credentials: "same-origin",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ currentPassword: passwordFields.currentPassword, newPassword: passwordFields.newPassword }),
      });
      const data = await response.json().catch(() => ({}));
      if (!response.ok) throw new Error(data.error || "Could not change password.");
      setPasswordFields({ currentPassword: "", newPassword: "", confirmPassword: "" });
      setPasswordNotice("Password changed successfully.");
    } catch (error) {
      setPasswordError(error instanceof Error ? error.message : "Could not change password.");
    } finally {
      setPasswordBusy(false);
    }
  }

  async function signOut() {
    try {
      await fetch("/api/auth/logout", { method: "POST", credentials: "same-origin" });
    } finally {
      // Account conversations must not remain visible after the account session ends.
      // This cache currently mixes local and synced chats, so clear it to avoid
      // exposing the signed-out user's private history on a shared browser.
      setUser(null);
      setSavedChats([]);
      setMessages([]);
      setApiHistory([]);
      setLastImage("");
      setEditVersions([]);
      setEditVersionIndex(-1);
      setPanel("");
      setError("");
      activeConversationId.current = "";
      try {
        localStorage.removeItem("lumina_react_chats");
        localStorage.removeItem("lumina_react_active_chat");
      } catch { /* The signed-out UI is still cleared if browser storage is unavailable. */ }
    }
  }

  useEffect(() => {
    try { localStorage.setItem("lumina_react_gallery", JSON.stringify(savedGallery.slice(0, 12))); }
    catch { setError("Browser storage is full. Remove saved images and try again."); }
  }, [savedGallery]);

  useEffect(() => {
    try { localStorage.setItem("lumina_react_chats", JSON.stringify(savedChats.slice(0, 100))); } catch {}
  }, [savedChats]);

  function runOperation(operation) {
    setPrompt(operation);
    setPanel("");
    setError("");
  }

  async function checkApiStatus() {
    setPanel("status");
    setApiStatus("Checking backend…");
    try {
      const response = await fetch("/health", { credentials: "same-origin" });
      const data = await response.json().catch(() => ({}));
      setApiStatus(response.ok ? "Backend is reachable." : `Backend returned HTTP ${response.status}.`);
    } catch {
      setApiStatus("Backend could not be reached. Check that Flask is running.");
    }
  }

  function toggleVoiceInput() {
    const SpeechRecognition = window.SpeechRecognition || window.webkitSpeechRecognition;
    if (!SpeechRecognition) {
      setError("Voice input is not supported in this browser. Try a recent version of Chrome.");
      return;
    }
    if (listening) {
      recognitionRef.current?.stop();
      setListening(false);
      return;
    }
    const recognition = new SpeechRecognition();
    recognition.lang = "en-US";
    recognition.interimResults = true;
    recognition.continuous = false;
    recognition.onresult = (event) => {
      const transcript = Array.from(event.results).map((result) => result[0].transcript).join("");
      setPrompt(transcript);
    };
    recognition.onerror = () => { setListening(false); setError("Voice input failed. Please try again or type your prompt."); };
    recognition.onend = () => setListening(false);
    recognitionRef.current = recognition;
    setListening(true);
    setError("");
    recognition.start();
  }

  function persistChatSnapshot(messageList, history = apiHistory, image = lastImage, versions = editVersions, versionIndex = editVersionIndex) {
    if (!Array.isArray(messageList) || !messageList.length || sharedReadOnly) return;
    const firstPrompt = messageList.find((item) => item.role === "user")?.text || "Lumina conversation";
    const id = activeConversationId.current || crypto.randomUUID();
    activeConversationId.current = id;
    try { localStorage.setItem("lumina_react_active_chat", id); } catch {}
    const chat = {
      id,
      title: firstPrompt.slice(0, 52) || "Lumina conversation",
      savedAt: new Date().toISOString(),
      messages: messageList.map((item) => ({
        role: item.role,
        text: item.text || "",
        ...(Array.isArray(item.images) && item.images.length ? { images: item.images } : {}),
        ...(item.failed ? { failed: true } : {}),
      })),
      apiHistory: history,
      lastImage: image,
      editVersions: versions,
      editVersionIndex: versionIndex,
    };
    const payload = {
      messages: chat.messages,
      apiHistory: chat.apiHistory,
      lastImage: chat.lastImage,
      editVersions: chat.editVersions,
      editVersionIndex: chat.editVersionIndex,
    };
    setSavedChats((current) => {
      const next = [chat, ...current.filter((item) => item.id !== id)].slice(0, 100);
      try { localStorage.setItem("lumina_react_chats", JSON.stringify(next)); } catch {}
      return next;
    });
    if (user) {
      fetch("/api/conversations", {
        method: "POST",
        credentials: "same-origin",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ id: chat.id, title: chat.title, payload }),
      }).then(async (response) => {
        if (!response.ok) {
          const result = await response.json().catch(() => ({}));
          setError(result.error || "Conversation is saved in this browser, but cloud sync failed.");
        }
      }).catch(() => setError("Conversation is saved in this browser, but cloud sync is unavailable."));
    }
  }

  function saveCurrentChat() {
    if (!messages.length) { setError("Start a conversation before saving it."); return; }
    persistChatSnapshot(messages);
    setPanel("history");
    setError(user ? "" : "Saved in this browser. Sign in to sync conversations with Supabase.");
  }
  async function deleteSavedChat(chat) {
    setSavedChats((current) => current.filter((item) => item.id !== chat.id));
    if (!user) return;
    try {
      const response = await fetch("/api/conversations/" + encodeURIComponent(chat.id), {
        method: "DELETE",
        credentials: "same-origin",
      });
      if (!response.ok && response.status !== 404) {
        const result = await response.json().catch(() => ({}));
        setError(result.error || "Conversation was removed locally, but cloud deletion failed.");
      }
    } catch {
      setError("Conversation was removed locally, but cloud deletion could not be confirmed.");
    }
  }

  async function shareCurrentChat() {
    if (!messages.length) { setError("Start a conversation before sharing it."); return; }
    if (!user) { setAuthOpen(true); setError("Sign in to create a share link."); return; }
    setError("");
    try {
      async function makeShareableImage(src) {
        if (typeof src !== "string" || !src.startsWith("blob:")) return src;
        const blob = await (await fetch(src)).blob();
        return await new Promise((resolve, reject) => {
          const reader = new FileReader();
          reader.onload = () => resolve(String(reader.result));
          reader.onerror = () => reject(new Error("An attached image could not be prepared for sharing."));
          reader.readAsDataURL(blob);
        });
      }
      const sharedMessages = await Promise.all(messages.map(async ({ role, text, images }) => ({
        role,
        text: text || "",
        ...(Array.isArray(images) && images.length ? { images: await Promise.all(images.map(makeShareableImage)) } : {}),
      })));
      const sharedVersions = await Promise.all(editVersions.map(async (version) => ({ ...version, src: await makeShareableImage(version.src) })));
      const response = await fetch("/api/share", {
        method: "POST",
        credentials: "same-origin",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ payload: { messages: sharedMessages, apiHistory, lastImage, editVersions: sharedVersions, editVersionIndex } }),
      });
      const data = await response.json().catch(() => ({}));
      if (!response.ok) throw new Error(data.error || "Could not create a share link.");
      const url = new URL(window.location.href);
      url.search = "";
      url.hash = "";
      url.searchParams.set("share", data.token);
      await navigator.clipboard.writeText(url.toString());
      setError("Share link copied. It expires in 7 days; anyone with the link can view this conversation.");
    } catch (error) {
      setError(error instanceof Error ? error.message : "Could not create or copy a share link.");
    }
  }

  function restoreChat(chat) {
    activeConversationId.current = chat.id || crypto.randomUUID();
    try { localStorage.setItem("lumina_react_active_chat", activeConversationId.current); } catch {}
    setMessages((chat.messages || []).map((item) => ({ ...item, id: crypto.randomUUID() })));
    setApiHistory(Array.isArray(chat.apiHistory) ? chat.apiHistory : []);
    setEditVersions(Array.isArray(chat.editVersions) ? chat.editVersions : []);
    setEditVersionIndex(Number.isInteger(chat.editVersionIndex) ? chat.editVersionIndex : -1);
    setLastImage(chat.lastImage || "");
    setPanel("");
    setError("");
  }

  function recordEditVersion(src) {
    if (!src) return;
    const normalized = src.startsWith("data:") || src.startsWith("http") ? src : "data:image/png;base64," + src;
    setEditVersions((current) => {
      const base = editVersionIndex >= 0 ? current.slice(0, editVersionIndex + 1) : current;
      const next = [...base, { src: normalized, createdAt: Date.now() }];
      const trimmed = next.slice(-20);
      setEditVersionIndex(trimmed.length - 1);
      return trimmed;
    });
    setLastImage(src.startsWith("data:") ? src.split(",", 2)[1] : src);
  }

  function undoImageEdit() {
    if (editVersionIndex <= 0) return;
    const nextIndex = editVersionIndex - 1;
    setEditVersionIndex(nextIndex);
    const src = editVersions[nextIndex]?.src || "";
    setLastImage(src.startsWith("data:") ? src.split(",", 2)[1] : src);
  }

  function redoImageEdit() {
    if (editVersionIndex < 0 || editVersionIndex >= editVersions.length - 1) return;
    const nextIndex = editVersionIndex + 1;
    setEditVersionIndex(nextIndex);
    const src = editVersions[nextIndex]?.src || "";
    setLastImage(src.startsWith("data:") ? src.split(",", 2)[1] : src);
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

  function chooseFiles(fileList) {
    const incoming = Array.from(fileList || []);
    if (!incoming.length) return;
    const allowedTypes = ["image/jpeg", "image/png", "image/webp", "image/gif", "image/bmp"];
    const accepted = [];
    for (const nextFile of incoming) {
      if (!allowedTypes.includes(nextFile.type)) { setError(nextFile.name + ": use a JPG, PNG, WEBP, GIF, or BMP image."); continue; }
      if (nextFile.size > 10 * 1024 * 1024) { setError(nextFile.name + ": images must be 10 MB or smaller."); continue; }
      accepted.push({ id: crypto.randomUUID(), file: nextFile, preview: URL.createObjectURL(nextFile) });
    }
    setAttachments((current) => {
      const combined = [...current, ...accepted];
      const allowed = combined.slice(0, 5);
      combined.slice(5).forEach((item) => URL.revokeObjectURL(item.preview));
      if (combined.length > 5) setError("You can attach up to 5 images per message.");
      return allowed;
    });
    if (fileInput.current) fileInput.current.value = "";
  }

  function removeAttachment(id) {
    setAttachments((current) => {
      const item = current.find((attachment) => attachment.id === id);
      if (item) URL.revokeObjectURL(item.preview);
      return current.filter((attachment) => attachment.id !== id);
    });
  }

  function clearAttachments() {
    // Keep object URLs alive because the sent user message still displays these previews.
    setAttachments([]);
    if (fileInput.current) fileInput.current.value = "";
  }

  function stopGeneration() {
    requestController.current?.abort();
    requestController.current = null;
    setBusy(false);
    setError("Generation stopped.");
  }

  function getProgressLabel(promptText, hasAttachments) {
    const value = String(promptText || "").toLowerCase();
    if (/\b(generate|create|draw|make)\b/.test(value) && /\b(image|picture|illustration|wallpaper|art)\b/.test(value)) return "Creating your image…";
    if (hasAttachments || /\b(edit|enhance|restore|remove background|denoise|upscale|rotate|crop|filter)\b/.test(value)) return "Processing your image…";
    if (/\b(code|python|javascript|debug|function|class|algorithm|program|sql|api)\b/.test(value)) return "Structuring your solution…";
    if (/\b(table|chart|compare|comparison|diagram|summarize|organize)\b/.test(value)) return "Organizing the details…";
    return "Forming a response…";
  }

  async function submitPrompt(event) {
    event.preventDefault();
    const text = prompt.trim();
    if (sharedReadOnly || (!text && !attachments.length) || busy) return;

    const userMessage = {
      id: crypto.randomUUID(),
      role: "user",
      text,
      images: attachments.map((attachment) => attachment.preview),
    };
    setMessages((current) => [...current, userMessage]);
    setProgressLabel(getProgressLabel(text, attachments.length > 0));
    setPrompt("");
    setError("");
    setBusy(true);
    const controller = new AbortController();
    requestController.current = controller;

    try {
      const batch = attachments.length ? attachments : [null];
      const answers = [];
      const resultImages = [];
      let currentHistory = apiHistory;
      const originalAttachmentPreview = attachments[0]?.preview || "";
      let currentLastImage = lastImage;
      for (let index = 0; index < batch.length; index++) {
        if (controller.signal.aborted) throw new DOMException("Request cancelled", "AbortError");
        const body = new FormData();
        body.append("prompt", text || (batch.length > 1 ? "Analyze this image (" + (index + 1) + " of " + batch.length + ")." : "Analyze this image."));
        body.append("history", JSON.stringify(currentHistory));
        if (currentLastImage) body.append("last_image", currentLastImage);
        if (batch[index]?.file) body.append("image", batch[index].file);
        const response = await fetch("/process", { method: "POST", body, signal: controller.signal });
        const data = await response.json().catch(() => ({}));
        if (!response.ok) throw new Error(data.message || data.error || "Lumina couldn't process image " + (index + 1) + ".");
        answers.push(data.message || data.reply || data.response || data.text || "Lumina completed the request.");
        const images = Array.isArray(data.images) ? data.images : Array.isArray(data.result_images) ? data.result_images : [];
        const validImages = [...images, ...(data.image ? [data.image] : [])].filter((src) => typeof src === "string" && src.trim().length > 0);
        for (const src of validImages) if (!resultImages.includes(src)) resultImages.push(src);
        if (Array.isArray(data.history)) currentHistory = data.history;
        if (typeof data.last_image === "string") currentLastImage = data.last_image;
      }
      setApiHistory(currentHistory);
      setLastImage(currentLastImage);
      if (resultImages.length) {
        // Compute the next history outside a state updater: React may invoke updater
        // functions more than once, and setting another state inside one is unsafe.
        const base = editVersionIndex >= 0 ? editVersions.slice(0, editVersionIndex + 1) : editVersions;
        const withOriginal = base.length ? base : (originalAttachmentPreview ? [{ src: originalAttachmentPreview, createdAt: Date.now(), label: "Original" }] : []);
        const normalized = resultImages.map((src) => ({
          src: src.startsWith("data:") || src.startsWith("blob:") || src.startsWith("http") ? src : "data:image/png;base64," + src,
          createdAt: Date.now(),
          label: "Edit",
        }));
        const next = [...withOriginal, ...normalized].slice(-20);
        setEditVersions(next);
        setEditVersionIndex(next.length - 1);
      }
      const assistantMessage = { id: crypto.randomUUID(), role: "assistant", text: answers.join("\n\n"), images: resultImages };
      const completedMessages = [...messages, userMessage, assistantMessage];
      setMessages(completedMessages);
      const completedVersions = resultImages.length
        ? (editVersionIndex >= 0 ? editVersions.slice(0, editVersionIndex + 1) : editVersions).concat(
            resultImages.map((src) => ({
              src: src.startsWith("data:") || src.startsWith("blob:") || src.startsWith("http") ? src : "data:image/png;base64," + src,
              createdAt: Date.now(),
              label: "Edit",
            }))
          ).slice(-20)
        : editVersions;
      const completedVersionIndex = resultImages.length ? completedVersions.length - 1 : editVersionIndex;
      persistChatSnapshot(completedMessages, currentHistory, currentLastImage, completedVersions, completedVersionIndex);
      clearAttachments();
    } catch (err) {
      if (err?.name === "AbortError") return;
      const message = err instanceof Error ? err.message : "Something went wrong.";
      setError(message);
      const failedMessage = { id: crypto.randomUUID(), role: "assistant", text: message, failed: true };
      const failedMessages = [...messages, userMessage, failedMessage];
      setMessages(failedMessages);
      persistChatSnapshot(failedMessages);
    } finally {
      if (requestController.current === controller) requestController.current = null;
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
        <button className="new-chat" onClick={() => { activeConversationId.current = ""; try { localStorage.removeItem("lumina_react_active_chat"); } catch {} setMessages([]); setApiHistory([]); setLastImage(""); setEditVersions([]); setEditVersionIndex(-1); setError(""); setPanel(""); }}>
          <PanelLeft size={17} /> New conversation
        </button>
        <button className="side-action" onClick={() => setPanel(panel === "history" ? "" : "history")}><History size={16} /> Saved conversations <span>{savedChats.length}</span></button>
        <button className="side-action" onClick={() => setPanel(panel === "gallery" ? "" : "gallery")}><Images size={16} /> Saved images <span>{savedGallery.length}</span></button>
        <button className="side-action" onClick={saveCurrentChat}><BookmarkPlus size={16} /> Save conversation</button>
        <button className="side-action" onClick={shareCurrentChat}><Share2 size={16} /> Share conversation</button>
        <button className="side-action" onClick={() => setPanel(panel === "operations" ? "" : "operations")}><WandSparkles size={16} /> Operations</button>
        <button className="side-action" onClick={checkApiStatus}><Activity size={16} /> API status</button>
        <div className="sidebar-note">
          <span className="status-dot" />
          <span>AI workspace</span>
        </div>
        <div className="sidebar-bottom">
          <div className="avatar">{user ? (user.firstName || user.username || "U").slice(0, 1).toUpperCase() : "L"}</div>
          <div className="account-details"><strong>{user ? (user.firstName || user.username || "Account") : "Lumina"}</strong><small>{user ? (user.email || user.username) : "Image assistant"}</small></div>
          {user ? <><button className="account-action" onClick={() => { setPasswordModalOpen(true); setPasswordError(""); setPasswordNotice(""); }}>Change password</button><button className="account-action" onClick={signOut}>Sign out</button></> : <button className="account-action" onClick={() => setAuthOpen(true)}>Sign in</button>}
        </div>
      </aside>

      <section className="workspace">
        <header className="topbar">
          <div className="mobile-brand"><span className="brand-mark"><Sparkles size={17} /></span> Lumina</div>
          <button className="mobile-menu-button" type="button" onClick={() => setPanel("menu")} aria-label="Open navigation menu"><PanelLeft size={18} /></button>
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
              {messages.map((message, messageIndex) => (
                <article className={`message message-${message.role}`} key={message.id}>
                  <div className="message-avatar">{message.role === "user" ? "Y" : <Sparkles size={16} />}</div>
                  <div className="message-content">
                    <strong>{message.role === "user" ? "You" : "Lumina"}</strong>
                    {message.text && (message.role === "assistant" ? <StructuredResponse text={message.text} onCopy={(value) => navigator.clipboard.writeText(value).catch(() => setError("Clipboard access failed. Please copy manually."))} /> : <p>{message.text}</p>)}
                    {message.image && <img className="message-image" src={message.image} alt="Uploaded input" />}
                    {message.role === "user" && message.images?.map((src, index) => <img className="message-image attachment-message-image" key={index} src={src} alt={"Attached image " + (index + 1)} />)}
                    {message.role !== "user" && message.images?.map((src, index) => {
                      const isCurrentHistoryImage = message.role === "assistant" && messageIndex === messages.length - 1 && index === 0 && editVersionIndex >= 0 && editVersions[editVersionIndex];
                      const displaySrc = isCurrentHistoryImage ? editVersions[editVersionIndex].src : src;
                      const imageSrc = displaySrc.startsWith("data:") || displaySrc.startsWith("blob:") || displaySrc.startsWith("http") ? displaySrc : "data:image/png;base64," + displaySrc;
                      const normalizedSrc = imageSrc;
                      return <div className="result-image-wrap" key={index}>
                        <button className="image-open-button" onClick={() => setLightbox(imageSrc)} aria-label="Open image preview"><img className="message-image" src={imageSrc} alt={"Generated or edited result " + (index + 1)} /></button>
                        <div className="image-action-row"><button className="save-image-button" onClick={() => saveImage(normalizedSrc)}><BookmarkPlus size={14} /> Save image</button><a className="image-download-link" href={imageSrc} download="lumina-image.png">Download</a><button className="image-download-link" onClick={() => copyImage(normalizedSrc)}>Copy image</button><button className="image-download-link" onClick={() => { setLastImage(normalizedSrc.startsWith("data:") ? normalizedSrc.split(",", 2)[1] : normalizedSrc.replace(/^blob:/, "")); setPrompt("Edit this image: "); }}>Use for edit</button><button className="image-download-link" disabled={editVersionIndex <= 0} onClick={undoImageEdit}>Undo edit</button><button className="image-download-link" disabled={editVersionIndex < 0 || editVersionIndex >= editVersions.length - 1} onClick={redoImageEdit}>Redo edit</button><button className="image-download-link" disabled={editVersionIndex <= 0} onClick={() => { setComparePosition(50); setCompareImages({ before: editVersions[editVersionIndex - 1]?.src, after: editVersions[editVersionIndex]?.src }); }}>Compare</button></div>
                      </div>;
                    })}
                  </div>
                </article>
              ))}
              {busy && <div className="loading-line"><LoaderCircle className="spin" size={17} /> {progressLabel}</div>}
            </div>
          )}
        </div>

        <div className="composer-area">
          {error && <div className="error-banner" role="alert">{error}</div>}
          <form className="composer" onSubmit={submitPrompt}>
            {attachments.length > 0 && <div className="attachment-strip" aria-label="Selected image attachments">{attachments.map((item, index) => <div className="attachment" key={item.id}><button type="button" className="attachment-preview-trigger" onClick={() => setLightbox(item.preview)} aria-label={"Preview image " + (index + 1)}><img src={item.preview} alt={"Selected upload preview " + (index + 1)} onError={(event) => { event.currentTarget.style.display = "none"; event.currentTarget.parentElement.parentElement.classList.add("attachment-preview-error"); }} /></button><span>{index + 1}</span><button type="button" className="attachment-remove" aria-label={"Remove image " + (index + 1)} onClick={() => removeAttachment(item.id)}><X size={15} /></button></div>)}</div>}
            <textarea
              disabled={sharedReadOnly}
              value={prompt}
              onPaste={(event) => {
                const imageItem = Array.from(event.clipboardData?.items || []).find((item) => item.type.startsWith("image/"));
                const pastedImage = imageItem?.getAsFile();
                if (pastedImage) {
                  event.preventDefault();
                  chooseFiles([new File([pastedImage], "pasted-image.png", { type: pastedImage.type || "image/png" })]);
                }
              }}
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
                  accept="image/jpeg,image/png,image/webp,image/gif,image/bmp"
                  multiple
                  hidden
                  onChange={(event) => chooseFiles(event.target.files)}
                />
                <button className="tool-button" type="button" onClick={() => fileInput.current?.click()} disabled={sharedReadOnly}>
                  <ImagePlus size={17} /><span>Add image</span>
                </button>
                <button className="tool-button" type="button" onClick={toggleVoiceInput} title="Voice input" disabled={sharedReadOnly}>
                  <Mic size={17} /><span>{listening ? "Listening…" : "Voice"}</span>
                </button>
                <span className="hint">Shift + Enter for a new line</span>
              </div>
              <button className={`send-button${busy ? " stop-button" : ""}`} type={busy ? "button" : "submit"} onClick={busy ? stopGeneration : undefined} disabled={sharedReadOnly || (!busy && (!prompt.trim() && !attachments.length))} aria-label={busy ? "Stop generation" : "Send message"} title={busy ? "Stop generation" : "Send message"}>
                {busy ? <X size={17} /> : <ArrowUp size={18} />}
              </button>
            </div>
          </form>
          <p className="disclaimer">Lumina can make mistakes. Review generated results before using them.</p>
        </div>
      </section>
      {compareImages && <div className="compare-backdrop" role="presentation" onMouseDown={(event) => { if (event.target === event.currentTarget) setCompareImages(null); }}>
        <section className="compare-dialog" role="dialog" aria-modal="true" aria-label="Compare image edits">
          <header className="compare-header"><strong>Before and after</strong><button type="button" onClick={() => setCompareImages(null)} aria-label="Close comparison"><X size={18} /></button></header>
          <div className="compare-slider" style={{ "--compare-position": comparePosition + "%" }}>
            <img className="compare-after" src={compareImages.after} alt="Current image version" />
            <div className="compare-before-clip" style={{ "--compare-position": comparePosition + "%" }}><img src={compareImages.before} alt="Previous image version" /></div>
            <div className="compare-slider-divider" style={{ left: comparePosition + "%" }} aria-hidden="true"><span>↔</span></div>
          </div>
          <div className="compare-slider-labels"><span>Previous version</span><span>Current version</span></div>
          <label className="compare-slider-control">Drag to compare<input aria-label="Before and after comparison position" type="range" min="0" max="100" value={comparePosition} onChange={(event) => setComparePosition(Number(event.target.value))} /></label>
        </section>
      </div>}
      {lightbox && <div className="lightbox-backdrop" onMouseDown={(event) => { if (event.target === event.currentTarget) setLightbox(""); }} role="presentation">
        <button className="lightbox-close" onClick={() => setLightbox("")} aria-label="Close image preview"><X size={20} /></button>
        <img src={lightbox} alt="Expanded Lumina result" />
        <div className="lightbox-actions"><a href={lightbox} download="lumina-image.png">Download image</a><button type="button" onClick={() => saveImage(lightbox)}><BookmarkPlus size={14} /> Save to gallery</button><button type="button" onClick={() => copyImage(lightbox)}><Copy size={14} /> Copy image</button></div>
      </div>}
      {panel && <div className="drawer-backdrop" onMouseDown={(event) => { if (event.target === event.currentTarget) setPanel(""); }}>
        <section className="drawer">
          <header className="drawer-header"><div><h2>{panel === "menu" ? "Lumina menu" : panel === "gallery" ? "Saved images" : panel === "operations" ? "Image operations" : panel === "status" ? "API status" : "Saved conversations"}</h2><p>{panel === "menu" ? "Navigate your workspace." : panel === "operations" ? "Choose an operation, add an image if needed, then send the prompt." : panel === "status" ? "Local Flask backend health" : "Saved locally in this browser"}</p></div><button className="drawer-close" onClick={() => setPanel("")} aria-label="Close panel"><X size={18} /></button></header>
          {panel === "menu" ? <div className="mobile-menu-actions"><button className="side-action" onClick={() => { activeConversationId.current = ""; try { localStorage.removeItem("lumina_react_active_chat"); } catch {} setMessages([]); setApiHistory([]); setLastImage(""); setEditVersions([]); setEditVersionIndex(-1); setError(""); setPanel(""); }}><PanelLeft size={16} /> New conversation</button><button className="side-action" onClick={() => setPanel("history")}><History size={16} /> Saved conversations <span>{savedChats.length}</span></button><button className="side-action" onClick={() => setPanel("gallery")}><Images size={16} /> Saved images <span>{savedGallery.length}</span></button><button className="side-action" onClick={() => { setPanel("operations"); }}><WandSparkles size={16} /> Operations</button><button className="side-action" onClick={checkApiStatus}><Activity size={16} /> API status</button><button className="side-action" onClick={saveCurrentChat}><BookmarkPlus size={16} /> Save conversation</button><button className="side-action" onClick={shareCurrentChat}><Share2 size={16} /> Share conversation</button>{user ? <><button className="side-action" onClick={() => { setPanel(""); setPasswordModalOpen(true); }}>Change password</button><button className="side-action" onClick={signOut}>Sign out</button></> : <button className="side-action" onClick={() => { setPanel(""); setAuthOpen(true); }}>Sign in</button>}</div> : panel === "operations" ? <div className="operations-content">{OPERATION_GROUPS.map(([group, operations]) => <section className="operation-group" key={group}><h3>{group}</h3><div className="operation-chips">{operations.map(([label, operation]) => <button type="button" className="operation-chip" key={label} onClick={() => runOperation(operation)}>{label}</button>)}</div></section>)}<button type="button" className="operation-chip operation-generate" onClick={() => runOperation("Generate an image from this description: ")}>✨ Generate from text</button></div> : panel === "status" ? <div className="status-content"><p>{apiStatus || "Checking backend…"}</p><button type="button" className="operation-chip" onClick={checkApiStatus}>Check again</button></div> : panel === "gallery" ? (savedGallery.length ? <div className="saved-gallery">{savedGallery.map((item) => <article className="saved-image-card" key={item.id}><img src={item.src} alt={item.prompt} /><p title={item.prompt}>{item.prompt}</p><div className="saved-image-actions"><a href={item.src} download="lumina-image.png">Download</a><button onClick={() => setSavedGallery((current) => current.filter((saved) => saved.id !== item.id))} aria-label="Remove saved image"><Trash2 size={14} /></button></div></article>)}</div> : <p className="drawer-empty">No saved images yet. Generate an image and select Save image.</p>) : (savedChats.length ? <div className="saved-chat-list">{savedChats.map((chat) => <article className="saved-chat-card" key={chat.id}><button className="saved-chat-open" onClick={() => restoreChat(chat)}><strong>{chat.title}</strong><small>{new Date(chat.savedAt).toLocaleString()} · {chat.messages.length} messages</small></button><button className="saved-chat-delete" onClick={() => deleteSavedChat(chat)} aria-label="Delete saved conversation"><Trash2 size={14} /></button></article>)}</div> : <p className="drawer-empty">No saved conversations yet. Use Save conversation to keep a text copy.</p>)}
        </section>
      </div>}
      {passwordModalOpen && <div className="modal-backdrop" role="presentation" onMouseDown={(event) => { if (event.target === event.currentTarget) setPasswordModalOpen(false); }}><section className="auth-card" role="dialog" aria-modal="true" aria-labelledby="change-password-heading"><button className="auth-close" type="button" onClick={() => setPasswordModalOpen(false)} aria-label="Close change password"><X size={18} /></button><div className="auth-mark"><Sparkles size={21} /></div><h2 id="change-password-heading">Change password</h2><p className="auth-subtitle">Confirm your current password and choose a strong new password.</p><form className="auth-form" onSubmit={submitPasswordChange}><label>Current password<input type="password" autoComplete="current-password" value={passwordFields.currentPassword} onChange={(event) => setPasswordFields((value) => ({ ...value, currentPassword: event.target.value }))} required /></label><label>New password<input type="password" autoComplete="new-password" minLength={8} value={passwordFields.newPassword} onChange={(event) => setPasswordFields((value) => ({ ...value, newPassword: event.target.value }))} required /></label><label>Confirm new password<input type="password" autoComplete="new-password" minLength={8} value={passwordFields.confirmPassword} onChange={(event) => setPasswordFields((value) => ({ ...value, confirmPassword: event.target.value }))} required /></label>{passwordError && <p className="auth-error" role="alert">{passwordError}</p>}{passwordNotice && <p className="auth-notice" role="status">{passwordNotice}</p>}<button className="auth-submit" type="submit" disabled={passwordBusy}>{passwordBusy ? "Updating…" : "Update password"}</button></form></section></div>}
      {authOpen && <AuthModal onClose={() => setAuthOpen(false)} onAuthenticated={(nextUser) => { setUser(nextUser); setAuthOpen(false); }} />}
    </main>
  );
}
