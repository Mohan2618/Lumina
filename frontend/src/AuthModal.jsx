import { useState } from "react";
import { X, Sparkles, LoaderCircle } from "lucide-react";

export default function AuthModal({ onClose, onAuthenticated }) {
  const [mode, setMode] = useState("login");
  const [form, setForm] = useState({ identifier: "", password: "", firstName: "", lastName: "", email: "", username: "", confirm: "" });
  const [error, setError] = useState("");
  const [busy, setBusy] = useState(false);

  function update(event) {
    setForm((current) => ({ ...current, [event.target.name]: event.target.value }));
  }

  async function submit(event) {
    event.preventDefault();
    setError("");
    if (mode === "signup" && form.password !== form.confirm) {
      setError("Passwords do not match.");
      return;
    }
    setBusy(true);
    try {
      const endpoint = mode === "login" ? "/api/auth/verify-password" : "/api/auth/signup";
      const payload = mode === "login"
        ? { identifier: form.identifier.trim(), password: form.password }
        : { firstName: form.firstName.trim(), lastName: form.lastName.trim(), email: form.email.trim(), username: form.username.trim(), password: form.password };
      const response = await fetch(endpoint, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        credentials: "same-origin",
        body: JSON.stringify(payload),
      });
      const data = await response.json().catch(() => ({}));
      if (!response.ok || data.valid === false) {
        throw new Error(data.error || (response.status === 401 ? "Incorrect username/email or password." : "Unable to sign in."));
      }
      const user = data.user;
      if (!user) throw new Error("The server did not return an account session.");
      onAuthenticated(user);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Something went wrong.");
    } finally {
      setBusy(false);
    }
  }

  return (
    <div className="modal-backdrop" role="presentation" onMouseDown={(event) => { if (event.target === event.currentTarget) onClose(); }}>
      <section className="auth-card" role="dialog" aria-modal="true" aria-labelledby="auth-heading">
        <button className="auth-close" onClick={onClose} aria-label="Close sign in"><X size={18} /></button>
        <div className="auth-mark"><Sparkles size={21} /></div>
        <h2 id="auth-heading">{mode === "login" ? "Welcome back" : "Create your account"}</h2>
        <p className="auth-subtitle">{mode === "login" ? "Sign in to continue with Lumina." : "Join Lumina and start creating."}</p>
        <form onSubmit={submit} className="auth-form">
          {mode === "signup" && <>
            <div className="auth-two">
              <label>First name<input name="firstName" value={form.firstName} onChange={update} required autoComplete="given-name" /></label>
              <label>Last name<input name="lastName" value={form.lastName} onChange={update} required autoComplete="family-name" /></label>
            </div>
            <label>Email<input name="email" type="email" value={form.email} onChange={update} required autoComplete="email" /></label>
            <label>Username<input name="username" value={form.username} onChange={update} required autoComplete="username" minLength={3} /></label>
          </>}
          {mode === "login" && <label>Email or username<input name="identifier" value={form.identifier} onChange={update} required autoComplete="username" /></label>}
          <label>Password<input name="password" type="password" value={form.password} onChange={update} required autoComplete={mode === "login" ? "current-password" : "new-password"} minLength={8} /></label>
          {mode === "signup" && <label>Confirm password<input name="confirm" type="password" value={form.confirm} onChange={update} required autoComplete="new-password" minLength={8} /></label>}
          {error && <p className="auth-error" role="alert">{error}</p>}
          <button className="auth-submit" type="submit" disabled={busy}>
            {busy && <LoaderCircle size={16} className="spin" />}
            {busy ? "Please wait…" : mode === "login" ? "Sign in" : "Create account"}
          </button>
        </form>
        <p className="auth-switch">{mode === "login" ? "New to Lumina?" : "Already have an account?"}{" "}
          <button onClick={() => { setMode(mode === "login" ? "signup" : "login"); setError(""); }}>
            {mode === "login" ? "Create account" : "Sign in"}
          </button>
        </p>
      </section>
    </div>
  );
}
