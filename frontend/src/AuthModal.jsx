import { useState } from "react";
import { X, Sparkles, LoaderCircle, ArrowLeft } from "lucide-react";

const initialForm = { identifier: "", password: "", firstName: "", lastName: "", email: "", username: "", confirm: "", otp: "" };

export default function AuthModal({ onClose, onAuthenticated }) {
  const [mode, setMode] = useState("login");
  const [form, setForm] = useState(initialForm);
  const [resetEmail, setResetEmail] = useState("");
  const [error, setError] = useState("");
  const [notice, setNotice] = useState("");
  const [busy, setBusy] = useState(false);

  function update(event) {
    setForm((current) => ({ ...current, [event.target.name]: event.target.value }));
  }

  async function postJson(url, payload) {
    const response = await fetch(url, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      credentials: "same-origin",
      body: JSON.stringify(payload),
    });
    const data = await response.json().catch(() => ({}));
    if (!response.ok || data.valid === false) {
      throw new Error(data.error || (response.status === 401 ? "Incorrect username/email or password." : "Request failed."));
    }
    return data;
  }

  async function submit(event) {
    event.preventDefault();
    setError("");
    setNotice("");
    if ((mode === "signup" || mode === "reset") && form.password !== form.confirm) {
      setError("Passwords do not match.");
      return;
    }
    setBusy(true);
    try {
      if (mode === "forgot") {
        const data = await postJson("/api/auth/send-otp", { email: form.identifier.trim() });
        setResetEmail(data.email || form.identifier.trim());
        setMode("reset");
        setNotice("A verification code has been sent to your registered email.");
        return;
      }
      if (mode === "reset") {
        await postJson("/api/auth/reset-password", { email: resetEmail, otp: form.otp.trim(), password: form.password });
        setMode("login");
        setForm((current) => ({ ...initialForm, identifier: resetEmail }));
        setNotice("Password reset successfully. Sign in with your new password.");
        return;
      }
      const isLogin = mode === "login";
      const data = await postJson(
        isLogin ? "/api/auth/verify-password" : "/api/auth/signup",
        isLogin
          ? { identifier: form.identifier.trim(), password: form.password }
          : { firstName: form.firstName.trim(), lastName: form.lastName.trim(), email: form.email.trim(), username: form.username.trim(), password: form.password },
      );
      if (!data.user) throw new Error("The server did not return an account session.");
      onAuthenticated(data.user);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Something went wrong.");
    } finally {
      setBusy(false);
    }
  }

  const isSignup = mode === "signup";
  const isForgot = mode === "forgot";
  const isReset = mode === "reset";
  const isLogin = mode === "login";
  const title = isLogin ? "Welcome back" : isSignup ? "Create your account" : isForgot ? "Reset your password" : "Enter verification code";
  const subtitle = isLogin ? "Sign in to continue with Lumina." : isSignup ? "Join Lumina and start creating." : isForgot ? "Enter your email or username to receive a one-time code." : "Enter the code sent to your registered email and choose a new password.";

  return (
    <div className="modal-backdrop" role="presentation" onMouseDown={(event) => { if (event.target === event.currentTarget) onClose(); }}>
      <section className="auth-card" role="dialog" aria-modal="true" aria-labelledby="auth-heading">
        <button className="auth-close" onClick={onClose} aria-label="Close sign in"><X size={18} /></button>
        <div className="auth-mark"><Sparkles size={21} /></div>
        <h2 id="auth-heading">{title}</h2>
        <p className="auth-subtitle">{subtitle}</p>
        <form onSubmit={submit} className="auth-form">
          {isSignup && <>
            <div className="auth-two">
              <label>First name<input name="firstName" value={form.firstName} onChange={update} required autoComplete="given-name" /></label>
              <label>Last name<input name="lastName" value={form.lastName} onChange={update} required autoComplete="family-name" /></label>
            </div>
            <label>Email<input name="email" type="email" value={form.email} onChange={update} required autoComplete="email" /></label>
            <label>Username<input name="username" value={form.username} onChange={update} required autoComplete="username" minLength={3} /></label>
          </>}
          {isLogin && <label>Email or username<input name="identifier" value={form.identifier} onChange={update} required autoComplete="username" /></label>}
          {isForgot && <label>Email or username<input name="identifier" value={form.identifier} onChange={update} required autoComplete="username" /></label>}
          {isReset && <>
            <label>Verification code<input name="otp" inputMode="numeric" pattern="[0-9]{6}" value={form.otp} onChange={update} required minLength={6} maxLength={6} autoComplete="one-time-code" /></label>
            <label>New password<input name="password" type="password" value={form.password} onChange={update} required minLength={8} autoComplete="new-password" /></label>
            <label>Confirm new password<input name="confirm" type="password" value={form.confirm} onChange={update} required minLength={8} autoComplete="new-password" /></label>
          </>}
          {(isLogin || isSignup) && <label>Password<input name="password" type="password" value={form.password} onChange={update} required autoComplete={isLogin ? "current-password" : "new-password"} minLength={8} /></label>}
          {isSignup && <label>Confirm password<input name="confirm" type="password" value={form.confirm} onChange={update} required autoComplete="new-password" minLength={8} /></label>}
          {error && <p className="auth-error" role="alert">{error}</p>}
          {notice && <p className="auth-notice" role="status">{notice}</p>}
          <button className="auth-submit" type="submit" disabled={busy}>
            {busy && <LoaderCircle size={16} className="spin" />}
            {busy ? "Please wait…" : isLogin ? "Sign in" : isSignup ? "Create account" : isForgot ? "Send verification code" : "Reset password"}
          </button>
        </form>
        <div className="auth-switch">
          {isLogin && <><button onClick={() => { setMode("forgot"); setError(""); setNotice(""); }}>Forgot password?</button><p>New to Lumina? <button onClick={() => { setMode("signup"); setError(""); setNotice(""); }}>Create account</button></p></>}
          {isSignup && <p>Already have an account? <button onClick={() => { setMode("login"); setError(""); setNotice(""); }}>Sign in</button></p>}
          {(isForgot || isReset) && <button onClick={() => { setMode("login"); setError(""); setNotice(""); }}> <ArrowLeft size={13} /> Back to sign in</button>}
        </div>
      </section>
    </div>
  );
}
