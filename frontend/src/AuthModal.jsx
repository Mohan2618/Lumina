import { useEffect, useRef, useState } from "react";
import { X, Sparkles, LoaderCircle, ArrowLeft } from "lucide-react";

const initialForm = { identifier: "", password: "", firstName: "", lastName: "", email: "", username: "", confirm: "", otp: "", dateOfBirth: "", acceptedTerms: false };

export default function AuthModal({ onClose, onAuthenticated }) {
  const [mode, setMode] = useState("login");
  const [form, setForm] = useState(initialForm);
  const [resetEmail, setResetEmail] = useState("");
  const [error, setError] = useState("");
  const [notice, setNotice] = useState("");
  const [busy, setBusy] = useState(false);
  const [showPasswords, setShowPasswords] = useState(false);
  const [lockoutSeconds, setLockoutSeconds] = useState(0);
  const [failedAttempts, setFailedAttempts] = useState(0);
  const [resendSeconds, setResendSeconds] = useState(0);
  const [otpDigits, setOtpDigits] = useState(["", "", "", "", "", ""]);
  const otpRefs = useRef([]);

  useEffect(() => {
    if (mode !== "reset" || resendSeconds <= 0) return;
    const timer = window.setTimeout(() => setResendSeconds((seconds) => Math.max(0, seconds - 1)), 1000);
    return () => window.clearTimeout(timer);
  }, [mode, resendSeconds]);

  useEffect(() => {
    if (lockoutSeconds <= 0) return;
    const timer = window.setTimeout(() => setLockoutSeconds((seconds) => Math.max(0, seconds - 1)), 1000);
    return () => window.clearTimeout(timer);
  }, [lockoutSeconds]);

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
    if (mode === "login" && lockoutSeconds > 0) { setError(`Please wait ${lockoutSeconds}s before trying again.`); return; }
    if (mode === "signup") {
      if (!/^[^\s@]+@[^\s@]+\.[^\s@]+$/.test(form.email.trim())) {
        setError("Enter a valid email address.");
        return;
      }
      if (!/^[a-zA-Z0-9_]{3,24}$/.test(form.username.trim())) {
        setError("Username must be 3–24 characters and use only letters, numbers, or underscores.");
        return;
      }
      if (form.password.length < 8 || !/[A-Za-z]/.test(form.password) || !/\d/.test(form.password)) {
        setError("Use at least 8 characters with at least one letter and one number.");
        return;
      }
      if (!form.dateOfBirth) {
        setError("Enter your date of birth to continue.");
        return;
      }
      const birthDate = new Date(form.dateOfBirth + "T00:00:00");
      const today = new Date();
      let age = today.getFullYear() - birthDate.getFullYear();
      const beforeBirthday = today.getMonth() < birthDate.getMonth() || (today.getMonth() === birthDate.getMonth() && today.getDate() < birthDate.getDate());
      if (beforeBirthday) age--;
      if (Number.isNaN(birthDate.getTime()) || birthDate > today || age < 13) {
        setError("You must be at least 13 years old to create an account.");
        return;
      }
      if (!form.acceptedTerms) {
        setError("Please accept the Terms of Service and Privacy Policy.");
        return;
      }
    }
    setBusy(true);
    try {
      if (mode === "forgot") {
        const data = await postJson("/api/auth/send-otp", { email: form.identifier.trim() });
        setResetEmail(data.email || form.identifier.trim());
        setResendSeconds(60);
        setOtpDigits(["", "", "", "", "", ""]);
        setForm((current) => ({ ...current, otp: "" }));
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
          : { firstName: form.firstName.trim(), lastName: form.lastName.trim(), email: form.email.trim(), username: form.username.trim(), password: form.password, dateOfBirth: form.dateOfBirth, acceptedTerms: form.acceptedTerms },
      );
      if (!data.user) throw new Error("The server did not return an account session.");
      if (isLogin) {
        setLockoutSeconds(0);
        setFailedAttempts(0);
      }
      onAuthenticated(data.user);
    } catch (err) {
      if (mode === "login") {
        const attempts = failedAttempts + 1;
        setFailedAttempts(attempts);
        if (attempts >= 5) {
          setFailedAttempts(0);
          setLockoutSeconds(30);
          setError("Too many sign-in attempts. Please wait 30 seconds before trying again.");
          return;
        }
      }
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
        <form onSubmit={submit} className="auth-form" autoComplete="off">
          {isSignup && <>
            <div className="auth-two">
              <label>First name<input name="firstName" value={form.firstName} onChange={update} required autoComplete="given-name" /></label>
              <label>Last name<input name="lastName" value={form.lastName} onChange={update} required autoComplete="family-name" /></label>
            </div>
            <label>Email<input name="email" type="email" value={form.email} onChange={update} required autoComplete="email" /></label>
            <label>Username<input name="username" value={form.username} onChange={update} required autoComplete="username" minLength={3} maxLength={24} pattern="[a-zA-Z0-9_]{3,24}" title="3–24 letters, numbers, or underscores" /></label>
            <label>Date of birth<input name="dateOfBirth" type="date" value={form.dateOfBirth} onChange={update} required max={new Date().toISOString().slice(0, 10)} /></label>
          </>}
          {isLogin && <label>Email or username<input name="identifier" value={form.identifier} onChange={update} required autoComplete="off" spellCheck={false} /></label>}
          {isForgot && <label>Email or username<input name="identifier" value={form.identifier} onChange={update} required autoComplete="username" /></label>}
          {isReset && <>
            <label>Verification code
              <div className="otp-digit-row" onPaste={(event) => {
                const pasted = event.clipboardData.getData("text").replace(/\D/g, "").slice(0, 6);
                if (!pasted) return;
                event.preventDefault();
                const next = Array.from({ length: 6 }, (_, index) => pasted[index] || "");
                setOtpDigits(next);
                setForm((current) => ({ ...current, otp: next.join("") }));
                otpRefs.current[Math.min(pasted.length, 5)]?.focus();
              }}>
                {otpDigits.map((digit, index) => <input key={index} ref={(element) => { otpRefs.current[index] = element; }} aria-label={"Verification digit " + (index + 1)} inputMode="numeric" pattern="[0-9]" value={digit} required maxLength={1} autoComplete={index === 0 ? "one-time-code" : "off"} onChange={(event) => {
                  const value = event.target.value.replace(/\D/g, "").slice(-1);
                  const next = [...otpDigits]; next[index] = value;
                  setOtpDigits(next);
                  setForm((current) => ({ ...current, otp: next.join("") }));
                  if (value && index < 5) otpRefs.current[index + 1]?.focus();
                }} onKeyDown={(event) => {
                  if (event.key === "Backspace" && !digit && index > 0) otpRefs.current[index - 1]?.focus();
                  if (event.key === "ArrowLeft" && index > 0) otpRefs.current[index - 1]?.focus();
                  if (event.key === "ArrowRight" && index < 5) otpRefs.current[index + 1]?.focus();
                }} />)}
              </div>
            </label>
            <label>New password<input name="password" type={showPasswords ? "text" : "password"} value={form.password} onChange={update} required minLength={8} autoComplete="new-password" /></label>
            <label>Confirm new password<input name="confirm" type={showPasswords ? "text" : "password"} value={form.confirm} onChange={update} required minLength={8} autoComplete="new-password" /></label>
          </>}
          {(isLogin || isSignup) && <label>Password<input name="password" type={showPasswords ? "text" : "password"} value={form.password} onChange={update} required autoComplete="new-password" minLength={8} /></label>}
          {isSignup && <label>Confirm password<input name="confirm" type={showPasswords ? "text" : "password"} value={form.confirm} onChange={update} required autoComplete="new-password" minLength={8} /></label>}
          {isSignup && <label className="auth-terms"><input type="checkbox" name="acceptedTerms" checked={form.acceptedTerms} onChange={(event) => setForm((current) => ({ ...current, acceptedTerms: event.target.checked }))} required /> <span>I agree to the Terms of Service and Privacy Policy.</span></label>}
          {(isLogin || isSignup || isReset) && <button className="password-visibility" type="button" onClick={() => setShowPasswords((shown) => !shown)}>{showPasswords ? "Hide passwords" : "Show passwords"}</button>}
          {isReset && <div className="resend-otp-control">{resendSeconds > 0 ? <span>Resend code in {resendSeconds}s</span> : <button type="button" disabled={busy} onClick={async () => { setError(""); setNotice(""); setBusy(true); try { await postJson("/api/auth/send-otp", { email: resetEmail }); setResendSeconds(60); setNotice("A new verification code has been sent."); } catch (err) { setError(err instanceof Error ? err.message : "Could not resend the code."); } finally { setBusy(false); } }}>Resend code</button>}</div>}
                    {error && <p className="auth-error" role="alert">{error}</p>}
          {notice && <p className="auth-notice" role="status">{notice}</p>}
          <button className="auth-submit" type="submit" disabled={busy}>
            {busy && <LoaderCircle size={16} className="spin" />}
            {busy ? "Please wait…" : isLogin && lockoutSeconds > 0 ? `Try again in ${lockoutSeconds}s` : isLogin ? "Sign in" : isSignup ? "Create account" : isForgot ? "Send verification code" : "Reset password"}
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
