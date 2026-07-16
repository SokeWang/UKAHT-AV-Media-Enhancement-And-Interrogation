import React, { useState } from 'react';
import { Lock, User, Compass, AlertCircle, Eye, EyeOff } from 'lucide-react';

interface LoginProps {
  onLoginSuccess: () => void;
}

const API_BASE = import.meta.env.VITE_API_BASE !== undefined 
  ? import.meta.env.VITE_API_BASE 
  : (typeof window !== 'undefined' && (window.location.hostname === 'localhost' || window.location.hostname === '127.0.0.1') ? 'http://localhost:8000' : '');

export const Login: React.FC<LoginProps> = ({ onLoginSuccess }) => {
  const [username, setUsername] = useState('');
  const [password, setPassword] = useState('');
  const [rememberMe, setRememberMe] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [loading, setLoading] = useState(false);
  const [showPassword, setShowPassword] = useState(false);

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    setError(null);

    if (!username.trim() || !password.trim()) {
      setError('Please fill in all fields.');
      return;
    }

    setLoading(true);

    try {
      const response = await fetch(`${API_BASE}/api/auth/login`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          username: username.trim(),
          password: password,
        }),
      });

      const data = await response.json();
      if (response.ok && data.code === 200) {
        if (rememberMe) {
          localStorage.setItem('ukaht_auth', 'true');
        } else {
          sessionStorage.setItem('ukaht_auth', 'true');
        }
        onLoginSuccess();
      } else {
        setError(data.message || 'Invalid username or password. Check the credentials below.');
        setLoading(false);
      }
    } catch (err) {
      console.error('Login error:', err);
      setError('Connection failed. Please ensure the backend service is running.');
      setLoading(false);
    }
  };

  return (
    <div style={{
      display: 'flex',
      alignItems: 'center',
      justifyContent: 'center',
      minHeight: '100vh',
      width: '100vw',
      padding: '20px',
      position: 'relative',
      overflow: 'hidden'
    }}>
      {/* Background Graphic Blobs */}
      <div style={{
        position: 'absolute',
        top: '-10%',
        left: '-10%',
        width: '50%',
        height: '60%',
        borderRadius: '50%',
        background: 'radial-gradient(circle, rgba(0, 102, 204, 0.08) 0%, transparent 70%)',
        zIndex: 0,
        pointerEvents: 'none'
      }} />
      <div style={{
        position: 'absolute',
        bottom: '-10%',
        right: '-10%',
        width: '60%',
        height: '60%',
        borderRadius: '50%',
        background: 'radial-gradient(circle, rgba(0, 180, 216, 0.08) 0%, transparent 70%)',
        zIndex: 0,
        pointerEvents: 'none'
      }} />

      {/* Main Glassmorphic Login Card */}
      <div className="glass-panel pulse-glow" style={{
        width: '100%',
        maxWidth: '440px',
        padding: '40px 30px',
        borderRadius: '20px',
        background: 'rgba(255, 255, 255, 0.85)',
        zIndex: 1,
        animation: 'slideUp 0.5s cubic-bezier(0.25, 0.8, 0.25, 1)',
        display: 'flex',
        flexDirection: 'column',
        gap: '24px'
      }}>
        {/* Branding & Logo */}
        <div style={{ textAlign: 'center', display: 'flex', flexDirection: 'column', alignItems: 'center', gap: '12px' }}>
          <div style={{
            width: '60px',
            height: '60px',
            borderRadius: '16px',
            background: 'linear-gradient(135deg, var(--accent-blue), var(--accent-cyan))',
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'center',
            boxShadow: '0 8px 24px rgba(0, 102, 204, 0.25)',
            color: '#ffffff'
          }}>
            <Compass size={32} className="animate-spin" style={{ animationDuration: '20s' }} />
          </div>
          <div>
            <h1 className="glow-text" style={{ 
              fontSize: '1.8rem', 
              fontWeight: 800, 
              fontFamily: 'var(--font-family-title)',
              letterSpacing: '-0.02em',
              marginTop: '8px'
            }}>
              UKAHT Portal
            </h1>
            <p style={{ 
              color: 'var(--text-secondary)', 
              fontSize: '0.85rem',
              fontWeight: 500,
              marginTop: '4px' 
            }}>
              UK Antarctic Heritage Trust · Archive Interrogation
            </p>
          </div>
        </div>

        {/* Login Form */}
        <form onSubmit={handleSubmit} style={{ display: 'flex', flexDirection: 'column', gap: '18px' }}>
          {error && (
            <div style={{
              display: 'flex',
              alignItems: 'center',
              gap: '8px',
              backgroundColor: 'rgba(255, 23, 68, 0.05)',
              border: '1px solid rgba(255, 23, 68, 0.2)',
              borderRadius: '8px',
              padding: '12px',
              color: 'var(--error)',
              fontSize: '0.85rem',
              animation: 'fadeIn 0.2s ease'
            }}>
              <AlertCircle size={16} style={{ flexShrink: 0 }} />
              <span>{error}</span>
            </div>
          )}

          {/* Username Field */}
          <div style={{ display: 'flex', flexDirection: 'column', gap: '6px' }}>
            <label style={{ fontSize: '0.85rem', fontWeight: 600, color: 'var(--text-primary)' }}>Username</label>
            <div style={{ position: 'relative', display: 'flex', alignItems: 'center' }}>
              <User size={18} style={{
                position: 'absolute',
                left: '14px',
                color: 'var(--text-muted)',
                pointerEvents: 'none'
              }} />
              <input
                type="text"
                value={username}
                onChange={(e) => setUsername(e.target.value)}
                placeholder="Enter username"
                disabled={loading}
                style={{
                  paddingLeft: '44px',
                  background: 'rgba(255, 255, 255, 0.9)'
                }}
              />
            </div>
          </div>

          {/* Password Field */}
          <div style={{ display: 'flex', flexDirection: 'column', gap: '6px' }}>
            <label style={{ fontSize: '0.85rem', fontWeight: 600, color: 'var(--text-primary)' }}>Password</label>
            <div style={{ position: 'relative', display: 'flex', alignItems: 'center' }}>
              <Lock size={18} style={{
                position: 'absolute',
                left: '14px',
                color: 'var(--text-muted)',
                pointerEvents: 'none'
              }} />
              <input
                type={showPassword ? 'text' : 'password'}
                value={password}
                onChange={(e) => setPassword(e.target.value)}
                placeholder="Enter password"
                disabled={loading}
                style={{
                  paddingLeft: '44px',
                  paddingRight: '44px',
                  width: '100%',
                  background: 'rgba(255, 255, 255, 0.9)',
                  border: '1px solid var(--border-color)',
                  borderRadius: '8px',
                  height: '46px',
                  outline: 'none',
                  fontSize: '0.95rem',
                  fontFamily: 'var(--font-family-body)',
                  transition: 'all var(--transition-fast)'
                }}
              />
              <button
                type="button"
                onClick={() => setShowPassword(!showPassword)}
                style={{
                  position: 'absolute',
                  right: '12px',
                  background: 'none',
                  border: 'none',
                  color: 'var(--text-muted)',
                  cursor: 'pointer',
                  padding: '4px',
                  display: 'flex',
                  alignItems: 'center',
                  justifyContent: 'center'
                }}
              >
                {showPassword ? <EyeOff size={18} /> : <Eye size={18} />}
              </button>
            </div>
          </div>

          {/* Remember Me Box */}
          <div style={{ display: 'flex', alignItems: 'center', gap: '8px', marginTop: '4px' }}>
            <input
              type="checkbox"
              id="remember"
              checked={rememberMe}
              onChange={(e) => setRememberMe(e.target.checked)}
              disabled={loading}
              style={{
                cursor: 'pointer',
                width: '16px',
                height: '16px',
                accentColor: 'var(--accent-blue)'
              }}
            />
            <label htmlFor="remember" style={{
              fontSize: '0.85rem',
              color: 'var(--text-secondary)',
              cursor: 'pointer',
              userSelect: 'none'
            }}>
              Keep me logged in
            </label>
          </div>

          {/* Submit Button */}
          <button
            type="submit"
            className="btn btn-primary"
            disabled={loading}
            style={{
              width: '100%',
              padding: '14px',
              borderRadius: '8px',
              fontSize: '1rem',
              marginTop: '10px',
              display: 'flex',
              alignItems: 'center',
              justifyContent: 'center',
              gap: '10px',
              cursor: loading ? 'not-allowed' : 'pointer'
            }}
          >
            {loading ? (
              <div style={{
                width: '20px',
                height: '20px',
                border: '2px solid rgba(255, 255, 255, 0.3)',
                borderTopColor: '#ffffff',
                borderRadius: '50%',
                animation: 'spin 0.6s linear infinite'
              }} />
            ) : (
              <span>Sign In</span>
            )}
          </button>
        </form>

      </div>
    </div>
  );
};
