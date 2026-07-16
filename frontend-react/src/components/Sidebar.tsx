import React from 'react';
import { Search, PlusCircle, Database, LogOut, RefreshCw } from 'lucide-react';

interface SidebarProps {
  onResetSearch: () => void;
  onOpenUpload: () => void;
  totalCount: number;
  originalCount: number;
  uploadedCount: number;
  onLogout: () => void;
  onSyncDatabase: () => void;
  syncing: boolean;
  syncStatusText: string;
}

export const Sidebar: React.FC<SidebarProps> = ({ 
  onResetSearch, 
  onOpenUpload,
  totalCount,
  originalCount,
  uploadedCount,
  onLogout,
  onSyncDatabase,
  syncing,
  syncStatusText
}) => {
  return (
    <aside className="glass-panel" style={{
      width: '280px',
      height: 'calc(100vh - 40px)',
      margin: '20px',
      padding: '30px 20px',
      display: 'flex',
      flexDirection: 'column',
      justifyContent: 'space-between',
      position: 'sticky',
      top: '20px',
      flexShrink: 0
    }}>
      <div>
        {/* Brand Header */}
        <div style={{ display: 'flex', alignItems: 'center', gap: '12px', marginBottom: '8px' }}>
          <span style={{ fontSize: '2rem' }}>🧊</span>
          <h2 className="glow-text" style={{ fontSize: '1.5rem', fontFamily: 'var(--font-family-title)' }}>
            UKAHT Archive
          </h2>
        </div>
        <p style={{
          color: 'var(--text-secondary)',
          fontSize: '0.85rem',
          lineHeight: '1.4',
          marginBottom: '30px',
          paddingLeft: '4px'
        }}>
          UK Antarctic Heritage Trust — Historical Image System
        </p>

        <div style={{
          height: '1px',
          background: 'var(--border-color)',
          marginBottom: '24px'
        }} />

        {/* Navigation Menu */}
        <nav style={{ display: 'flex', flexDirection: 'column', gap: '8px', marginBottom: '24px' }}>
          <button
            onClick={onResetSearch}
            className="btn btn-primary"
            style={{
              justifyContent: 'flex-start',
              width: '100%',
              padding: '14px 18px',
              color: '#ffffff'
            }}
          >
            <Search size={18} />
            <span>AI Chat Explorer</span>
          </button>

          <button
            onClick={onOpenUpload}
            className="btn btn-secondary"
            style={{
              justifyContent: 'flex-start',
              width: '100%',
              padding: '14px 18px'
            }}
          >
            <PlusCircle size={18} style={{ color: 'var(--accent-blue)' }} />
            <span>Upload New Image</span>
          </button>

          <button
            onClick={onSyncDatabase}
            className="btn btn-secondary"
            disabled={syncing}
            style={{
              justifyContent: 'flex-start',
              width: '100%',
              padding: '14px 18px',
              cursor: syncing ? 'not-allowed' : 'pointer'
            }}
          >
            <RefreshCw 
              size={18} 
              style={{ 
                color: 'var(--accent-cyan)',
                animation: syncing ? 'spin 1.5s linear infinite' : 'none'
              }} 
            />
            <span>{syncStatusText}</span>
          </button>
        </nav>

        {/* Stats Panel */}
        <div style={{
          background: 'rgba(94, 168, 241, 0.05)',
          border: '1px solid rgba(94, 168, 241, 0.1)',
          borderRadius: '10px',
          padding: '16px',
          display: 'flex',
          flexDirection: 'column',
          gap: '12px'
        }}>
          <div style={{ display: 'flex', alignItems: 'center', gap: '8px', fontSize: '0.85rem', fontWeight: 600, color: 'var(--text-primary)' }}>
            <Database size={15} style={{ color: 'var(--accent-cyan)' }} />
            <span>Archive Statistics</span>
          </div>
          <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: '0.8rem' }}>
            <span style={{ color: 'var(--text-secondary)' }}>Total Images:</span>
            <span style={{ fontWeight: 600 }}>{totalCount}</span>
          </div>
          <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: '0.8rem' }}>
            <span style={{ color: 'var(--text-secondary)' }}>Original:</span>
            <span style={{ fontWeight: 600 }}>{originalCount}</span>
          </div>
          <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: '0.8rem' }}>
            <span style={{ color: 'var(--text-secondary)' }}>Uploaded:</span>
            <span style={{ fontWeight: 600 }}>{uploadedCount}</span>
          </div>
        </div>
      </div>

      {/* Footer Info */}
      <div>
        <button
          onClick={onLogout}
          className="btn btn-danger"
          style={{
            width: '100%',
            justifyContent: 'center',
            padding: '10px 14px',
            fontSize: '0.85rem',
            marginBottom: '15px'
          }}
        >
          <LogOut size={14} />
          <span>Log Out</span>
        </button>
        <div style={{
          height: '1px',
          background: 'var(--border-color)',
          marginBottom: '15px'
        }} />
        <div style={{
          color: 'var(--text-muted)',
          fontSize: '0.75rem',
          textAlign: 'center',
          lineHeight: '1.6'
        }}>
          <p style={{ fontWeight: '500' }}>Team 17</p>
          <p>University of Bristol · 2026</p>
        </div>
      </div>
    </aside>
  );
};
