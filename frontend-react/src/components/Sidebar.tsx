import React from 'react';
import { Search, Edit } from 'lucide-react';

interface SidebarProps {
  activeTab: 'search' | 'annotate';
  setActiveTab: (tab: 'search' | 'annotate') => void;
}

export const Sidebar: React.FC<SidebarProps> = ({ activeTab, setActiveTab }) => {
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
          marginBottom: '30px'
        }} />

        {/* Navigation Menu */}
        <nav style={{ display: 'flex', flexDirection: 'column', gap: '8px' }}>
          <button
            onClick={() => setActiveTab('search')}
            className={`btn ${activeTab === 'search' ? 'btn-primary' : 'btn-secondary'}`}
            style={{
              justifyContent: 'flex-start',
              width: '100%',
              padding: '14px 18px',
              border: activeTab === 'search' ? 'none' : '1px solid transparent',
              background: activeTab === 'search' ? undefined : 'transparent',
              color: activeTab === 'search' ? '#070d19' : 'var(--text-primary)'
            }}
          >
            <Search size={18} />
            <span>Smart Search & Chat</span>
          </button>

          <button
            onClick={() => setActiveTab('annotate')}
            className={`btn ${activeTab === 'annotate' ? 'btn-primary' : 'btn-secondary'}`}
            style={{
              justifyContent: 'flex-start',
              width: '100%',
              padding: '14px 18px',
              border: activeTab === 'annotate' ? 'none' : '1px solid transparent',
              background: activeTab === 'annotate' ? undefined : 'transparent',
              color: activeTab === 'annotate' ? '#070d19' : 'var(--text-primary)'
            }}
          >
            <Edit size={18} />
            <span>Caption Annotator</span>
          </button>
        </nav>
      </div>

      {/* Footer Info */}
      <div>
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
