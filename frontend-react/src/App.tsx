import { useState, useEffect } from 'react';
import { Sidebar } from './components/Sidebar';
import { ChatAssistant } from './components/ChatAssistant';
import { DetailDrawer } from './components/DetailDrawer';
import { UploadModal } from './components/UploadModal';
import { Login } from './components/Login';
import Dashboard from './components/Dashboard';
import { Search, Sparkles, RefreshCw } from 'lucide-react';

interface Asset {
  id: string;
  url: string;
  title: string;
  category: string;
  description: string;
  base_code?: string;
  subject_type?: string;
  shooting_year?: string;
  copyright?: string;
  data_source?: string;
  score?: number;
  stacked_assets?: Asset[];
}

interface ChatMessage {
  role: 'user' | 'assistant';
  content: string;
  assets?: any[];
}

const API_BASE = import.meta.env.VITE_API_BASE !== undefined 
  ? import.meta.env.VITE_API_BASE 
  : (typeof window !== 'undefined' && (window.location.hostname === 'localhost' || window.location.hostname === '127.0.0.1') ? 'http://localhost:8000' : '');

const generateSessionId = () => `sess_${Math.random().toString(36).substring(2, 10)}`;

function App() {
  // Authentication State
  const [isLoggedIn, setIsLoggedIn] = useState(() => {
    return localStorage.getItem('ukaht_auth') === 'true' || sessionStorage.getItem('ukaht_auth') === 'true';
  });

  // Navigation & UI States
  const [isChatActive, setIsChatActive] = useState(false);
  const [isUploadOpen, setIsUploadOpen] = useState(false);
  const [isDashboardOpen, setIsDashboardOpen] = useState(false);
  const [activeDrawerAsset, setActiveDrawerAsset] = useState<Asset | null>(null);

  // Synchronization States
  const [syncing, setSyncing] = useState(false);
  const [syncStatusText, setSyncStatusText] = useState('Sync S3 Database');

  // Search & Database Asset States
  const [stats, setStats] = useState({ totalCount: 0, originalCount: 0, uploadedCount: 0 });
  const [recentAssets, setRecentAssets] = useState<Asset[]>([]);
  const [recentLoading, setRecentLoading] = useState(false);
  const [hasMoreRecent, setHasMoreRecent] = useState(true);
  const [nextOffset, setNextOffset] = useState(0);
  const [searchInput, setSearchInput] = useState('');
  const [activeQuery, setActiveQuery] = useState('');
  const [groupBy, setGroupBy] = useState<'none' | 'folder' | 'year' | 'base' | 'category'>('none');
  const [expandedRecentStacks, setExpandedRecentStacks] = useState<{ [key: string]: boolean }>({});

  const renderAssetCard = (asset: Asset) => {
    const isStacked = asset.stacked_assets && asset.stacked_assets.length > 0;
    const isExpanded = expandedRecentStacks[asset.id];
    
    return (
      <div key={asset.id} style={{ position: 'relative' }}>
        {/* Background Stack Effect Layers */}
        {isStacked && (
          <>
            <div style={{
              position: 'absolute',
              top: '6px',
              left: '6px',
              right: '-6px',
              bottom: '-6px',
              background: 'rgba(5, 10, 20, 0.4)',
              border: '1px solid rgba(255, 255, 255, 0.05)',
              borderRadius: '12px',
              zIndex: 1
            }} />
            <div style={{
              position: 'absolute',
              top: '3px',
              left: '3px',
              right: '-3px',
              bottom: '-3px',
              background: 'rgba(10, 18, 36, 0.6)',
              border: '1px solid rgba(255, 255, 255, 0.08)',
              borderRadius: '12px',
              zIndex: 2
            }} />
          </>
        )}

        {/* Main Card */}
        <div
          onClick={() => handleQuickImageClick(asset)}
          className="glass-panel glass-panel-interactive animate-slide-up"
          style={{
            zIndex: 3,
            position: 'relative',
            overflow: 'hidden',
            display: 'flex',
            flexDirection: 'column',
            height: isExpanded ? 'auto' : '220px',
            transition: 'all 0.3s ease'
          }}
        >
          <div style={{ height: '140px', backgroundColor: '#050a14', overflow: 'hidden', position: 'relative' }}>
            <img
              src={getImageUrl(asset.url)}
              alt={asset.title}
              style={{ width: '100%', height: '100%', objectFit: 'cover' }}
            />
            {isStacked && (
              <div style={{
                position: 'absolute',
                top: '8px',
                left: '8px',
                background: 'rgba(0, 229, 255, 0.95)',
                color: '#050a14',
                padding: '2px 6px',
                borderRadius: '4px',
                fontSize: '0.65rem',
                fontWeight: 700,
                boxShadow: '0 2px 5px rgba(0, 229, 255, 0.3)'
              }}>
                +{asset.stacked_assets!.length + 1} Stacked
              </div>
            )}
          </div>
          <div style={{ padding: '12px', flex: 1, display: 'flex', flexDirection: 'column', justifyContent: 'space-between', gap: '8px' }}>
            <div>
              <div style={{ fontWeight: 600, fontSize: '0.85rem', whiteSpace: 'nowrap', overflow: 'hidden', textOverflow: 'ellipsis' }}>
                {asset.title}
              </div>
              <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', fontSize: '0.75rem', marginTop: '4px' }}>
                <span style={{ color: 'var(--text-secondary)' }}>{asset.category}</span>
                <span className={`badge ${asset.data_source === 'new_addition' ? 'badge-accent' : 'badge-default'}`} style={{ fontSize: '0.65rem' }}>
                  {asset.data_source === 'new_addition' ? 'New' : 'Archive'}
                </span>
              </div>
            </div>

            {/* Stacking controls inside the card */}
            {isStacked && (
              <button
                onClick={(e) => {
                  e.stopPropagation();
                  setExpandedRecentStacks(prev => ({ ...prev, [asset.id]: !prev[asset.id] }));
                }}
                className="btn btn-secondary"
                style={{
                  width: '100%',
                  fontSize: '0.75rem',
                  padding: '4px 8px',
                  borderColor: 'var(--accent-cyan)',
                  color: 'var(--accent-cyan)',
                  marginTop: '4px'
                }}
              >
                {isExpanded ? 'Hide Similar ↑' : `Show ${asset.stacked_assets!.length} Similar ↓`}
              </button>
            )}

            {/* Stacked sub-assets list */}
            {isExpanded && asset.stacked_assets && (
              <div style={{
                display: 'flex',
                flexDirection: 'column',
                gap: '8px',
                marginTop: '8px',
                paddingTop: '8px',
                borderTop: '1px solid var(--border-color)'
              }}>
                {asset.stacked_assets.map((subAsset) => (
                  <div
                    key={subAsset.id}
                    onClick={(e) => {
                      e.stopPropagation();
                      handleQuickImageClick(subAsset);
                    }}
                    style={{
                      display: 'flex',
                      alignItems: 'center',
                      gap: '8px',
                      background: 'rgba(5, 10, 20, 0.4)',
                      padding: '4px 8px',
                      borderRadius: '6px',
                      cursor: 'pointer',
                      border: '1px solid transparent'
                    }}
                    onMouseEnter={(e) => e.currentTarget.style.borderColor = 'var(--accent-cyan)'}
                    onMouseLeave={(e) => e.currentTarget.style.borderColor = 'transparent'}
                  >
                    <img
                      src={getImageUrl(subAsset.url)}
                      alt={subAsset.title}
                      style={{ width: '36px', height: '36px', objectFit: 'cover', borderRadius: '4px' }}
                    />
                    <div style={{
                      fontSize: '0.75rem',
                      fontWeight: 600,
                      color: 'var(--text-primary)',
                      whiteSpace: 'nowrap',
                      overflow: 'hidden',
                      textOverflow: 'ellipsis',
                      flex: 1
                    }}>
                      {subAsset.title}
                    </div>
                  </div>
                ))}
              </div>
            )}
          </div>
        </div>
      </div>
    );
  };

  const getGroupedAssets = () => {
    if (groupBy === 'none') return null;

    const groups: { [key: string]: Asset[] } = {};
    recentAssets.forEach((asset) => {
      let key = 'Other';
      if (groupBy === 'folder') {
        if (asset.url) {
          try {
            const decoded = decodeURIComponent(asset.url);
            const pathParts = decoded.split('?')[0].split('/');
            if (pathParts.length > 1) {
              const parent = pathParts[pathParts.length - 2];
              if (parent && !parent.includes('.com') && !parent.startsWith('http') && parent !== 'uploads' && parent !== 'assets') {
                key = parent.replace(/_/g, ' ').replace(/-/g, ' ');
              } else {
                key = 'Root / General Uploads';
              }
            }
          } catch (e) {
            key = 'Root / General Uploads';
          }
        }
      } else if (groupBy === 'year') {
        key = asset.shooting_year ? `Year: ${asset.shooting_year}` : 'Unknown Year';
      } else if (groupBy === 'base') {
        key = asset.base_code ? `Base ${asset.base_code.toUpperCase()}` : 'Unknown Base';
      } else if (groupBy === 'category') {
        key = asset.category || 'Uncategorized';
      }

      if (!groups[key]) {
        groups[key] = [];
      }
      groups[key].push(asset);
    });
    return groups;
  };

  // Conversational Assistant States
  const [chatSessionId, setChatSessionId] = useState(generateSessionId());
  const [chatHistory, setChatHistory] = useState<ChatMessage[]>([]);

  // Initial load: Fetch stats and showcase recently added images
  useEffect(() => {
    loadStats();
    loadRecentAssets(0, true);
  }, []);

  const loadStats = async () => {
    try {
      const response = await fetch(`${API_BASE}/api/assets/stats`);
      if (response.ok) {
        const resJson = await response.json();
        if (resJson.code === 200) {
          setStats(resJson.data);
        }
      }
    } catch (err) {
      console.error('Error loading stats:', err);
    }
  };

  const loadRecentAssets = async (offset = 0, isInitial = false) => {
    if (recentLoading) return;
    setRecentLoading(true);
    try {
      const response = await fetch(`${API_BASE}/api/assets/recent?offset=${offset}&limit=6`);
      if (response.ok) {
        const resJson = await response.json();
        if (resJson.code === 200) {
          const fetched = resJson.data || [];
          if (isInitial) {
            setRecentAssets(fetched);
          } else {
            setRecentAssets(prev => [...prev, ...fetched]);
          }
          setNextOffset(resJson.next_offset);
          setHasMoreRecent(resJson.has_more);
        }
      }
    } catch (err) {
      console.error('Error loading recent assets:', err);
    } finally {
      setRecentLoading(false);
    }
  };

  const loadMoreRecentAssets = () => {
    if (!recentLoading && hasMoreRecent) {
      loadRecentAssets(nextOffset, false);
    }
  };

  const handleScroll = (e: React.UIEvent<HTMLDivElement>) => {
    const target = e.currentTarget;
    const threshold = 100; // pixels from the bottom
    const isNearBottom = target.scrollHeight - target.scrollTop - target.clientHeight < threshold;
    if (isNearBottom && !recentLoading && hasMoreRecent) {
      loadMoreRecentAssets();
    }
  };

  const handleUploadSuccess = () => {
    setIsUploadOpen(false);
    loadStats();
    loadRecentAssets(0, true);
  };

  const handleSyncDatabase = async () => {
    if (syncing) return;
    setSyncing(true);
    setSyncStatusText('Syncing S3...');
    try {
      const response = await fetch(`${API_BASE}/api/assets/sync`, { method: 'POST' });
      const result = await response.json();
      if (result.code === 200) {
        pollSyncStatus();
      } else {
        alert(result.detail || result.message || 'Sync failed to start');
        setSyncing(false);
        setSyncStatusText('Sync S3 Database');
      }
    } catch (err) {
      console.error(err);
      alert('Error connecting to backend for database sync');
      setSyncing(false);
      setSyncStatusText('Sync S3 Database');
    }
  };

  const pollSyncStatus = () => {
    const interval = setInterval(async () => {
      try {
        const response = await fetch(`${API_BASE}/api/assets/sync/status`);
        const result = await response.json();
        if (result.code === 200) {
          const status = result.data.status;
          const processed = result.data.processed;
          const total = result.data.total;
          const msg = result.data.message;

          if (status === 'scanning') {
            setSyncStatusText('Scanning S3...');
          } else if (status === 'syncing') {
            setSyncStatusText(`Syncing (${processed}/${total})`);
          } else if (status === 'success') {
            setSyncStatusText('Sync Completed!');
            clearInterval(interval);
            setTimeout(() => {
              setSyncing(false);
              setSyncStatusText('Sync S3 Database');
              loadStats();
              loadRecentAssets(0, true);
            }, 3000);
          } else if (status === 'error') {
            alert(`Sync Error: ${msg}`);
            setSyncing(false);
            setSyncStatusText('Sync S3 Database');
            clearInterval(interval);
          }
        }
      } catch (err) {
        console.error(err);
        clearInterval(interval);
        setSyncing(false);
        setSyncStatusText('Sync S3 Database');
      }
    }, 2000);
  };

  const handleSearchSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    if (!searchInput.trim()) return;

    const query = searchInput.trim();
    setActiveQuery(query);
    setIsChatActive(true);

    // Reset Chat session when query is submitted from main portal
    resetChatSession();
  };

  const handleQuickImageClick = (asset: Asset) => {
    setActiveDrawerAsset(asset);
  };

  const handleSelectAsset = async (assetId: string) => {
    const found = recentAssets.find(a => a.id === assetId);
    if (found) {
      setActiveDrawerAsset(found);
      return;
    }

    try {
      const response = await fetch(`${API_BASE}/api/assets/${assetId}`);
      if (response.ok) {
        const resJson = await response.json();
        if (resJson.code === 200) {
          setActiveDrawerAsset(resJson.data);
        }
      }
    } catch (err) {
      console.error('Error fetching asset details:', err);
    }
  };

  const handleResetSearch = () => {
    setIsChatActive(false);
    setSearchInput('');
    setActiveQuery('');
    resetChatSession();
  };

  const resetChatSession = async () => {
    try {
      await fetch(`${API_BASE}/api/agent/session/${chatSessionId}`, {
        method: 'DELETE'
      });
    } catch (err) {
      console.error('Failed to reset backend session:', err);
    }
    setChatSessionId(generateSessionId());
    setChatHistory([]);
  };

  const handleAssetUpdate = (updatedAsset: Asset) => {
    setRecentAssets(prev => prev.map(a => a.id === updatedAsset.id ? updatedAsset : a));
    if (activeDrawerAsset && activeDrawerAsset.id === updatedAsset.id) {
      setActiveDrawerAsset(updatedAsset);
    }
  };

  const getImageUrl = (url: string) => {
    if (url.startsWith('http')) return url;
    if (url.startsWith('/')) return `${API_BASE}${url}`;
    return url;
  };

  // Stats computation for sidebar
  const totalCount = stats.totalCount;
  const originalCount = stats.originalCount;
  const uploadedCount = stats.uploadedCount;

  const handleLogout = () => {
    localStorage.removeItem('ukaht_auth');
    sessionStorage.removeItem('ukaht_auth');
    setIsLoggedIn(false);
  };

  if (!isLoggedIn) {
    return <Login onLoginSuccess={() => setIsLoggedIn(true)} />;
  }

  return (
    <div style={{
      display: 'flex',
      minHeight: '100vh',
      width: '100vw',
      backgroundColor: 'var(--bg-primary)',
      overflow: 'hidden'
    }}>
      {/* Sidebar Navigation */}
      <Sidebar 
        onResetSearch={handleResetSearch}
        onOpenUpload={() => setIsUploadOpen(true)}
        onOpenDashboard={() => setIsDashboardOpen(true)}
        totalCount={totalCount}
        originalCount={originalCount}
        uploadedCount={uploadedCount}
        onLogout={handleLogout}
        onSyncDatabase={handleSyncDatabase}
        syncing={syncing}
        syncStatusText={syncStatusText}
      />

      {/* Main Content Workspace */}
      <main 
        onScroll={handleScroll}
        style={{
          flex: 1,
          height: '100vh',
          overflowY: 'auto',
          padding: '24px 30px',
          display: 'flex',
          flexDirection: 'column',
          position: 'relative'
        }}
      >
        {!isChatActive ? (
          /* =========================================================================
             1. Landing Page State: Big central search + Recently Added Gallery
             ========================================================================= */
          <div style={{
            maxWidth: '900px',
            width: '100%',
            margin: '8% auto 0 auto',
            display: 'flex',
            flexDirection: 'column',
            gap: '40px',
            animation: 'fadeIn 0.4s ease'
          }}>
            {/* Landing Branding Header */}
            <div style={{ textAlign: 'center' }}>
              <h1 className="glow-text" style={{ 
                fontSize: '2.8rem', 
                fontWeight: 800, 
                marginBottom: '12px',
                fontFamily: 'var(--font-family-title)',
                letterSpacing: '-0.02em'
              }}>
                Explore UKAHT Antarctic Archive
              </h1>
              <p style={{ color: 'var(--text-secondary)', fontSize: '1.1rem', maxWidth: '600px', margin: '0 auto' }}>
                Ask historical questions or search expedition photos via conversational AI
              </p>
            </div>

            {/* Giant Central Search Bar */}
            <form onSubmit={handleSearchSubmit} className="glass-panel pulse-glow" style={{
              display: 'flex',
              alignItems: 'center',
              padding: '8px 16px',
              borderRadius: '99px',
              border: '1px solid rgba(0, 102, 204, 0.25)',
              background: 'rgba(255, 255, 255, 0.9)',
              gap: '12px'
            }}>
              <Search size={22} style={{ color: 'var(--text-muted)', flexShrink: 0 }} />
              <input
                type="text"
                value={searchInput}
                onChange={(e) => setSearchInput(e.target.value)}
                placeholder="Ask e.g. 'How did sledging dogs help?' or search 'horseshoe island'..."
                style={{
                  border: 'none',
                  boxShadow: 'none',
                  padding: '12px 4px',
                  fontSize: '1.1rem',
                  background: 'transparent'
                }}
              />
              <button 
                type="submit" 
                className="btn btn-primary"
                style={{
                  borderRadius: '50px',
                  padding: '10px 24px',
                  display: 'flex',
                  alignItems: 'center',
                  gap: '6px',
                  whiteSpace: 'nowrap'
                }}
              >
                <Sparkles size={16} />
                <span>Ask AI</span>
              </button>
            </form>

            {/* Showcase Gallery: Recently Uploaded Images */}
            <div style={{ marginTop: '20px' }}>
              <div style={{
                display: 'flex',
                justifyContent: 'space-between',
                alignItems: 'center',
                marginBottom: '16px',
                flexWrap: 'wrap',
                gap: '12px'
              }}>
                <h3 style={{ fontSize: '1.2rem', color: 'var(--text-primary)', margin: 0, fontWeight: 600 }}>
                  Recently Added Images
                </h3>
                <div style={{
                  display: 'flex',
                  alignItems: 'center',
                  gap: '8px',
                  background: 'rgba(255, 255, 255, 0.05)',
                  padding: '4px',
                  borderRadius: '8px',
                  border: '1px solid var(--border-color)',
                  fontSize: '0.8rem'
                }}>
                  <span style={{ color: 'var(--text-muted)', paddingLeft: '8px' }}>Group by:</span>
                  {[
                    { id: 'none', label: 'None' },
                    { id: 'folder', label: 'Folder' },
                    { id: 'year', label: 'Year' },
                    { id: 'base', label: 'Base' },
                    { id: 'category', label: 'Category' }
                  ].map(opt => (
                    <button
                      key={opt.id}
                      onClick={() => setGroupBy(opt.id as any)}
                      style={{
                        background: groupBy === opt.id ? 'var(--accent-blue)' : 'transparent',
                        color: groupBy === opt.id ? '#ffffff' : 'var(--text-secondary)',
                        border: 'none',
                        borderRadius: '6px',
                        padding: '4px 8px',
                        fontSize: '0.75rem',
                        cursor: 'pointer',
                        fontWeight: groupBy === opt.id ? 600 : 400,
                        transition: 'all 0.2s'
                      }}
                    >
                      {opt.label}
                    </button>
                  ))}
                </div>
              </div>

              {recentLoading && recentAssets.length === 0 ? (
                <div style={{ display: 'flex', alignItems: 'center', gap: '8px', color: 'var(--text-muted)' }}>
                  <RefreshCw size={16} style={{ animation: 'spin 1.5s linear infinite' }} /> Loading archive...
                </div>
              ) : recentAssets.length === 0 && !recentLoading ? (
                <div style={{ color: 'var(--text-muted)', fontStyle: 'italic', fontSize: '0.9rem' }}>
                  No archive images found. Try uploading some!
                </div>
              ) : (
                <div style={{ display: 'flex', flexDirection: 'column', gap: '24px' }}>
                  {groupBy === 'none' ? (
                    <div style={{
                      display: 'grid',
                      gridTemplateColumns: 'repeat(3, 1fr)',
                      gap: '20px'
                    }}>
                      {recentAssets.map((asset) => renderAssetCard(asset))}
                    </div>
                  ) : (
                    Object.entries(getGroupedAssets() || {}).map(([groupName, assets]) => (
                      <div key={groupName} style={{ display: 'flex', flexDirection: 'column', gap: '12px' }}>
                        <div style={{
                          display: 'flex',
                          alignItems: 'center',
                          gap: '10px',
                          marginTop: '8px',
                          borderBottom: '1px solid var(--border-color)',
                          paddingBottom: '8px'
                        }}>
                          <span style={{
                            width: '4px',
                            height: '16px',
                            background: 'var(--accent-cyan)',
                            borderRadius: '2px'
                          }}></span>
                          <span style={{ fontWeight: 600, fontSize: '0.95rem', color: 'var(--text-primary)' }}>
                            {groupName}
                          </span>
                          <span style={{
                            fontSize: '0.75rem',
                            color: 'var(--text-muted)',
                            background: 'rgba(255,255,255,0.06)',
                            padding: '2px 6px',
                            borderRadius: '4px'
                          }}>
                            {assets.length} items
                          </span>
                        </div>
                        <div style={{
                          display: 'grid',
                          gridTemplateColumns: 'repeat(3, 1fr)',
                          gap: '20px'
                        }}>
                          {assets.map((asset) => renderAssetCard(asset))}
                        </div>
                      </div>
                    ))
                  )}
                  {recentLoading && (
                    <div style={{ display: 'flex', justifyContent: 'center', padding: '20px' }}>
                      <RefreshCw size={24} style={{ color: 'var(--accent-cyan)', animation: 'spin 1.5s linear infinite' }} />
                    </div>
                  )}
                  {!hasMoreRecent && recentAssets.length > 0 && (
                    <div style={{ textAlign: 'center', color: 'var(--text-muted)', fontSize: '0.8rem', padding: '10px' }}>
                      All assets loaded
                    </div>
                  )}
                </div>
              )}
            </div>
          </div>
        ) : (
          /* =========================================================================
             2. Conversational State: Unified AI Chat Portal (No Split View)
             ========================================================================= */
          <div style={{
            maxWidth: '900px',
            width: '100%',
            height: 'calc(100vh - 48px)',
            margin: '0 auto',
            overflow: 'hidden',
            animation: 'fadeIn 0.3s ease'
          }}>
            <ChatAssistant
              apiBase={API_BASE}
              activeQuery={activeQuery}
              sessionId={chatSessionId}
              onResetSession={() => resetChatSession()}
              chatHistory={chatHistory}
              setChatHistory={setChatHistory}
              onSelectAsset={handleSelectAsset}
            />
          </div>
        )}

        {/* Slide-out detail drawer */}
        {activeDrawerAsset && (
          <DetailDrawer 
            asset={activeDrawerAsset}
            onClose={() => setActiveDrawerAsset(null)}
            apiBase={API_BASE}
            onAssetUpdate={handleAssetUpdate}
            onSelectAsset={handleSelectAsset}
          />
        )}

        {/* Upload Modal */}
        <UploadModal 
          isOpen={isUploadOpen}
          onClose={() => setIsUploadOpen(false)}
          onUploadSuccess={handleUploadSuccess}
          apiBase={API_BASE}
        />

        {/* Deep Learning Evaluation Dashboard */}
        {isDashboardOpen && (
          <Dashboard 
            apiBase={API_BASE}
            onSelectAsset={handleSelectAsset}
            onClose={() => setIsDashboardOpen(false)}
          />
        )}
      </main>
    </div>
  );
}

export default App;
