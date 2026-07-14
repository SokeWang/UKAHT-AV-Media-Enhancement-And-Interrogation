import { useState, useEffect } from 'react';
import { Sidebar } from './components/Sidebar';
import { ChatAssistant } from './components/ChatAssistant';
import { DetailDrawer } from './components/DetailDrawer';
import { UploadModal } from './components/UploadModal';
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
  // Navigation & UI States
  const [isChatActive, setIsChatActive] = useState(false);
  const [isUploadOpen, setIsUploadOpen] = useState(false);
  const [activeDrawerAsset, setActiveDrawerAsset] = useState<Asset | null>(null);

  // Search & Database Asset States
  const [allAssets, setAllAssets] = useState<Asset[]>([]);
  const [searchResults, setSearchResults] = useState<Asset[]>([]);
  const [searchLoading, setSearchLoading] = useState(false);
  const [searchInput, setSearchInput] = useState('');
  const [activeQuery, setActiveQuery] = useState('');

  // Conversational Assistant States
  const [chatSessionId, setChatSessionId] = useState(generateSessionId());
  const [chatHistory, setChatHistory] = useState<ChatMessage[]>([]);

  // Initial load: Fetch all assets to count stats and showcase recently uploaded
  useEffect(() => {
    loadAllAssets();
  }, []);

  const loadAllAssets = async () => {
    setSearchLoading(true);
    try {
      const response = await fetch(`${API_BASE}/api/assets`);
      if (response.ok) {
        const resJson = await response.json();
        if (resJson.code === 200) {
          setAllAssets(resJson.data);
          // Default search results is all assets
          setSearchResults(resJson.data);
        }
      }
    } catch (err) {
      console.error('Error loading all assets:', err);
    } finally {
      setSearchLoading(false);
    }
  };

  const fetchSearchResults = async (query: string) => {
    setSearchLoading(true);
    try {
      const response = await fetch(`${API_BASE}/api/search`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ query })
      });
      
      if (response.ok) {
        const resJson = await response.json();
        if (resJson.code === 200) {
          setSearchResults(resJson.data);
        }
      }
    } catch (err) {
      console.error('Error searching assets:', err);
    } finally {
      setSearchLoading(false);
    }
  };

  const handleSearchSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    if (!searchInput.trim()) return;

    const query = searchInput.trim();
    setActiveQuery(query);
    setIsChatActive(true);
    fetchSearchResults(query);

    // Reset Chat session when query is submitted from main portal
    resetChatSession();
  };

  const handleQuickImageClick = (asset: Asset) => {
    setActiveDrawerAsset(asset);
  };

  const handleSelectAsset = (assetId: string) => {
    const found = allAssets.find(a => a.id === assetId);
    if (found) {
      setActiveDrawerAsset(found);
    }
  };

  const handleResetSearch = () => {
    setIsChatActive(false);
    setSearchInput('');
    setActiveQuery('');
    setSearchResults(allAssets);
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
    setAllAssets(prev => prev.map(a => a.id === updatedAsset.id ? updatedAsset : a));
    setSearchResults(prev => prev.map(a => a.id === updatedAsset.id ? updatedAsset : a));
    if (activeDrawerAsset && activeDrawerAsset.id === updatedAsset.id) {
      setActiveDrawerAsset(updatedAsset);
    }
  };

  // Get recently uploaded or recently added images (new_addition first, then latest)
  const getRecentAssets = () => {
    const uploaded = allAssets.filter(a => a.data_source === 'new_addition');
    const original = allAssets.filter(a => a.data_source !== 'new_addition');
    // Combine them with uploaded first (reversed to show latest first)
    const combined = [...uploaded.reverse(), ...original.reverse()];
    return combined.slice(0, 6);
  };

  const getImageUrl = (url: string) => {
    if (url.startsWith('http')) return url;
    if (url.startsWith('/')) return `${API_BASE}${url}`;
    return url;
  };

  // Stats computation for sidebar
  const totalCount = allAssets.length;
  const originalCount = allAssets.filter(a => a.data_source !== 'new_addition').length;
  const uploadedCount = allAssets.filter(a => a.data_source === 'new_addition').length;

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
        totalCount={totalCount}
        originalCount={originalCount}
        uploadedCount={uploadedCount}
      />

      {/* Main Content Workspace */}
      <main style={{
        flex: 1,
        height: '100vh',
        overflowY: 'auto',
        padding: '24px 30px',
        display: 'flex',
        flexDirection: 'column',
        position: 'relative'
      }}>
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
              <h3 style={{ fontSize: '1.2rem', color: 'var(--text-primary)', marginBottom: '16px', fontWeight: 600 }}>
                Recently Added Images
              </h3>
              {searchLoading ? (
                <div style={{ display: 'flex', alignItems: 'center', gap: '8px', color: 'var(--text-muted)' }}>
                  <RefreshCw size={16} className="animate-spin" /> Loading archive...
                </div>
              ) : allAssets.length === 0 ? (
                <div style={{ color: 'var(--text-muted)', fontStyle: 'italic', fontSize: '0.9rem' }}>
                  No archive images found. Try uploading some!
                </div>
              ) : (
                <div style={{
                  display: 'grid',
                  gridTemplateColumns: 'repeat(3, 1fr)',
                  gap: '20px'
                }}>
                  {getRecentAssets().map((asset) => (
                    <div
                      key={asset.id}
                      onClick={() => handleQuickImageClick(asset)}
                      className="glass-panel glass-panel-interactive animate-slide-up"
                      style={{
                        overflow: 'hidden',
                        display: 'flex',
                        flexDirection: 'column',
                        height: '220px'
                      }}
                    >
                      <div style={{ height: '140px', backgroundColor: '#050a14', overflow: 'hidden' }}>
                        <img
                          src={getImageUrl(asset.url)}
                          alt={asset.title}
                          style={{ width: '100%', height: '100%', objectFit: 'cover' }}
                        />
                      </div>
                      <div style={{ padding: '12px', flex: 1, display: 'flex', flexDirection: 'column', justifyContent: 'space-between' }}>
                        <div style={{ fontWeight: 600, fontSize: '0.85rem', whiteSpace: 'nowrap', overflow: 'hidden', textOverflow: 'ellipsis' }}>
                          {asset.title}
                        </div>
                        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', fontSize: '0.75rem' }}>
                          <span style={{ color: 'var(--text-secondary)' }}>{asset.category}</span>
                          <span className={`badge ${asset.data_source === 'new_addition' ? 'badge-accent' : 'badge-default'}`} style={{ fontSize: '0.65rem' }}>
                            {asset.data_source === 'new_addition' ? 'New' : 'Archive'}
                          </span>
                        </div>
                      </div>
                    </div>
                  ))}
                </div>
              )}
            </div>
          </div>
        ) : (
          /* =========================================================================
             2. Conversational State: Split layout with retrieval results & AI Chat
             ========================================================================= */
          <div style={{
            display: 'flex',
            gap: '24px',
            height: 'calc(100vh - 48px)',
            overflow: 'hidden',
            animation: 'fadeIn 0.3s ease'
          }}>
            {/* Left Column: Retrieval Grid */}
            <div style={{
              flex: '1.2 1 0',
              display: 'flex',
              flexDirection: 'column',
              overflow: 'hidden'
            }}>
              {/* Query Header / Active filter chip */}
              <div style={{ display: 'flex', alignItems: 'center', gap: '10px', marginBottom: '16px' }}>
                <h3 style={{ fontSize: '1.1rem', fontWeight: 600, color: 'var(--text-primary)' }}>
                  Retrieval Grid
                </h3>
                <div 
                  onClick={handleResetSearch}
                  style={{
                    background: 'rgba(0, 102, 204, 0.08)',
                    border: '1px solid rgba(0, 102, 204, 0.2)',
                    padding: '4px 10px',
                    borderRadius: '50px',
                    fontSize: '0.8rem',
                    color: 'var(--accent-blue)',
                    display: 'flex',
                    alignItems: 'center',
                    gap: '6px',
                    cursor: 'pointer'
                  }}
                  title="Clear active query and go back home"
                >
                  <span>Query: <strong>"{activeQuery}"</strong></span>
                  <span style={{ fontWeight: 800 }}>×</span>
                </div>
              </div>

              {/* Grid Scroll Area */}
              <div style={{ flex: 1, overflowY: 'auto', paddingRight: '8px' }}>
                {searchLoading ? (
                  <div style={{ display: 'flex', alignItems: 'center', gap: '8px', color: 'var(--text-secondary)' }}>
                    <RefreshCw size={16} className="animate-spin" /> Searching archive...
                  </div>
                ) : searchResults.length === 0 ? (
                  <div className="glass-panel" style={{ padding: '40px 20px', textAlign: 'center', color: 'var(--text-muted)' }}>
                    No images match the query directly.
                  </div>
                ) : (
                  <div style={{
                    display: 'grid',
                    gridTemplateColumns: 'repeat(2, 1fr)',
                    gap: '16px'
                  }}>
                    {searchResults.map((asset) => (
                      <div
                        key={asset.id}
                        onClick={() => handleQuickImageClick(asset)}
                        className="glass-panel glass-panel-interactive animate-slide-up"
                        style={{
                          overflow: 'hidden',
                          display: 'flex',
                          flexDirection: 'column',
                          height: '200px'
                        }}
                      >
                        <div style={{ height: '120px', backgroundColor: '#050a14', overflow: 'hidden', position: 'relative' }}>
                          <img
                            src={getImageUrl(asset.url)}
                            alt={asset.title}
                            style={{ width: '100%', height: '100%', objectFit: 'cover' }}
                          />
                          {asset.score !== undefined && (
                            <div style={{
                              position: 'absolute',
                              bottom: '6px',
                              right: '6px',
                              background: 'rgba(5, 10, 20, 0.8)',
                              padding: '2px 6px',
                              borderRadius: '4px',
                              fontSize: '0.7rem',
                              color: 'var(--accent-cyan)'
                            }}>
                              Score: {asset.score.toFixed(3)}
                            </div>
                          )}
                        </div>
                        <div style={{ padding: '10px', flex: 1, display: 'flex', flexDirection: 'column', justifyContent: 'space-between' }}>
                          <div style={{ fontWeight: 600, fontSize: '0.8rem', whiteSpace: 'nowrap', overflow: 'hidden', textOverflow: 'ellipsis' }}>
                            {asset.title}
                          </div>
                          <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', fontSize: '0.75rem' }}>
                            <span style={{ color: 'var(--text-secondary)' }}>{asset.category}</span>
                            <span className={`badge ${asset.data_source === 'new_addition' ? 'badge-accent' : 'badge-default'}`} style={{ fontSize: '0.65rem' }}>
                              {asset.data_source === 'new_addition' ? 'New' : 'Archive'}
                            </span>
                          </div>
                        </div>
                      </div>
                    ))}
                  </div>
                )}
              </div>
            </div>

            {/* Right Column: Chat Assistant */}
            <div style={{
              flex: '0.9 1 0',
              height: '100%'
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
          onUploadSuccess={loadAllAssets}
          apiBase={API_BASE}
        />
      </main>
    </div>
  );
}

export default App;
