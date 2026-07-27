import { useState, useEffect } from 'react';
import { X, CheckCircle, RefreshCw, Image as ImageIcon, MapPin, Calendar, Copyright, Globe, Bookmark, ArrowLeft, Maximize2, Download } from 'lucide-react';

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
}

interface DetailDrawerProps {
  asset: Asset;
  onClose: () => void;
  apiBase: string;
  onAssetUpdate?: (updatedAsset: Asset) => void;
  onSelectAsset?: (assetId: string) => void;
}

export function DetailDrawer({ asset, onClose, apiBase, onAssetUpdate, onSelectAsset }: DetailDrawerProps) {
  const [caption, setCaption] = useState(asset.description);
  const [saving, setSaving] = useState(false);
  const [goldenSaving, setGoldenSaving] = useState(false);
  const [saveMessage, setSaveMessage] = useState<string | null>(null);
  const [goldenMessage, setGoldenMessage] = useState<string | null>(null);
  const [isFullscreen, setIsFullscreen] = useState(false);
  
  // Similar Recommendations State
  const [similarAssets, setSimilarAssets] = useState<Asset[]>([]);
  const [loadingSimilar, setLoadingSimilar] = useState(false);

  // Sync state when asset changes
  useEffect(() => {
    setCaption(asset.description);
    setSaveMessage(null);
    setGoldenMessage(null);
    fetchSimilar(asset.id);
  }, [asset]);

  // Handle ESC key to close
  useEffect(() => {
    const handleKeyDown = (e: KeyboardEvent) => {
      if (e.key === 'Escape') {
        if (isFullscreen) {
          setIsFullscreen(false);
        } else {
          onClose();
        }
      }
    };
    window.addEventListener('keydown', handleKeyDown);
    return () => window.removeEventListener('keydown', handleKeyDown);
  }, [isFullscreen, onClose]);

  const fetchSimilar = async (id: string) => {
    setLoadingSimilar(true);
    try {
      const response = await fetch(`${apiBase}/api/recommend`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ id, limit: 6 })
      });
      if (response.ok) {
        const resJson = await response.json();
        if (resJson.code === 200) {
          setSimilarAssets(resJson.data.filter((item: Asset) => item.id !== id));
        }
      }
    } catch (err) {
      console.error('Error fetching similar recommendations:', err);
    } finally {
      setLoadingSimilar(false);
    }
  };

  const handleSaveCaption = async () => {
    setSaving(true);
    setSaveMessage(null);
    try {
      const response = await fetch(`${apiBase}/api/assets/${asset.id}/caption`, {
        method: 'PUT',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ caption })
      });
      if (response.ok) {
        const resJson = await response.json();
        if (resJson.code === 200) {
          setSaveMessage('Caption updated successfully!');
          if (onAssetUpdate) {
            onAssetUpdate({ ...asset, description: caption });
          }
        } else {
          setSaveMessage(`Error: ${resJson.message}`);
        }
      } else {
        setSaveMessage('Failed to update caption. Server error.');
      }
    } catch (err) {
      setSaveMessage('Failed to update caption. Network error.');
    } finally {
      setSaving(false);
    }
  };

  const handleVerifyGolden = async () => {
    setGoldenSaving(true);
    setGoldenMessage(null);
    try {
      const response = await fetch(`${apiBase}/api/golden`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ asset_id: asset.id, caption })
      });
      if (response.ok) {
        const resJson = await response.json();
        if (resJson.code === 200) {
          setGoldenMessage('Saved to Golden Test Set!');
        } else {
          setGoldenMessage(`Error: ${resJson.message}`);
        }
      } else {
        setGoldenMessage('Failed to verify. Server error.');
      }
    } catch (err) {
      setGoldenMessage('Failed to verify. Network error.');
    } finally {
      setGoldenSaving(false);
    }
  };

  // Construct server-proxied URL to guarantee 100% successful image loading
  const getImageUrl = (url?: string) => {
    if (!url) return '';
    if (url.startsWith('/api/image-proxy')) {
      return `${apiBase}${url}`;
    }
    if (url.startsWith('http://') || url.startsWith('https://')) {
      return `${apiBase}/api/image-proxy?url=${encodeURIComponent(url)}`;
    }
    if (url.startsWith('/')) {
      return `${apiBase}${url}`;
    }
    return `${apiBase}/api/image-proxy?url=${encodeURIComponent(url)}`;
  };

  const currentDisplayImageUrl = getImageUrl(asset.url);

  return (
    <>
      {/* Fullscreen Lightbox Modal */}
      {isFullscreen && (
        <div 
          onClick={() => setIsFullscreen(false)}
          style={{
            position: 'fixed',
            top: 0,
            left: 0,
            right: 0,
            bottom: 0,
            backgroundColor: 'rgba(5, 10, 20, 0.95)',
            backdropFilter: 'blur(8px)',
            zIndex: 3000,
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'center',
            padding: '24px',
            cursor: 'zoom-out'
          }}
        >
          <button 
            onClick={() => setIsFullscreen(false)}
            style={{
              position: 'absolute',
              top: '20px',
              right: '20px',
              background: 'rgba(255, 255, 255, 0.1)',
              border: '1px solid rgba(255, 255, 255, 0.2)',
              color: '#ffffff',
              borderRadius: '50%',
              width: '44px',
              height: '44px',
              display: 'flex',
              alignItems: 'center',
              justifyContent: 'center',
              cursor: 'pointer'
            }}
          >
            <X size={24} />
          </button>
          <img 
            src={currentDisplayImageUrl} 
            alt={asset.title} 
            style={{
              maxWidth: '95vw',
              maxHeight: '92vh',
              objectFit: 'contain',
              borderRadius: '8px',
              boxShadow: '0 20px 50px rgba(0,0,0,0.5)'
            }}
          />
        </div>
      )}

      {/* Secondary Details Page Overlay Container (二级页面) */}
      <div style={{
        position: 'fixed',
        top: 0,
        left: 0,
        right: 0,
        bottom: 0,
        backgroundColor: 'rgba(10, 18, 36, 0.75)',
        backdropFilter: 'blur(10px)',
        zIndex: 2000,
        display: 'flex',
        alignItems: 'center',
        justifyContent: 'center',
        padding: '20px',
        animation: 'fadeIn 0.25s ease-out forwards'
      }}>
        {/* Main Secondary Page Container (二级页面卡片) */}
        <div style={{
          width: '1080px',
          maxWidth: '95vw',
          height: '90vh',
          maxHeight: '900px',
          backgroundColor: '#ffffff',
          borderRadius: '16px',
          boxShadow: '0 24px 60px rgba(0, 102, 204, 0.25)',
          border: '1px solid var(--border-color)',
          display: 'flex',
          flexDirection: 'column',
          overflow: 'hidden',
          animation: 'slideUp 0.3s cubic-bezier(0.16, 1, 0.3, 1) forwards'
        }}>
          {/* Top Secondary Page Navigation Header */}
          <div style={{
            padding: '16px 24px',
            borderBottom: '1px solid var(--border-color)',
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'space-between',
            background: 'linear-gradient(135deg, rgba(0, 102, 204, 0.03), rgba(0, 102, 204, 0.08))'
          }}>
            {/* Breadcrumb / Back Button */}
            <button
              onClick={onClose}
              className="btn btn-secondary"
              style={{
                display: 'flex',
                alignItems: 'center',
                gap: '8px',
                padding: '8px 14px',
                fontSize: '0.85rem',
                fontWeight: 600
              }}
            >
              <ArrowLeft size={16} />
              <span>Back to Archive Search</span>
            </button>

            {/* Title & Category Badge */}
            <div style={{ textAlign: 'center', display: 'flex', alignItems: 'center', gap: '10px' }}>
              <span className="badge badge-accent" style={{ fontSize: '0.75rem', padding: '4px 10px' }}>
                {asset.category || 'Archive Asset'}
              </span>
              <h2 style={{ fontSize: '1.25rem', fontWeight: 700, color: 'var(--text-primary)', margin: 0 }}>
                {asset.title || 'Asset Details'}
              </h2>
              <span style={{ fontSize: '0.8rem', color: 'var(--text-muted)' }}>({asset.id})</span>
            </div>

            {/* Action Buttons */}
            <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
              <button 
                onClick={onClose}
                style={{
                  background: 'rgba(0, 0, 0, 0.05)',
                  border: '1px solid var(--border-color)',
                  color: 'var(--text-secondary)',
                  cursor: 'pointer',
                  padding: '8px',
                  borderRadius: '50%',
                  display: 'flex',
                  alignItems: 'center',
                  justifyContent: 'center',
                  transition: 'all 0.2s ease'
                }}
                onMouseEnter={(e) => e.currentTarget.style.backgroundColor = 'rgba(239, 68, 68, 0.1)'}
                onMouseLeave={(e) => e.currentTarget.style.backgroundColor = 'rgba(0, 0, 0, 0.05)'}
              >
                <X size={20} />
              </button>
            </div>
          </div>

          {/* Secondary Page 2-Column Content Body */}
          <div style={{
            flex: 1,
            display: 'grid',
            gridTemplateColumns: '55% 45%',
            overflow: 'hidden'
          }}>
            {/* Left Column: Image Hero View & View Tools */}
            <div style={{
              padding: '24px',
              backgroundColor: '#0a0f1d',
              display: 'flex',
              flexDirection: 'column',
              justifyContent: 'space-between',
              gap: '16px',
              borderRight: '1px solid var(--border-color)',
              overflowY: 'auto'
            }}>
              {/* Image Frame */}
              <div style={{
                flex: 1,
                minHeight: '380px',
                width: '100%',
                borderRadius: '12px',
                overflow: 'hidden',
                backgroundColor: '#000000',
                display: 'flex',
                alignItems: 'center',
                justifyContent: 'center',
                position: 'relative',
                boxShadow: '0 12px 36px rgba(0, 0, 0, 0.4)',
                border: '1px solid rgba(255, 255, 255, 0.1)'
              }}>
                <img 
                  src={currentDisplayImageUrl} 
                  alt={asset.title} 
                  onError={(e) => {
                    const img = e.target as HTMLImageElement;
                    if (!img.src.includes('/api/image-proxy')) {
                      img.src = `${apiBase}/api/image-proxy?url=${encodeURIComponent(asset.url)}`;
                    }
                  }}
                  style={{
                    maxWidth: '100%',
                    maxHeight: '100%',
                    objectFit: 'contain',
                    cursor: 'zoom-in'
                  }}
                  onClick={() => setIsFullscreen(true)}
                />
                <button
                  onClick={() => setIsFullscreen(true)}
                  style={{
                    position: 'absolute',
                    bottom: '12px',
                    right: '12px',
                    background: 'rgba(15, 23, 42, 0.85)',
                    backdropFilter: 'blur(4px)',
                    border: '1px solid rgba(255, 255, 255, 0.2)',
                    color: '#ffffff',
                    padding: '6px 12px',
                    borderRadius: '8px',
                    fontSize: '0.78rem',
                    display: 'flex',
                    alignItems: 'center',
                    gap: '6px',
                    cursor: 'pointer'
                  }}
                >
                  <Maximize2 size={14} /> Fullscreen Zoom
                </button>
              </div>

              {/* Image Toolbar / Asset Link */}
              <div style={{
                display: 'flex',
                alignItems: 'center',
                justifyContent: 'space-between',
                backgroundColor: 'rgba(255, 255, 255, 0.05)',
                padding: '12px 16px',
                borderRadius: '8px',
                border: '1px solid rgba(255, 255, 255, 0.1)',
                color: '#94a3b8',
                fontSize: '0.8rem'
              }}>
                <div>
                  <span style={{ fontWeight: 600, color: '#e2e8f0' }}>Asset Image Spec:</span> High-Res UKAHT Heritage Archive Photo
                </div>
                <a
                  href={currentDisplayImageUrl}
                  target="_blank"
                  rel="noreferrer"
                  className="btn btn-secondary"
                  style={{ padding: '6px 12px', fontSize: '0.75rem', gap: '6px', color: '#ffffff' }}
                >
                  <Download size={14} /> Download Image
                </a>
              </div>
            </div>

            {/* Right Column: Metadata & Annotation & Similar Archive Assets */}
            <div style={{
              padding: '24px',
              overflowY: 'auto',
              display: 'flex',
              flexDirection: 'column',
              gap: '24px',
              backgroundColor: '#ffffff'
            }}>
              {/* Structured Metadata Grid */}
              <div>
                <h4 style={{ fontSize: '0.9rem', fontWeight: 700, color: 'var(--text-secondary)', marginBottom: '12px', textTransform: 'uppercase', letterSpacing: '0.5px' }}>
                  Metadata Attributes
                </h4>
                <div style={{
                  display: 'grid',
                  gridTemplateColumns: '1fr 1fr',
                  gap: '12px',
                  background: '#f8fafc',
                  padding: '16px',
                  borderRadius: '10px',
                  border: '1px solid var(--border-color)'
                }}>
                  <div style={{ display: 'flex', alignItems: 'center', gap: '10px' }}>
                    <MapPin size={16} style={{ color: 'var(--accent-blue)' }} />
                    <div>
                      <div style={{ fontSize: '0.7rem', color: 'var(--text-muted)' }}>Base Code</div>
                      <div style={{ fontSize: '0.88rem', fontWeight: 600, color: 'var(--text-primary)' }}>{asset.base_code || '—'}</div>
                    </div>
                  </div>

                  <div style={{ display: 'flex', alignItems: 'center', gap: '10px' }}>
                    <ImageIcon size={16} style={{ color: 'var(--accent-blue)' }} />
                    <div>
                      <div style={{ fontSize: '0.7rem', color: 'var(--text-muted)' }}>Subject Type</div>
                      <div style={{ fontSize: '0.88rem', fontWeight: 600, color: 'var(--text-primary)', textTransform: 'capitalize' }}>
                        {asset.subject_type || '—'}
                      </div>
                    </div>
                  </div>

                  <div style={{ display: 'flex', alignItems: 'center', gap: '10px' }}>
                    <Calendar size={16} style={{ color: 'var(--accent-blue)' }} />
                    <div>
                      <div style={{ fontSize: '0.7rem', color: 'var(--text-muted)' }}>Shooting Year</div>
                      <div style={{ fontSize: '0.88rem', fontWeight: 600, color: 'var(--text-primary)' }}>{asset.shooting_year || '—'}</div>
                    </div>
                  </div>

                  <div style={{ display: 'flex', alignItems: 'center', gap: '10px' }}>
                    <Copyright size={16} style={{ color: 'var(--accent-blue)' }} />
                    <div>
                      <div style={{ fontSize: '0.7rem', color: 'var(--text-muted)' }}>Copyright Credit</div>
                      <div style={{ fontSize: '0.88rem', fontWeight: 600, color: 'var(--text-primary)', whiteSpace: 'nowrap', overflow: 'hidden', textOverflow: 'ellipsis', maxWidth: '140px' }}>
                        {asset.copyright || 'UKAHT Collection'}
                      </div>
                    </div>
                  </div>

                  <div style={{ display: 'flex', alignItems: 'center', gap: '10px', gridColumn: 'span 2', borderTop: '1px solid var(--border-color)', paddingTop: '10px', marginTop: '4px' }}>
                    <Globe size={16} style={{ color: 'var(--accent-blue)' }} />
                    <div>
                      <div style={{ fontSize: '0.7rem', color: 'var(--text-muted)' }}>Data Source Provenance</div>
                      <div style={{ fontSize: '0.88rem', fontWeight: 600, color: 'var(--text-primary)' }}>
                        {asset.data_source === 'new_addition' ? 'Uploaded Addition (Manual Sync)' : 'UKAHT Official Digitised Archive'}
                      </div>
                    </div>
                  </div>
                </div>
              </div>

              {/* Caption & Description Editor */}
              <div style={{ display: 'flex', flexDirection: 'column', gap: '8px' }}>
                <label style={{ fontSize: '0.9rem', fontWeight: 700, color: 'var(--text-secondary)', textTransform: 'uppercase', letterSpacing: '0.5px' }}>
                  Verified Caption / Description
                </label>
                <textarea
                  value={caption}
                  onChange={(e) => setCaption(e.target.value)}
                  rows={4}
                  placeholder="Type manually verified caption..."
                  style={{
                    resize: 'none',
                    padding: '12px',
                    fontSize: '0.9rem',
                    borderRadius: '8px',
                    border: '1px solid var(--border-color)',
                    backgroundColor: '#ffffff'
                  }}
                />
                
                <div style={{ display: 'flex', gap: '10px', marginTop: '4px' }}>
                  <button
                    onClick={handleSaveCaption}
                    disabled={saving}
                    className="btn btn-primary"
                    style={{ flex: 1, padding: '10px', fontSize: '0.85rem' }}
                  >
                    {saving ? <RefreshCw size={16} className="animate-spin" /> : 'Save Annotation'}
                  </button>

                  <button
                    onClick={handleVerifyGolden}
                    disabled={goldenSaving}
                    className="btn btn-secondary"
                    style={{ padding: '10px 16px', display: 'flex', alignItems: 'center', gap: '6px', fontSize: '0.85rem' }}
                    title="Add to Golden Test Set"
                  >
                    {goldenSaving ? <RefreshCw size={16} className="animate-spin" /> : <CheckCircle size={16} style={{ color: 'var(--success)' }} />}
                    Golden Set
                  </button>
                </div>

                {saveMessage && (
                  <div style={{ 
                    fontSize: '0.82rem', 
                    color: saveMessage.includes('Error') ? 'var(--error)' : 'var(--success)',
                    marginTop: '4px'
                  }}>
                    {saveMessage}
                  </div>
                )}

                {goldenMessage && (
                  <div style={{ 
                    fontSize: '0.82rem', 
                    color: 'var(--success)',
                    marginTop: '4px',
                    display: 'flex',
                    alignItems: 'center',
                    gap: '4px'
                  }}>
                    <Bookmark size={14} /> {goldenMessage}
                  </div>
                )}
              </div>

              {/* Related Visual Assets Showcase */}
              <div style={{ display: 'flex', flexDirection: 'column', gap: '12px', borderTop: '1px solid var(--border-color)', paddingTop: '20px' }}>
                <h4 style={{ fontSize: '0.9rem', fontWeight: 700, color: 'var(--text-secondary)', textTransform: 'uppercase', letterSpacing: '0.5px' }}>
                  Related Visual Archive Photos
                </h4>
                
                {loadingSimilar ? (
                  <div style={{ display: 'flex', alignItems: 'center', gap: '8px', fontSize: '0.85rem', color: 'var(--text-muted)', padding: '10px 0' }}>
                    <RefreshCw size={14} className="animate-spin" /> Loading related archive assets...
                  </div>
                ) : similarAssets.length === 0 ? (
                  <div style={{ fontSize: '0.85rem', color: 'var(--text-muted)', fontStyle: 'italic' }}>
                    No related visual assets found.
                  </div>
                ) : (
                  <div style={{
                    display: 'grid',
                    gridTemplateColumns: 'repeat(3, 1fr)',
                    gap: '12px'
                  }}>
                    {similarAssets.slice(0, 6).map((item) => (
                      <div 
                        key={item.id}
                        onClick={() => onSelectAsset && onSelectAsset(item.id)}
                        style={{
                          borderRadius: '8px',
                          overflow: 'hidden',
                          backgroundColor: '#ffffff',
                          border: '1px solid var(--border-color)',
                          cursor: 'pointer',
                          boxShadow: '0 2px 6px rgba(0, 0, 0, 0.04)',
                          transition: 'all 0.25s cubic-bezier(0.16, 1, 0.3, 1)'
                        }}
                        onMouseEnter={(e) => {
                          e.currentTarget.style.borderColor = 'var(--accent-blue)';
                          e.currentTarget.style.transform = 'translateY(-2px)';
                          e.currentTarget.style.boxShadow = '0 6px 16px rgba(0, 102, 204, 0.15)';
                        }}
                        onMouseLeave={(e) => {
                          e.currentTarget.style.borderColor = 'var(--border-color)';
                          e.currentTarget.style.transform = 'translateY(0)';
                          e.currentTarget.style.boxShadow = '0 2px 6px rgba(0, 0, 0, 0.04)';
                        }}
                      >
                        <div style={{ height: '75px', backgroundColor: '#050a14', overflow: 'hidden' }}>
                          <img 
                            src={getImageUrl(item.url)} 
                            alt={item.title} 
                            onError={(e) => {
                              const img = e.target as HTMLImageElement;
                              if (!img.src.includes('/api/image-proxy')) {
                                img.src = `${apiBase}/api/image-proxy?url=${encodeURIComponent(item.url)}`;
                              }
                            }}
                            style={{ width: '100%', height: '100%', objectFit: 'cover' }}
                          />
                        </div>
                        <div style={{
                          padding: '6px 8px',
                          fontSize: '0.75rem',
                          fontWeight: 600,
                          whiteSpace: 'nowrap',
                          overflow: 'hidden',
                          textOverflow: 'ellipsis',
                          color: 'var(--text-primary)'
                        }}>
                          {item.title}
                        </div>
                      </div>
                    ))}
                  </div>
                )}
              </div>
            </div>
          </div>
        </div>
      </div>
      
      {/* CSS Keyframe Animations */}
      <style>{`
        @keyframes fadeIn {
          from { opacity: 0; }
          to { opacity: 1; }
        }
        @keyframes slideUp {
          from { opacity: 0; transform: translateY(20px) scale(0.98); }
          to { opacity: 1; transform: translateY(0) scale(1); }
        }
      `}</style>
    </>
  );
}
