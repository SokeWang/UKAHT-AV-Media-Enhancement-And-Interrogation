import { useState, useEffect } from 'react';
import { X, CheckCircle, RefreshCw, Image as ImageIcon, MapPin, Calendar, Copyright, Globe, Bookmark } from 'lucide-react';

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

  const fetchSimilar = async (id: string) => {
    setLoadingSimilar(true);
    try {
      const response = await fetch(`${apiBase}/api/recommend`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ id, limit: 5 })
      });
      if (response.ok) {
        const resJson = await response.json();
        if (resJson.code === 200) {
          // Filter out self
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
          setGoldenMessage('Saved to Golden Set!');
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

  const getImageUrl = (url: string) => {
    if (url.startsWith('http')) return url;
    if (url.startsWith('/')) return `${apiBase}${url}`;
    return url;
  };

  return (
    <div style={{
      position: 'fixed',
      top: 0,
      right: 0,
      height: '100vh',
      width: '420px',
      backgroundColor: 'var(--bg-secondary)',
      borderLeft: '1px solid var(--border-color)',
      boxShadow: '-8px 0 32px rgba(10, 15, 30, 0.15)',
      display: 'flex',
      flexDirection: 'column',
      zIndex: 1000,
      animation: 'slideInRight 0.3s cubic-bezier(0.25, 0.8, 0.25, 1) forwards'
    }}>
      {/* Header */}
      <div style={{
        padding: '20px',
        borderBottom: '1px solid var(--border-color)',
        display: 'flex',
        alignItems: 'center',
        justifyContent: 'space-between'
      }}>
        <div>
          <h3 style={{ fontSize: '1.2rem', color: 'var(--text-primary)', marginBottom: '4px' }}>
            Asset Details
          </h3>
          <span className="badge badge-accent" style={{ fontSize: '0.7rem' }}>
            {asset.category || 'Archive'}
          </span>
        </div>
        <button 
          onClick={onClose}
          style={{
            background: 'none',
            border: 'none',
            color: 'var(--text-secondary)',
            cursor: 'pointer',
            padding: '8px',
            borderRadius: '50%',
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'center',
            transition: 'background var(--transition-fast)'
          }}
          onMouseEnter={(e) => e.currentTarget.style.backgroundColor = 'rgba(0, 102, 204, 0.05)'}
          onMouseLeave={(e) => e.currentTarget.style.backgroundColor = 'transparent'}
        >
          <X size={20} />
        </button>
      </div>

      {/* Scrollable Content */}
      <div style={{
        flex: 1,
        overflowY: 'auto',
        padding: '20px',
        display: 'flex',
        flexDirection: 'column',
        gap: '24px'
      }}>
        {/* Large Image Preview */}
        <div style={{
          width: '100%',
          height: '240px',
          borderRadius: '10px',
          overflow: 'hidden',
          backgroundColor: '#050a14',
          position: 'relative',
          boxShadow: '0 4px 12px rgba(0,0,0,0.1)'
        }}>
          <img 
            src={getImageUrl(asset.url)} 
            alt={asset.title} 
            style={{ width: '100%', height: '100%', objectFit: 'contain' }}
          />
        </div>

        {/* Title */}
        <div>
          <h4 style={{ fontSize: '1.1rem', fontWeight: 600, color: 'var(--text-primary)', marginBottom: '4px' }}>
            {asset.title}
          </h4>
          <span style={{ fontSize: '0.8rem', color: 'var(--text-muted)' }}>ID: {asset.id}</span>
        </div>

        {/* Metadata Grid */}
        <div style={{
          display: 'grid',
          gridTemplateColumns: '1fr 1fr',
          gap: '12px',
          background: 'rgba(94, 168, 241, 0.05)',
          padding: '16px',
          borderRadius: '8px',
          border: '1px solid rgba(94, 168, 241, 0.1)'
        }}>
          <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
            <MapPin size={15} style={{ color: 'var(--accent-cyan)' }} />
            <div>
              <div style={{ fontSize: '0.7rem', color: 'var(--text-muted)' }}>Base Code</div>
              <div style={{ fontSize: '0.85rem', fontWeight: 500 }}>{asset.base_code || '—'}</div>
            </div>
          </div>

          <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
            <ImageIcon size={15} style={{ color: 'var(--accent-cyan)' }} />
            <div>
              <div style={{ fontSize: '0.7rem', color: 'var(--text-muted)' }}>Subject</div>
              <div style={{ fontSize: '0.85rem', fontWeight: 500, textTransform: 'capitalize' }}>
                {asset.subject_type || '—'}
              </div>
            </div>
          </div>

          <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
            <Calendar size={15} style={{ color: 'var(--accent-cyan)' }} />
            <div>
              <div style={{ fontSize: '0.7rem', color: 'var(--text-muted)' }}>Year</div>
              <div style={{ fontSize: '0.85rem', fontWeight: 500 }}>{asset.shooting_year || '—'}</div>
            </div>
          </div>

          <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
            <Copyright size={15} style={{ color: 'var(--accent-cyan)' }} />
            <div>
              <div style={{ fontSize: '0.7rem', color: 'var(--text-muted)' }}>Copyright</div>
              <div style={{ fontSize: '0.85rem', fontWeight: 500, whiteSpace: 'nowrap', overflow: 'hidden', textOverflow: 'ellipsis', maxWidth: '140px' }}>
                {asset.copyright || '—'}
              </div>
            </div>
          </div>

          <div style={{ display: 'flex', alignItems: 'center', gap: '8px', gridColumn: 'span 2', borderTop: '1px solid rgba(94, 168, 241, 0.1)', paddingTop: '8px', marginTop: '4px' }}>
            <Globe size={15} style={{ color: 'var(--accent-cyan)' }} />
            <div>
              <div style={{ fontSize: '0.7rem', color: 'var(--text-muted)' }}>Source</div>
              <div style={{ fontSize: '0.85rem', fontWeight: 500, textTransform: 'capitalize' }}>
                {asset.data_source === 'new_addition' ? 'Uploaded Addition' : 'Original Archive'}
              </div>
            </div>
          </div>
        </div>

        {/* Caption Editor */}
        <div style={{ display: 'flex', flexDirection: 'column', gap: '8px' }}>
          <label style={{ fontSize: '0.85rem', fontWeight: 600, color: 'var(--text-primary)' }}>
            Description (Caption)
          </label>
          <textarea
            value={caption}
            onChange={(e) => setCaption(e.target.value)}
            rows={4}
            placeholder="Type manually verified caption..."
            style={{ resize: 'none' }}
          />
          
          <div style={{ display: 'flex', gap: '8px', marginTop: '4px' }}>
            <button
              onClick={handleSaveCaption}
              disabled={saving}
              className="btn btn-primary"
              style={{ flex: 1, padding: '10px' }}
            >
              {saving ? <RefreshCw size={16} className="animate-spin" /> : 'Save Annotation'}
            </button>

            <button
              onClick={handleVerifyGolden}
              disabled={goldenSaving}
              className="btn btn-secondary"
              style={{ padding: '10px', display: 'flex', alignItems: 'center', gap: '6px' }}
              title="Add to Golden Test Set"
            >
              {goldenSaving ? <RefreshCw size={16} className="animate-spin" /> : <CheckCircle size={16} style={{ color: 'var(--success)' }} />}
              Verify
            </button>
          </div>

          {saveMessage && (
            <div style={{ 
              fontSize: '0.8rem', 
              color: saveMessage.includes('Error') ? 'var(--error)' : 'var(--success)',
              marginTop: '4px'
            }}>
              {saveMessage}
            </div>
          )}

          {goldenMessage && (
            <div style={{ 
              fontSize: '0.8rem', 
              color: 'var(--success)',
              marginTop: '4px',
              display: 'flex',
              alignItems: 'center',
              gap: '4px'
            }}>
              <Bookmark size={12} /> {goldenMessage}
            </div>
          )}
        </div>

        {/* Similar Images Recommendations */}
        <div style={{ display: 'flex', flexDirection: 'column', gap: '10px', borderTop: '1px solid var(--border-color)', paddingTop: '20px' }}>
          <h5 style={{ fontSize: '0.9rem', fontWeight: 600, color: 'var(--text-primary)' }}>
            Similar Images
          </h5>
          
          {loadingSimilar ? (
            <div style={{ display: 'flex', alignItems: 'center', gap: '8px', fontSize: '0.8rem', color: 'var(--text-muted)', padding: '10px 0' }}>
              <RefreshCw size={12} className="animate-spin" /> Loading recommendations...
            </div>
          ) : similarAssets.length === 0 ? (
            <div style={{ fontSize: '0.8rem', color: 'var(--text-muted)', fontStyle: 'italic' }}>
              No similar images found.
            </div>
          ) : (
            <div style={{
              display: 'grid',
              gridTemplateColumns: 'repeat(2, 1fr)',
              gap: '10px'
            }}>
              {similarAssets.map((item) => (
                <div 
                  key={item.id}
                  onClick={() => onSelectAsset && onSelectAsset(item.id)}
                  style={{
                    borderRadius: '6px',
                    overflow: 'hidden',
                    backgroundColor: 'rgba(10, 15, 30, 0.05)',
                    border: '1px solid var(--border-color)',
                    cursor: 'pointer',
                    transition: 'all var(--transition-fast)'
                  }}
                  onMouseEnter={(e) => {
                    e.currentTarget.style.borderColor = 'var(--accent-blue)';
                    e.currentTarget.style.transform = 'scale(1.02)';
                  }}
                  onMouseLeave={(e) => {
                    e.currentTarget.style.borderColor = 'var(--border-color)';
                    e.currentTarget.style.transform = 'scale(1.0)';
                  }}
                >
                  <div style={{ height: '70px', backgroundColor: '#000', overflow: 'hidden' }}>
                    <img 
                      src={getImageUrl(item.url)} 
                      alt={item.title} 
                      style={{ width: '100%', height: '100%', objectFit: 'cover' }}
                    />
                  </div>
                  <div style={{
                    padding: '6px',
                    fontSize: '0.75rem',
                    fontWeight: 500,
                    whiteSpace: 'nowrap',
                    overflow: 'hidden',
                    textOverflow: 'ellipsis'
                  }}>
                    {item.title}
                  </div>
                </div>
              ))}
            </div>
          )}
        </div>
      </div>
      
      {/* CSS Animation injection */}
      <style>{`
        @keyframes slideInRight {
          from { transform: translateX(100%); }
          to { transform: translateX(0); }
        }
      `}</style>
    </div>
  );
}
