import React, { useState, useEffect } from 'react';
import { Save, Star, ChevronLeft, ChevronRight, FileJson, Check, AlertTriangle, Loader2, Upload } from 'lucide-react';
import { UploadModal } from './UploadModal';

interface Asset {
  id: string;
  url: string;
  title: string;
  category: string;
  description: string;
}

interface GoldenEntry {
  asset_id: string;
  caption: string;
  query: string;
  relevant_ids: string[];
}

interface AnnotatorProps {
  apiBase: string;
}

export const Annotator: React.FC<AnnotatorProps> = ({ apiBase }) => {
  const [assets, setAssets] = useState<Asset[]>([]);
  const [loading, setLoading] = useState(true);
  const [activeIndex, setActiveIndex] = useState(0);
  const [currentCaption, setCurrentCaption] = useState('');
  
  const [goldenList, setGoldenList] = useState<GoldenEntry[]>([]);
  const [saveLoading, setSaveLoading] = useState(false);
  const [goldenLoading, setGoldenLoading] = useState(false);
  const [message, setMessage] = useState<{ type: 'success' | 'error' | 'info'; text: string } | null>(null);
  const [uploadOpen, setUploadOpen] = useState(false);

  // Fetch all assets and the golden test set on mount
  useEffect(() => {
    fetchData();
  }, []);

  const fetchData = async (selectAssetId?: string) => {
    setLoading(true);
    try {
      // 1. Fetch assets (try assets first, fall back to empty query search)
      let assetsData: Asset[] = [];
      const resAssets = await fetch(`${apiBase}/api/assets`);
      if (resAssets.ok) {
        const resJson = await resAssets.json();
        if (resJson.code === 200) {
          assetsData = resJson.data;
        }
      } else {
        const resSearch = await fetch(`${apiBase}/api/search`, {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ query: '' }),
        });
        if (resSearch.ok) {
          const resJson = await resSearch.json();
          if (resJson.code === 200) {
            assetsData = resJson.data;
          }
        }
      }
      setAssets(assetsData);

      // Set caption & active index
      if (assetsData.length > 0) {
        if (selectAssetId) {
          const index = assetsData.findIndex((a) => a.id === selectAssetId);
          if (index !== -1) {
            setActiveIndex(index);
            setCurrentCaption(assetsData[index].description || '');
          } else {
            setActiveIndex(0);
            setCurrentCaption(assetsData[0].description || '');
          }
        } else {
          setActiveIndex(0);
          setCurrentCaption(assetsData[0].description || '');
        }
      }

      // 2. Fetch Golden List
      const resGolden = await fetch(`${apiBase}/api/golden`);
      if (resGolden.ok) {
        const resJson = await resGolden.json();
        if (resJson.code === 200) {
          setGoldenList(resJson.data);
        }
      }
    } catch (err) {
      console.error('Error fetching annotator data:', err);
    } finally {
      setLoading(false);
    }
  };

  const handleUploadSuccess = (newAsset: any) => {
    if (newAsset && newAsset.id) {
      fetchData(newAsset.id);
    } else {
      fetchData();
    }
    setUploadOpen(false);
  };

  // Sync current caption state when active asset index changes
  useEffect(() => {
    if (assets.length > 0 && activeIndex < assets.length) {
      setCurrentCaption(assets[activeIndex].description || '');
      setMessage(null);
    }
  }, [activeIndex, assets]);

  const handleIndexChange = (newIndex: number) => {
    if (newIndex >= 0 && newIndex < assets.length) {
      setActiveIndex(newIndex);
    }
  };

  const handleSaveCaption = async () => {
    if (assets.length === 0) return;
    const activeAsset = assets[activeIndex];
    
    setSaveLoading(true);
    setMessage(null);

    try {
      const response = await fetch(`${apiBase}/api/assets/${activeAsset.id}/caption`, {
        method: 'PUT',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ caption: currentCaption }),
      });

      const resJson = await response.json();
      if (resJson.code !== 200) throw new Error(resJson.message || 'Failed to update caption');

      // Update local asset list description
      const updatedAssets = [...assets];
      updatedAssets[activeIndex].description = currentCaption;
      setAssets(updatedAssets);

      setMessage({ type: 'success', text: 'Caption successfully saved to database.' });
    } catch (err: any) {
      setMessage({ type: 'error', text: err.message || 'Failed to save caption.' });
    } finally {
      setSaveLoading(false);
    }
  };

  const handleAddToGolden = async () => {
    if (assets.length === 0) return;
    const activeAsset = assets[activeIndex];

    setGoldenLoading(true);
    setMessage(null);

    try {
      // Step 1: Save the caption first to sync DB
      const resPut = await fetch(`${apiBase}/api/assets/${activeAsset.id}/caption`, {
        method: 'PUT',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ caption: currentCaption }),
      });
      const resPutJson = await resPut.json();
      if (resPutJson.code !== 200) throw new Error(resPutJson.message || 'Failed to save caption');

      const updatedAssets = [...assets];
      updatedAssets[activeIndex].description = currentCaption;
      setAssets(updatedAssets);

      // Step 2: Add to Golden Set
      const response = await fetch(`${apiBase}/api/golden`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ asset_id: activeAsset.id, caption: currentCaption }),
      });

      const resJson = await response.json();
      if (resJson.code !== 200) throw new Error(resJson.message || 'Failed to add to golden test set');

      const goldenResult = resJson.data;
      
      if (goldenResult.status === 'exists') {
        setMessage({ type: 'info', text: 'This image is already in the golden set.' });
      } else {
        // Refetch golden list
        const resGolden = await fetch(`${apiBase}/api/golden`);
        if (resGolden.ok) {
          const resGoldenJson = await resGolden.json();
          if (resGoldenJson.code === 200) {
            setGoldenList(resGoldenJson.data);
          }
        }
        setMessage({ type: 'success', text: `Added to golden set! (${goldenList.length + 1} total verified entries)` });
      }
    } catch (err: any) {
      setMessage({ type: 'error', text: err.message || 'Failed to add to golden set.' });
    } finally {
      setGoldenLoading(false);
    }
  };

  const handleDownloadGolden = () => {
    if (goldenList.length === 0) return;
    const dataStr = 'data:text/json;charset=utf-8,' + encodeURIComponent(JSON.stringify(goldenList, null, 2));
    const downloadAnchor = document.createElement('a');
    downloadAnchor.setAttribute('href', dataStr);
    downloadAnchor.setAttribute('download', 'golden_test_set.json');
    document.body.appendChild(downloadAnchor);
    downloadAnchor.click();
    downloadAnchor.remove();
  };

  const getImageUrl = (url: string) => {
    if (url.startsWith('http')) return url;
    if (url.startsWith('/')) return `${apiBase}${url}`;
    return url;
  };

  if (loading) {
    return (
      <div style={{ textAlign: 'center', padding: '100px 20px', color: 'var(--text-secondary)' }}>
        <Loader2 size={40} className="animate-spin" style={{ animation: 'spin 1.5s linear infinite', margin: '0 auto 16px auto' }} />
        Loading Caption Annotator...
      </div>
    );
  }

  if (assets.length === 0) {
    return (
      <div className="glass-panel" style={{ textAlign: 'center', padding: '60px 20px', color: 'var(--text-secondary)' }}>
        <AlertTriangle size={36} style={{ color: 'var(--warning)', marginBottom: '12px' }} />
        <h3 style={{ marginBottom: '8px' }}>No Images Found</h3>
        <p style={{ color: 'var(--text-muted)' }}>No images in the database yet. Run ingestion or upload new photos first.</p>
      </div>
    );
  }

  const activeAsset = assets[activeIndex];
  const isGolden = goldenList.some((e) => e.asset_id === activeAsset.id);
  const goldenProgress = Math.min(goldenList.length / 100, 1.0);

  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: '20px' }}>
      {/* Header */}
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start', flexWrap: 'wrap', gap: '15px' }}>
        <div>
          <h1 className="glow-text" style={{ fontSize: '2rem', marginBottom: '8px', fontFamily: 'var(--font-family-title)' }}>
            ✏️ Caption Annotation Workspace
          </h1>
          <p style={{ color: 'var(--text-secondary)', fontSize: '0.95rem' }}>
            Review auto-generated captions, correct any errors, and mark verified image-text pairs as part of the <strong>Golden Test Set</strong>.
          </p>
        </div>
        <button 
          onClick={() => setUploadOpen(true)}
          className="btn btn-primary"
          style={{ gap: '6px' }}
        >
          <Upload size={16} />
          Upload & Index Image
        </button>
      </div>

      {/* Main Annotation Panel */}
      <div style={{
        display: 'grid',
        gridTemplateColumns: 'repeat(auto-fit, minmax(350px, 1fr))',
        gap: '24px'
      }}>
        {/* Left Side: Image Display Container */}
        <div className="glass-panel" style={{
          padding: '24px',
          display: 'flex',
          flexDirection: 'column',
          alignItems: 'center',
          justifyContent: 'space-between',
          gap: '20px',
          minHeight: '450px'
        }}>
          {/* Pagination bar */}
          <div style={{ display: 'flex', alignItems: 'center', gap: '15px', width: '100%', justifyContent: 'space-between' }}>
            <button
              onClick={() => handleIndexChange(activeIndex - 1)}
              disabled={activeIndex === 0}
              className="btn btn-secondary"
              style={{ padding: '8px 12px' }}
            >
              <ChevronLeft size={16} />
            </button>

            <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
              <span style={{ fontSize: '0.9rem', color: 'var(--text-secondary)' }}>Image</span>
              <input
                type="text"
                value={activeIndex + 1}
                onChange={(e) => {
                  const val = parseInt(e.target.value);
                  if (!isNaN(val)) handleIndexChange(val - 1);
                }}
                style={{ width: '60px', textAlign: 'center', padding: '6px 8px' }}
              />
              <span style={{ fontSize: '0.9rem', color: 'var(--text-muted)' }}>/ {assets.length}</span>
            </div>

            <button
              onClick={() => handleIndexChange(activeIndex + 1)}
              disabled={activeIndex === assets.length - 1}
              className="btn btn-secondary"
              style={{ padding: '8px 12px' }}
            >
              <ChevronRight size={16} />
            </button>
          </div>

          {/* Active Image */}
          <div style={{
            width: '100%',
            height: '280px',
            overflow: 'hidden',
            borderRadius: '8px',
            background: '#050a14',
            border: '1px solid var(--border-color)',
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'center',
            position: 'relative'
          }}>
            <img
              src={getImageUrl(activeAsset.url)}
              alt={activeAsset.title}
              style={{ maxWidth: '100%', maxHeight: '100%', objectFit: 'contain' }}
            />
            {isGolden && (
              <div style={{
                position: 'absolute',
                top: '12px',
                right: '12px',
                background: 'rgba(0, 210, 255, 0.9)',
                color: '#070d19',
                padding: '4px 8px',
                borderRadius: '4px',
                fontSize: '0.75rem',
                fontWeight: '600',
                display: 'flex',
                alignItems: 'center',
                gap: '4px',
                boxShadow: '0 0 10px var(--accent-glow-strong)'
              }}>
                <Star size={12} fill="#070d19" />
                Golden
              </div>
            )}
          </div>

          {/* Image Meta info */}
          <div style={{ width: '100%', display: 'flex', justifyContent: 'space-between', fontSize: '0.85rem', color: 'var(--text-secondary)' }}>
            <span><strong>ID:</strong> {activeAsset.id}</span>
            <span><strong>Category:</strong> {activeAsset.category || '—'}</span>
          </div>
        </div>

        {/* Right Side: Caption Editing Form */}
        <div className="glass-panel" style={{
          padding: '24px',
          display: 'flex',
          flexDirection: 'column',
          justifyContent: 'space-between',
          gap: '20px'
        }}>
          <div>
            <h3 style={{ fontSize: '1.25rem', marginBottom: '15px', borderBottom: '1px solid var(--border-color)', paddingBottom: '8px' }}>
              {activeAsset.title || 'Untitled'}
            </h3>

            <div style={{ display: 'flex', flexDirection: 'column', gap: '8px', marginBottom: '20px' }}>
              <label style={{ fontSize: '0.85rem', color: 'var(--text-secondary)', fontWeight: '500' }}>
                Caption (edit to correct)
              </label>
              <textarea
                value={currentCaption}
                onChange={(e) => setCurrentCaption(e.target.value)}
                style={{ height: '140px', resize: 'none', lineHeight: '1.5' }}
                placeholder="Enter caption text here..."
              />
            </div>

            {/* Notification messages */}
            {message && (
              <div className="animate-fade" style={{
                padding: '12px 16px',
                borderRadius: '6px',
                fontSize: '0.88rem',
                display: 'flex',
                alignItems: 'center',
                gap: '8px',
                background: message.type === 'success' ? 'rgba(0, 230, 118, 0.08)' : message.type === 'error' ? 'rgba(255, 82, 82, 0.08)' : 'rgba(0, 210, 255, 0.08)',
                border: message.type === 'success' ? '1px solid rgba(0, 230, 118, 0.25)' : message.type === 'error' ? '1px solid rgba(255, 82, 82, 0.25)' : '1px solid rgba(0, 210, 255, 0.25)',
                color: message.type === 'success' ? 'var(--success)' : message.type === 'error' ? 'var(--error)' : 'var(--accent-cyan)'
              }}>
                {message.type === 'success' && <Check size={16} />}
                <span>{message.text}</span>
              </div>
            )}
          </div>

          {/* Action Buttons */}
          <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '15px' }}>
            <button
              onClick={handleSaveCaption}
              disabled={saveLoading || goldenLoading}
              className="btn btn-secondary"
              style={{ padding: '14px', width: '100%' }}
            >
              {saveLoading ? (
                <Loader2 size={16} className="animate-spin" style={{ animation: 'spin 1s linear infinite' }} />
              ) : (
                <Save size={16} />
              )}
              <span>Save Caption</span>
            </button>

            <button
              onClick={handleAddToGolden}
              disabled={saveLoading || goldenLoading}
              className="btn btn-primary"
              style={{ padding: '14px', width: '100%' }}
            >
              {goldenLoading ? (
                <Loader2 size={16} className="animate-spin" style={{ animation: 'spin 1s linear infinite' }} />
              ) : (
                <Star size={16} />
              )}
              <span>Add to Golden Set</span>
            </button>
          </div>
        </div>
      </div>

      {/* Golden Set Summary Panel */}
      <div className="glass-panel animate-fade" style={{ padding: '24px', marginTop: '10px' }}>
        <div style={{ display: 'flex', flexWrap: 'wrap', justifyContent: 'space-between', alignItems: 'center', gap: '15px', marginBottom: '16px' }}>
          <div>
            <h3 style={{ fontSize: '1.1rem', marginBottom: '4px' }}>Golden Test Set Progress</h3>
            <p style={{ fontSize: '0.82rem', color: 'var(--text-secondary)' }}>
              Target: <strong>50 – 100</strong> verified entries. Currently verified: <strong>{goldenList.length}</strong>
            </p>
          </div>

          <button
            onClick={handleDownloadGolden}
            disabled={goldenList.length === 0}
            className="btn btn-secondary"
            style={{ fontSize: '0.85rem', gap: '8px' }}
          >
            <FileJson size={16} />
            Download golden_test_set.json
          </button>
        </div>

        {/* Progress Bar */}
        <div style={{
          width: '100%',
          height: '10px',
          background: 'rgba(94, 168, 241, 0.1)',
          borderRadius: '5px',
          overflow: 'hidden',
          position: 'relative'
        }}>
          <div style={{
            width: `${goldenProgress * 100}%`,
            height: '100%',
            background: 'linear-gradient(90deg, var(--accent-blue), var(--accent-cyan))',
            borderRadius: '5px',
            boxShadow: '0 0 8px var(--accent-cyan)',
            transition: 'width 0.4s ease'
          }} />
        </div>
      </div>

      {/* Upload Modal */}
      <UploadModal
        isOpen={uploadOpen}
        onClose={() => setUploadOpen(false)}
        onUploadSuccess={handleUploadSuccess}
        apiBase={apiBase}
      />
    </div>
  );
};
