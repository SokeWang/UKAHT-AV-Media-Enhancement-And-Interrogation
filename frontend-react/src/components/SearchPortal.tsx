import React, { useState, useEffect } from 'react';
import { Search, Image as ImageIcon, RefreshCw } from 'lucide-react';

interface Asset {
  id: string;
  url: string;
  title: string;
  category: string;
  description: string;
  score?: number;
}

interface SearchPortalProps {
  apiBase: string;
  activeQuery: string;
  activeCategory: string;
  similarForId: string | null;
  setSimilarForId: (id: string | null) => void;
  searchResults: Asset[];
  loading: boolean;
  onSearchSubmit: (query: string, category: string) => void;
  isSplitView: boolean;
}

export const SearchPortal: React.FC<SearchPortalProps> = ({
  apiBase,
  activeQuery,
  activeCategory,
  similarForId,
  setSimilarForId,
  searchResults,
  loading,
  onSearchSubmit,
  isSplitView,
}) => {
  const [queryInput, setQueryInput] = useState(activeQuery);
  const [categoryInput, setCategoryInput] = useState(activeCategory);
  const [recommendations, setRecommendations] = useState<Asset[]>([]);
  const [recLoading, setRecLoading] = useState(false);

  // Sync inputs with active parent state (e.g. if reset or search triggers from outside)
  useEffect(() => {
    setQueryInput(activeQuery);
  }, [activeQuery]);

  useEffect(() => {
    setCategoryInput(activeCategory);
  }, [activeCategory]);

  // Load recommendations when similarForId changes
  useEffect(() => {
    if (similarForId) {
      fetchRecommendations(similarForId);
    } else {
      setRecommendations([]);
    }
  }, [similarForId]);

  const fetchRecommendations = async (assetId: string) => {
    setRecLoading(true);
    try {
      const response = await fetch(`${apiBase}/api/recommend`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ id: assetId, limit: 4 }),
      });
      if (response.ok) {
        const resJson = await response.json();
        if (resJson.code === 200) {
          setRecommendations(resJson.data);
        } else {
          console.error('Failed to get recommendations:', resJson.message);
        }
      }
    } catch (err) {
      console.error('Failed to fetch recommendations:', err);
    } finally {
      setRecLoading(false);
    }
  };

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    onSearchSubmit(queryInput, categoryInput);
  };

  const getImageUrl = (url: string) => {
    if (url.startsWith('http')) return url;
    if (url.startsWith('/')) return `${apiBase}${url}`;
    return url;
  };

  const catOptions = ["", "Landscape", "Building", "Equipment", "Wildlife", "People", "Uploaded"];

  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: '20px' }}>
      {/* Search Header */}
      <div>
        <h1 className="glow-text" style={{ fontSize: '2rem', marginBottom: '8px', fontFamily: 'var(--font-family-title)' }}>
          🔍 Smart Search & Q&A Portal
        </h1>
        <p style={{ color: 'var(--text-secondary)', fontSize: '0.95rem' }}>
          Search the UKAHT historical photograph archive instantly, and ask follow-up questions to explore stories behind the collection.
        </p>
      </div>

      {/* Search Form Panel */}
      <form onSubmit={handleSubmit} className="glass-panel" style={{
        padding: '20px',
        display: 'flex',
        flexWrap: 'wrap',
        gap: '15px',
        alignItems: 'flex-end'
      }}>
        <div style={{ flex: '2 1 300px', display: 'flex', flexDirection: 'column', gap: '8px' }}>
          <label style={{ fontSize: '0.85rem', color: 'var(--text-secondary)', fontWeight: '500' }}>Search or Ask</label>
          <div style={{ position: 'relative' }}>
            <input
              type="text"
              value={queryInput}
              onChange={(e) => setQueryInput(e.target.value)}
              placeholder="e.g. dogs, station leaders, or Ask 'What was the task of Milestone 3?'"
              style={{ paddingLeft: '40px' }}
            />
            <Search size={18} style={{
              position: 'absolute',
              left: '14px',
              top: '50%',
              transform: 'translateY(-50%)',
              color: 'var(--text-muted)'
            }} />
          </div>
        </div>

        <div style={{ flex: '1 1 180px', display: 'flex', flexDirection: 'column', gap: '8px' }}>
          <label style={{ fontSize: '0.85rem', color: 'var(--text-secondary)', fontWeight: '500' }}>Category Filter</label>
          <select
            value={categoryInput}
            onChange={(e) => setCategoryInput(e.target.value)}
          >
            {catOptions.map((opt) => (
              <option key={opt} value={opt}>
                {opt === "" ? "All Categories" : opt}
              </option>
            ))}
          </select>
        </div>

        <div style={{ flex: '0 0 auto', display: 'flex', gap: '10px' }}>
          <button type="submit" className="btn btn-primary" style={{ padding: '12px 24px' }}>
            Search & Ask
          </button>
        </div>
      </form>

      {/* Main Results Grid */}
      <div style={{ display: 'flex', flexDirection: 'column', gap: '15px' }}>
        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
          <h3 style={{ fontSize: '1.2rem', color: 'var(--text-primary)', display: 'flex', alignItems: 'center', gap: '8px' }}>
            <ImageIcon size={18} style={{ color: 'var(--accent-blue)' }} />
            {activeQuery ? `Search Results for "${activeQuery}"` : "Archived Historical Photos"}
          </h3>
          <span style={{ fontSize: '0.85rem', color: 'var(--text-secondary)', background: 'rgba(94, 168, 241, 0.1)', padding: '4px 10px', borderRadius: '12px' }}>
            <strong>{searchResults.length}</strong> {searchResults.length === 1 ? 'image' : 'images'}
          </span>
        </div>

        {loading ? (
          <div style={{ textAlign: 'center', padding: '60px 20px', color: 'var(--text-secondary)' }}>
            <RefreshCw size={32} className="animate-spin" style={{ animation: 'spin 1.5s linear infinite', margin: '0 auto 12px auto' }} />
            Loading archive data...
          </div>
        ) : searchResults.length === 0 ? (
          <div className="glass-panel" style={{ textAlign: 'center', padding: '40px 20px', color: 'var(--text-muted)' }}>
            No images match the query directly.
          </div>
        ) : (
          <div style={{
            display: 'grid',
            gridTemplateColumns: isSplitView ? 'repeat(2, 1fr)' : 'repeat(auto-fill, minmax(280px, 1fr))',
            gap: '20px'
          }}>
            {searchResults.map((asset) => (
              <div 
                key={asset.id} 
                className="glass-panel animate-slide-up"
                style={{
                  display: 'flex',
                  flexDirection: 'column',
                  overflow: 'hidden',
                  position: 'relative',
                  border: similarForId === asset.id ? '1px solid var(--accent-blue)' : undefined,
                  boxShadow: similarForId === asset.id ? '0 0 15px var(--accent-glow)' : undefined
                }}
              >
                {/* Image Container */}
                <div style={{ height: '180px', overflow: 'hidden', background: '#050a14', position: 'relative' }}>
                  <img
                    src={getImageUrl(asset.url)}
                    alt={asset.title}
                    style={{
                      width: '100%',
                      height: '100%',
                      objectFit: 'cover',
                      transition: 'transform var(--transition-normal)'
                    }}
                    onMouseEnter={(e) => e.currentTarget.style.transform = 'scale(1.05)'}
                    onMouseLeave={(e) => e.currentTarget.style.transform = 'scale(1.00)'}
                  />
                  {asset.score !== undefined && (
                    <div style={{
                      position: 'absolute',
                      bottom: '8px',
                      right: '8px',
                      background: 'rgba(7, 13, 25, 0.8)',
                      padding: '2px 6px',
                      borderRadius: '4px',
                      fontSize: '0.75rem',
                      color: 'var(--accent-cyan)',
                      border: '1px solid var(--border-color)'
                    }}>
                      Score: {asset.score.toFixed(3)}
                    </div>
                  )}
                </div>

                {/* Card Content */}
                <div style={{ padding: '16px', display: 'flex', flexDirection: 'column', flex: 1, justifyContent: 'space-between', gap: '12px' }}>
                  <div>
                    <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start', gap: '8px', marginBottom: '8px' }}>
                      <h4 style={{ fontSize: '1rem', color: 'var(--text-primary)', fontWeight: '600', lineHeight: '1.3' }}>
                        {asset.title || 'Untitled'}
                      </h4>
                    </div>
                    
                    <div style={{ display: 'flex', gap: '6px', marginBottom: '10px' }}>
                      <span className="badge badge-default">{asset.category || 'Unknown'}</span>
                      <span className="badge badge-default" style={{ fontFamily: 'monospace', textTransform: 'none' }}>ID: {asset.id}</span>
                    </div>

                    <p style={{
                      color: 'var(--text-secondary)',
                      fontSize: '0.85rem',
                      lineHeight: '1.4',
                      display: '-webkit-box',
                      WebkitLineClamp: 3,
                      WebkitBoxOrient: 'vertical',
                      overflow: 'hidden',
                      textOverflow: 'ellipsis'
                    }}>
                      {asset.description || 'No description available.'}
                    </p>
                  </div>

                  <button
                    onClick={() => setSimilarForId(similarForId === asset.id ? null : asset.id)}
                    className="btn btn-secondary"
                    style={{
                      width: '100%',
                      fontSize: '0.82rem',
                      padding: '8px 12px',
                      gap: '4px',
                      backgroundColor: similarForId === asset.id ? 'rgba(0, 210, 255, 0.15)' : undefined,
                      borderColor: similarForId === asset.id ? 'var(--accent-blue)' : undefined
                    }}
                  >
                    Similar Images →
                  </button>
                </div>
              </div>
            ))}
          </div>
        )}
      </div>

      {/* Recommendations Slider / Drawer */}
      {similarForId && (
        <div className="glass-panel animate-fade" style={{
          padding: '20px',
          border: '1px dashed var(--accent-blue)',
          background: 'rgba(0, 210, 255, 0.02)'
        }}>
          <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '15px' }}>
            <h4 style={{ fontSize: '1.05rem', color: 'var(--text-primary)' }}>
              Images Similar to <span style={{ color: 'var(--accent-cyan)' }}>{similarForId}</span>
            </h4>
            <button 
              onClick={() => setSimilarForId(null)}
              className="btn btn-secondary"
              style={{ padding: '4px 10px', fontSize: '0.75rem' }}
            >
              Clear
            </button>
          </div>

          {recLoading ? (
            <div style={{ display: 'flex', alignItems: 'center', gap: '8px', color: 'var(--text-secondary)', fontSize: '0.9rem', padding: '10px 0' }}>
              <RefreshCw size={14} className="animate-spin" style={{ animation: 'spin 1.5s linear infinite' }} />
              Finding recommendations...
            </div>
          ) : recommendations.length === 0 ? (
            <p style={{ color: 'var(--text-muted)', fontSize: '0.85rem' }}>No similar images found.</p>
          ) : (
            <div style={{
              display: 'grid',
              gridTemplateColumns: 'repeat(auto-fill, minmax(200px, 1fr))',
              gap: '15px'
            }}>
              {recommendations.map((rec) => (
                <div key={rec.id} className="glass-panel" style={{
                  display: 'flex',
                  flexDirection: 'column',
                  overflow: 'hidden',
                  background: 'rgba(13, 22, 39, 0.4)',
                  padding: '6px',
                  borderRadius: '8px'
                }}>
                  <div style={{ height: '110px', overflow: 'hidden', borderRadius: '6px' }}>
                    <img 
                      src={getImageUrl(rec.url)} 
                      alt={rec.title} 
                      style={{ width: '100%', height: '100%', objectFit: 'cover' }}
                    />
                  </div>
                  <div style={{ padding: '8px 4px 4px 4px' }}>
                    <div style={{
                      fontWeight: '600',
                      fontSize: '0.85rem',
                      whiteSpace: 'nowrap',
                      overflow: 'hidden',
                      textOverflow: 'ellipsis',
                      marginBottom: '2px'
                    }}>
                      {rec.title}
                    </div>
                    <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: '0.75rem', color: 'var(--text-muted)' }}>
                      <span>{rec.category}</span>
                      {rec.score !== undefined && (
                        <span style={{ color: 'var(--accent-cyan)' }}>S: {rec.score.toFixed(2)}</span>
                      )}
                    </div>
                  </div>
                </div>
              ))}
            </div>
          )}
        </div>
      )}
    </div>
  );
};
