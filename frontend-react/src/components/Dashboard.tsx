import React, { useState, useEffect, useRef } from 'react';
import { Sparkles, RefreshCw, X, TrendingUp, Info, HelpCircle } from 'lucide-react';

interface Point2D {
  id: string;
  title: string;
  category: string;
  url: string;
  baseline: { x: number; y: number };
  adapted: { x: number; y: number };
}

interface QueryDetail {
  query: string;
  relevant_count: number;
  baseline: { ap: number; ndcg: number; first_rank: number };
  adapted: { ap: number; ndcg: number; first_rank: number };
}

interface EvalData {
  has_golden: boolean;
  queries_count: number;
  adapter_loaded: boolean;
  baseline: { map: number; ndcg: number };
  adapted: { map: number; ndcg: number };
  queries: QueryDetail[];
}

interface DashboardProps {
  apiBase: string;
  onSelectAsset: (assetId: string) => void;
  onClose: () => void;
}

const CATEGORY_COLORS: { [key: string]: string } = {
  'exterior': '#00e5ff',
  'main hut': '#39ff14',
  'artifact': '#ffff00',
  'sfm': '#ff007f',
  's3 ingested': '#a855f7',
  'uncategorized': '#94a3b8'
};

export default function Dashboard({ apiBase, onSelectAsset, onClose }: DashboardProps) {
  const [activeTab, setActiveTab] = useState<'metrics' | 'projection' | 'queries'>('metrics');
  const [projectionMode, setProjectionMode] = useState<'baseline' | 'adapted'>('baseline');
  const [loading, setLoading] = useState(true);
  const [evalData, setEvalData] = useState<EvalData | null>(null);
  const [points, setPoints] = useState<Point2D[]>([]);
  const [hoveredPoint, setHoveredPoint] = useState<Point2D | null>(null);
  const [searchQuery, setSearchQuery] = useState('');

  const canvasRef = useRef<HTMLCanvasElement | null>(null);
  const animationRef = useRef<number | null>(null);
  
  // Transition animation values
  const currentCoords = useRef<{ [key: string]: { x: number; y: number } }>({});

  useEffect(() => {
    loadDashboardData();
  }, []);

  const loadDashboardData = async () => {
    setLoading(true);
    try {
      // 1. Load evaluation metrics
      const evalResp = await fetch(`${apiBase}/api/evaluate`);
      if (evalResp.ok) {
        const evalJson = await evalResp.json();
        if (evalJson.code === 200) {
          setEvalData(evalJson.data);
        }
      }
      
      // 2. Load projections
      const projResp = await fetch(`${apiBase}/api/evaluate/projection`);
      if (projResp.ok) {
        const projJson = await projResp.json();
        if (projJson.code === 200) {
          const fetchedPoints = projJson.data.points || [];
          setPoints(fetchedPoints);
          
          // Initialize coordinator positions
          fetchedPoints.forEach((p: Point2D) => {
            currentCoords.current[p.id] = {
              x: p.baseline.x,
              y: p.baseline.y
            };
          });
        }
      }
    } catch (err) {
      console.error('Failed to load dashboard data:', err);
    } finally {
      setLoading(false);
    }
  };

  // Canvas drawing loop with animated transition between baseline and adapted spaces
  useEffect(() => {
    if (activeTab !== 'projection' || points.length === 0 || !canvasRef.current) return;
    
    const canvas = canvasRef.current;
    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    const speed = 0.08; // transition speed

    const render = () => {
      ctx.clearRect(0, 0, canvas.width, canvas.height);
      
      // Draw grid lines
      ctx.strokeStyle = 'rgba(255, 255, 255, 0.03)';
      ctx.lineWidth = 1;
      const gridSize = 40;
      for (let x = 0; x < canvas.width; x += gridSize) {
        ctx.beginPath();
        ctx.moveTo(x, 0);
        ctx.lineTo(x, canvas.height);
        ctx.stroke();
      }
      for (let y = 0; y < canvas.height; y += gridSize) {
        ctx.beginPath();
        ctx.moveTo(0, y);
        ctx.lineTo(canvas.width, y);
        ctx.stroke();
      }

      // Draw center crosshair
      ctx.strokeStyle = 'rgba(255, 255, 255, 0.07)';
      ctx.beginPath();
      ctx.moveTo(canvas.width / 2, 0);
      ctx.lineTo(canvas.width / 2, canvas.height);
      ctx.moveTo(0, canvas.height / 2);
      ctx.lineTo(canvas.width, canvas.height / 2);
      ctx.stroke();

      // Update positions & draw points
      points.forEach((p) => {
        const current = currentCoords.current[p.id] || { x: p.baseline.x, y: p.baseline.y };
        const targetX = projectionMode === 'baseline' ? p.baseline.x : p.adapted.x;
        const targetY = projectionMode === 'baseline' ? p.baseline.y : p.adapted.y;

        // Smooth transition formula
        current.x += (targetX - current.x) * speed;
        current.y += (targetY - current.y) * speed;
        currentCoords.current[p.id] = current;

        // Map projection coordinates [-100, 100] to canvas viewport coordinates
        // Pad by 40px to prevent points sitting on absolute canvas borders
        const padding = 40;
        const canvasX = padding + ((current.x + 100) / 200) * (canvas.width - padding * 2);
        const canvasY = padding + ((current.y + 100) / 200) * (canvas.height - padding * 2);

        const color = CATEGORY_COLORS[p.category.toLowerCase()] || CATEGORY_COLORS['uncategorized'];

        // Draw outer glow
        ctx.shadowBlur = 8;
        ctx.shadowColor = color;
        ctx.fillStyle = color;
        
        ctx.beginPath();
        ctx.arc(canvasX, canvasY, 4, 0, Math.PI * 2);
        ctx.fill();

        // Draw inner point
        ctx.shadowBlur = 0; // reset glow for inner
        ctx.fillStyle = '#ffffff';
        ctx.beginPath();
        ctx.arc(canvasX, canvasY, 1.5, 0, Math.PI * 2);
        ctx.fill();
      });

      animationRef.current = requestAnimationFrame(render);
    };

    render();

    return () => {
      if (animationRef.current) {
        cancelAnimationFrame(animationRef.current);
      }
    };
  }, [activeTab, points, projectionMode]);

  // Handle canvas mouse move hover checks
  const handleCanvasMouseMove = (e: React.MouseEvent<HTMLCanvasElement>) => {
    if (!canvasRef.current || points.length === 0) return;
    const canvas = canvasRef.current;
    const rect = canvas.getBoundingClientRect();
    const mouseX = e.clientX - rect.left;
    const mouseY = e.clientY - rect.top;

    let closestPoint: Point2D | null = null;
    let minDistance = 12; // hover threshold distance in pixels

    points.forEach((p) => {
      const current = currentCoords.current[p.id] || { x: 0, y: 0 };
      const padding = 40;
      const canvasX = padding + ((current.x + 100) / 200) * (canvas.width - padding * 2);
      const canvasY = padding + ((current.y + 100) / 200) * (canvas.height - padding * 2);

      const dx = mouseX - canvasX;
      const dy = mouseY - canvasY;
      const dist = Math.sqrt(dx * dx + dy * dy);

      if (dist < minDistance) {
        minDistance = dist;
        closestPoint = p;
      }
    });

    setHoveredPoint(closestPoint);
  };

  const handleCanvasClick = () => {
    if (hoveredPoint) {
      onSelectAsset(hoveredPoint.id);
    }
  };

  const getFilteredQueries = () => {
    if (!evalData) return [];
    return evalData.queries.filter(q => 
      q.query.toLowerCase().includes(searchQuery.toLowerCase())
    );
  };

  // Pre-signed URL formatting helper
  const getImageUrl = (url: string) => {
    if (url.startsWith('http')) return url;
    if (url.startsWith('/')) return `${apiBase}${url}`;
    return url;
  };

  return (
    <div style={{
      position: 'absolute',
      top: 0,
      left: 0,
      width: '100%',
      height: '100%',
      background: 'radial-gradient(circle at top right, rgba(16, 24, 48, 0.98), rgba(5, 10, 20, 0.99))',
      color: 'var(--text-primary)',
      zIndex: 100,
      display: 'flex',
      flexDirection: 'column',
      padding: '24px 30px',
      overflowY: 'auto'
    }}>
      {/* Dashboard Top Header */}
      <div style={{
        display: 'flex',
        alignItems: 'center',
        justifyContent: 'space-between',
        borderBottom: '1px solid var(--border-color)',
        paddingBottom: '16px',
        marginBottom: '24px'
      }}>
        <div style={{ display: 'flex', alignItems: 'center', gap: '12px' }}>
          <div style={{
            background: 'linear-gradient(135deg, var(--accent-cyan), var(--accent-blue))',
            borderRadius: '10px',
            padding: '8px',
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'center',
            boxShadow: '0 4px 15px rgba(0, 229, 255, 0.2)'
          }}>
            <Sparkles size={22} style={{ color: '#ffffff' }} />
          </div>
          <div>
            <h2 style={{ fontSize: '1.4rem', fontWeight: 700, margin: 0, background: 'linear-gradient(90deg, #ffffff, var(--text-secondary))', WebkitBackgroundClip: 'text', WebkitTextFillColor: 'transparent' }}>
              Deep Learning Model Evaluation Dashboard
            </h2>
            <p style={{ fontSize: '0.8rem', color: 'var(--text-muted)', marginTop: '2px' }}>
              Milestone 3 & 5: Visualizing domain gap alignment & Triplet Loss optimization on Golden Test Set
            </p>
          </div>
        </div>

        <div style={{ display: 'flex', alignItems: 'center', gap: '12px' }}>
          <button 
            onClick={loadDashboardData}
            className="btn btn-secondary"
            style={{ padding: '8px 12px', fontSize: '0.8rem', gap: '6px' }}
            disabled={loading}
          >
            <RefreshCw size={14} className={loading ? 'animate-spin' : ''} />
            Recalculate Metrics
          </button>
          <button
            onClick={onClose}
            style={{
              background: 'rgba(255, 255, 255, 0.05)',
              border: '1px solid var(--border-color)',
              color: 'var(--text-secondary)',
              borderRadius: '50%',
              width: '36px',
              height: '36px',
              display: 'flex',
              alignItems: 'center',
              justifyContent: 'center',
              cursor: 'pointer',
              transition: 'all 0.2s'
            }}
            onMouseEnter={(e) => e.currentTarget.style.background = 'rgba(255,255,255,0.1)'}
            onMouseLeave={(e) => e.currentTarget.style.background = 'rgba(255,255,255,0.05)'}
          >
            <X size={18} />
          </button>
        </div>
      </div>

      {loading ? (
        <div style={{ display: 'flex', flex: 1, flexDirection: 'column', alignItems: 'center', justifyContent: 'center', gap: '16px', minHeight: '300px' }}>
          <RefreshCw size={32} className="animate-spin" style={{ color: 'var(--accent-cyan)' }} />
          <div style={{ color: 'var(--text-muted)', fontSize: '0.9rem' }}>Compiling model weights & executing FAISS evaluation...</div>
        </div>
      ) : (
        <>
          {/* Tabs Navigation */}
          <div style={{
            display: 'flex',
            gap: '12px',
            borderBottom: '1px solid var(--border-color)',
            marginBottom: '24px',
            paddingBottom: '2px'
          }}>
            {[
              { id: 'metrics', label: 'Quantitative Metrics' },
              { id: 'projection', label: '2D Embedding Space Visualizer' },
              { id: 'queries', label: 'Golden Set Queries Breakdown' }
            ].map(tab => (
              <button
                key={tab.id}
                onClick={() => setActiveTab(tab.id as any)}
                style={{
                  background: 'transparent',
                  border: 'none',
                  color: activeTab === tab.id ? 'var(--accent-cyan)' : 'var(--text-muted)',
                  padding: '8px 16px',
                  fontSize: '0.9rem',
                  fontWeight: activeTab === tab.id ? 600 : 400,
                  cursor: 'pointer',
                  borderBottom: activeTab === tab.id ? '2px solid var(--accent-cyan)' : '2px solid transparent',
                  transition: 'all 0.2s',
                  marginBottom: '-2px'
                }}
              >
                {tab.label}
              </button>
            ))}
          </div>

          {/* TAB 1: Quantitative Metrics */}
          {activeTab === 'metrics' && (
            <div style={{ display: 'flex', flexDirection: 'column', gap: '24px' }}>
              {/* Top Stats Row */}
              <div style={{ display: 'grid', gridTemplateColumns: 'repeat(3, 1fr)', gap: '20px' }}>
                <div className="glass-panel" style={{ padding: '20px', display: 'flex', flexDirection: 'column', gap: '8px' }}>
                  <span style={{ fontSize: '0.8rem', color: 'var(--text-muted)' }}>Golden Benchmark Size</span>
                  <span style={{ fontSize: '2rem', fontWeight: 700, color: '#ffffff' }}>
                    {evalData?.queries_count || 0} Queries
                  </span>
                  <span style={{ fontSize: '0.75rem', color: 'var(--text-secondary)' }}>
                    Manually verified "strong positive" image-text pairs
                  </span>
                </div>
                
                <div className="glass-panel" style={{ padding: '20px', display: 'flex', flexDirection: 'column', gap: '8px', position: 'relative', overflow: 'hidden' }}>
                  <span style={{ fontSize: '0.8rem', color: 'var(--text-muted)' }}>MLP Projection Adapter</span>
                  <span style={{ fontSize: '2rem', fontWeight: 700, color: evalData?.adapter_loaded ? 'var(--accent-cyan)' : 'var(--text-secondary)' }}>
                    {evalData?.adapter_loaded ? 'ACTIVE' : 'NOT TRAINED'}
                  </span>
                  <span style={{ fontSize: '0.75rem', color: 'var(--text-secondary)' }}>
                    {evalData?.adapter_loaded 
                      ? 'Weights file static/models/adapter.pth loaded.' 
                      : 'Place trained adapter.pth in models folder to evaluate.'
                    }
                  </span>
                </div>

                <div className="glass-panel" style={{ padding: '20px', display: 'flex', flexDirection: 'column', gap: '8px' }}>
                  <span style={{ fontSize: '0.8rem', color: 'var(--text-muted)' }}>Mean Average Precision Delta</span>
                  {evalData && evalData.adapted.map > evalData.baseline.map ? (
                    <div style={{ display: 'flex', alignItems: 'center', gap: '6px' }}>
                      <TrendingUp size={24} style={{ color: 'var(--accent-cyan)' }} />
                      <span style={{ fontSize: '2rem', fontWeight: 700, color: 'var(--accent-cyan)' }}>
                        +{((evalData.adapted.map - evalData.baseline.map) * 100).toFixed(1)}%
                      </span>
                    </div>
                  ) : (
                    <span style={{ fontSize: '2rem', fontWeight: 700, color: 'var(--text-secondary)' }}>
                      0.0%
                    </span>
                  )}
                  <span style={{ fontSize: '0.75rem', color: 'var(--text-secondary)' }}>
                    Domain alignment projection MAP gain
                  </span>
                </div>
              </div>

              {/* Main Metrics Comparison */}
              <div style={{ display: 'grid', gridTemplateColumns: 'repeat(2, 1fr)', gap: '20px' }}>
                {/* Mean Average Precision (MAP) */}
                <div className="glass-panel" style={{ padding: '24px' }}>
                  <h4 style={{ fontSize: '1.1rem', fontWeight: 600, color: 'var(--text-primary)', marginBottom: '20px', display: 'flex', alignItems: 'center', gap: '8px' }}>
                    Mean Average Precision (MAP)
                    <span style={{ cursor: 'pointer', display: 'inline-flex' }} title="Calculates average precision across all queries, penalizing relevant items ranked lower.">
                      <HelpCircle size={16} style={{ color: 'var(--text-muted)' }} />
                    </span>
                  </h4>
                  
                  {/* Chart representation */}
                  <div style={{ display: 'flex', flexDirection: 'column', gap: '20px' }}>
                    <div>
                      <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: '0.85rem', marginBottom: '8px' }}>
                        <span style={{ color: 'var(--text-secondary)' }}>Vanilla CLIP Baseline</span>
                        <span style={{ fontWeight: 600, color: '#ffffff' }}>{(evalData?.baseline.map || 0).toFixed(4)}</span>
                      </div>
                      <div style={{ height: '12px', background: 'rgba(255,255,255,0.06)', borderRadius: '6px', overflow: 'hidden' }}>
                        <div style={{
                          height: '100%',
                          width: `${(evalData?.baseline.map || 0) * 100}%`,
                          background: 'linear-gradient(90deg, #64748b, #94a3b8)',
                          borderRadius: '6px'
                        }}></div>
                      </div>
                    </div>

                    <div>
                      <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: '0.85rem', marginBottom: '8px' }}>
                        <span style={{ color: 'var(--accent-cyan)' }}>Adapted CLIP + MLP</span>
                        <span style={{ fontWeight: 600, color: 'var(--accent-cyan)' }}>{(evalData?.adapted.map || 0).toFixed(4)}</span>
                      </div>
                      <div style={{ height: '12px', background: 'rgba(255,255,255,0.06)', borderRadius: '6px', overflow: 'hidden' }}>
                        <div style={{
                          height: '100%',
                          width: `${(evalData?.adapted.map || 0) * 100}%`,
                          background: 'linear-gradient(90deg, var(--accent-cyan), var(--accent-blue))',
                          borderRadius: '6px'
                        }}></div>
                      </div>
                    </div>
                  </div>
                </div>

                {/* nDCG@10 */}
                <div className="glass-panel" style={{ padding: '24px' }}>
                  <h4 style={{ fontSize: '1.1rem', fontWeight: 600, color: 'var(--text-primary)', marginBottom: '20px', display: 'flex', alignItems: 'center', gap: '8px' }}>
                    Normalized Discounted Cumulative Gain (nDCG@10)
                    <span style={{ cursor: 'pointer', display: 'inline-flex' }} title="Weights highly relevant search hits higher, applying a logarithmic decay factor for items ranked lower.">
                      <HelpCircle size={16} style={{ color: 'var(--text-muted)' }} />
                    </span>
                  </h4>
                  
                  <div style={{ display: 'flex', flexDirection: 'column', gap: '20px' }}>
                    <div>
                      <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: '0.85rem', marginBottom: '8px' }}>
                        <span style={{ color: 'var(--text-secondary)' }}>Vanilla CLIP Baseline</span>
                        <span style={{ fontWeight: 600, color: '#ffffff' }}>{(evalData?.baseline.ndcg || 0).toFixed(4)}</span>
                      </div>
                      <div style={{ height: '12px', background: 'rgba(255,255,255,0.06)', borderRadius: '6px', overflow: 'hidden' }}>
                        <div style={{
                          height: '100%',
                          width: `${(evalData?.baseline.ndcg || 0) * 100}%`,
                          background: 'linear-gradient(90deg, #64748b, #94a3b8)',
                          borderRadius: '6px'
                        }}></div>
                      </div>
                    </div>

                    <div>
                      <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: '0.85rem', marginBottom: '8px' }}>
                        <span style={{ color: 'var(--accent-cyan)' }}>Adapted CLIP + MLP</span>
                        <span style={{ fontWeight: 600, color: 'var(--accent-cyan)' }}>{(evalData?.adapted.ndcg || 0).toFixed(4)}</span>
                      </div>
                      <div style={{ height: '12px', background: 'rgba(255,255,255,0.06)', borderRadius: '6px', overflow: 'hidden' }}>
                        <div style={{
                          height: '100%',
                          width: `${(evalData?.adapted.ndcg || 0) * 100}%`,
                          background: 'linear-gradient(90deg, var(--accent-cyan), var(--accent-blue))',
                          borderRadius: '6px'
                        }}></div>
                      </div>
                    </div>
                  </div>
                </div>
              </div>

              {/* Informative Note */}
              <div className="glass-panel" style={{ padding: '16px 20px', display: 'flex', alignItems: 'flex-start', gap: '12px', borderColor: 'rgba(0, 229, 255, 0.15)' }}>
                <Info size={18} style={{ color: 'var(--accent-cyan)', marginTop: '2px', flexShrink: 0 }} />
                <div style={{ fontSize: '0.8rem', color: 'var(--text-secondary)', lineHeight: '1.4' }}>
                  <strong>How to improve the model:</strong> To align general CLIP text-image features with the specific historical architecture of Antarctic expedition huts and artifacts, we construct a 2-layer Multi-Layer Perceptron (MLP) adapter. The adapter is trained in a Triplet Loss framework matching anchor base photos to their verified golden captions, repelling negative pairings. See details in <code>notebooks/train_adapter.ipynb</code>.
                </div>
              </div>
            </div>
          )}

          {/* TAB 2: Embedding Space Visualizer */}
          {activeTab === 'projection' && (
            <div style={{ display: 'grid', gridTemplateColumns: '1fr 320px', gap: '24px', flex: 1, minHeight: '450px' }}>
              {/* Canvas Interactive Plot Container */}
              <div className="glass-panel" style={{
                position: 'relative',
                display: 'flex',
                alignItems: 'center',
                justifyContent: 'center',
                background: '#040812',
                overflow: 'hidden',
                borderRadius: '12px',
                height: '520px'
              }}>
                <canvas
                  ref={canvasRef}
                  width={620}
                  height={500}
                  onMouseMove={handleCanvasMouseMove}
                  onClick={handleCanvasClick}
                  style={{
                    cursor: hoveredPoint ? 'pointer' : 'default',
                    maxWidth: '100%',
                    maxHeight: '100%'
                  }}
                />

                {/* Dimension projection control bar */}
                <div style={{
                  position: 'absolute',
                  top: '16px',
                  left: '16px',
                  display: 'flex',
                  background: 'rgba(5, 10, 20, 0.85)',
                  padding: '4px',
                  borderRadius: '8px',
                  border: '1px solid var(--border-color)',
                  zIndex: 10
                }}>
                  {[
                    { id: 'baseline', label: 'Baseline CLIP Space' },
                    { id: 'adapted', label: 'MLP Projected Space' }
                  ].map(mode => (
                    <button
                      key={mode.id}
                      onClick={() => setProjectionMode(mode.id as any)}
                      style={{
                        background: projectionMode === mode.id ? 'var(--accent-blue)' : 'transparent',
                        color: projectionMode === mode.id ? '#ffffff' : 'var(--text-secondary)',
                        border: 'none',
                        borderRadius: '6px',
                        padding: '6px 12px',
                        fontSize: '0.75rem',
                        cursor: 'pointer',
                        fontWeight: projectionMode === mode.id ? 600 : 400,
                        transition: 'all 0.2s'
                      }}
                    >
                      {mode.label}
                    </button>
                  ))}
                </div>

                {/* Floating canvas legend */}
                <div style={{
                  position: 'absolute',
                  bottom: '16px',
                  left: '16px',
                  background: 'rgba(5, 10, 20, 0.85)',
                  padding: '10px 14px',
                  borderRadius: '8px',
                  border: '1px solid var(--border-color)',
                  fontSize: '0.7rem',
                  display: 'grid',
                  gridTemplateColumns: 'repeat(2, 1fr)',
                  gap: '8px',
                  zIndex: 10
                }}>
                  {Object.entries(CATEGORY_COLORS).map(([cat, col]) => (
                    <div key={cat} style={{ display: 'flex', alignItems: 'center', gap: '6px' }}>
                      <span style={{ width: '8px', height: '8px', borderRadius: '50%', backgroundColor: col }}></span>
                      <span style={{ textTransform: 'capitalize', color: 'var(--text-secondary)' }}>{cat}</span>
                    </div>
                  ))}
                </div>
              </div>

              {/* Point Detail Preview Sidebar */}
              <div style={{ display: 'flex', flexDirection: 'column', gap: '20px' }}>
                <div className="glass-panel" style={{ padding: '20px', flex: 1, display: 'flex', flexDirection: 'column', gap: '14px' }}>
                  <h4 style={{ fontSize: '0.95rem', fontWeight: 600, color: 'var(--text-primary)', margin: 0, borderBottom: '1px solid var(--border-color)', paddingBottom: '10px' }}>
                    Asset Inspector
                  </h4>
                  
                  {hoveredPoint ? (
                    <div style={{ display: 'flex', flexDirection: 'column', gap: '12px', height: '100%' }}>
                      <div style={{ height: '150px', backgroundColor: '#050a14', borderRadius: '8px', overflow: 'hidden', border: '1px solid var(--border-color)' }}>
                        <img 
                          src={getImageUrl(hoveredPoint.url)} 
                          alt={hoveredPoint.title} 
                          style={{ width: '100%', height: '100%', objectFit: 'cover' }} 
                        />
                      </div>
                      <div>
                        <div style={{ fontSize: '0.7rem', color: 'var(--text-muted)', textTransform: 'uppercase', letterSpacing: '0.05em' }}>Title</div>
                        <div style={{ fontSize: '0.85rem', fontWeight: 600, color: '#ffffff', marginTop: '2px' }}>{hoveredPoint.title}</div>
                      </div>
                      <div>
                        <div style={{ fontSize: '0.7rem', color: 'var(--text-muted)', textTransform: 'uppercase', letterSpacing: '0.05em' }}>Category</div>
                        <div style={{ display: 'flex', alignItems: 'center', gap: '6px', marginTop: '4px' }}>
                          <span style={{
                            width: '8px',
                            height: '8px',
                            borderRadius: '50%',
                            backgroundColor: CATEGORY_COLORS[hoveredPoint.category.toLowerCase()] || CATEGORY_COLORS['uncategorized']
                          }}></span>
                          <span style={{ fontSize: '0.8rem', color: 'var(--text-secondary)', textTransform: 'capitalize' }}>
                            {hoveredPoint.category}
                          </span>
                        </div>
                      </div>
                      <div style={{ marginTop: 'auto', fontSize: '0.75rem', color: 'var(--text-muted)', fontStyle: 'italic', textAlign: 'center', background: 'rgba(255,255,255,0.03)', padding: '8px', borderRadius: '6px' }}>
                        Click point to inspect full archival metadata.
                      </div>
                    </div>
                  ) : (
                    <div style={{ display: 'flex', flex: 1, flexDirection: 'column', alignItems: 'center', justifyContent: 'center', textAlign: 'center', color: 'var(--text-muted)', fontSize: '0.8rem', padding: '20px' }}>
                      <HelpCircle size={28} style={{ strokeWidth: 1.5, marginBottom: '10px' }} />
                      Hover over any dot in the 2D projection scatter plot to inspect its visual representation and path parameters.
                    </div>
                  )}
                </div>
              </div>
            </div>
          )}

          {/* TAB 3: Golden Set Queries Breakdown */}
          {activeTab === 'queries' && (
            <div className="glass-panel" style={{ padding: '20px', display: 'flex', flexDirection: 'column', gap: '16px' }}>
              {/* Search filter bar */}
              <div style={{ display: 'flex', gap: '12px' }}>
                <input
                  type="text"
                  placeholder="Search queries..."
                  value={searchQuery}
                  onChange={(e) => setSearchQuery(e.target.value)}
                  style={{
                    flex: 1,
                    background: 'rgba(5, 10, 20, 0.7)',
                    border: '1px solid var(--border-color)',
                    borderRadius: '8px',
                    padding: '10px 14px',
                    fontSize: '0.85rem',
                    color: '#ffffff',
                    outline: 'none',
                    transition: 'all 0.2s'
                  }}
                />
              </div>

              {/* Queries details table */}
              <div style={{ overflowX: 'auto' }}>
                <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: '0.85rem', textAlign: 'left' }}>
                  <thead>
                    <tr style={{ borderBottom: '2px solid var(--border-color)', color: 'var(--text-secondary)' }}>
                      <th style={{ padding: '12px 16px' }}>Query Text</th>
                      <th style={{ padding: '12px 16px', textAlign: 'center' }}>Anchor Size</th>
                      <th style={{ padding: '12px 16px', textAlign: 'center' }}>Baseline AP</th>
                      <th style={{ padding: '12px 16px', textAlign: 'center' }}>Adapted AP</th>
                      <th style={{ padding: '12px 16px', textAlign: 'center' }}>First Hit Rank Change</th>
                    </tr>
                  </thead>
                  <tbody>
                    {getFilteredQueries().map((q, idx) => {
                      const rankDiff = q.baseline.first_rank - q.adapted.first_rank;
                      const hasImprovement = q.adapted.ap > q.baseline.ap || rankDiff > 0;
                      return (
                        <tr key={idx} style={{ borderBottom: '1px solid var(--border-color)', transition: 'background 0.2s' }} className="table-row-hover">
                          <td style={{ padding: '14px 16px', fontWeight: 500, color: '#ffffff' }}>{q.query}</td>
                          <td style={{ padding: '14px 16px', textAlign: 'center', color: 'var(--text-secondary)' }}>
                            {q.relevant_count}
                          </td>
                          <td style={{ padding: '14px 16px', textAlign: 'center', color: 'var(--text-muted)' }}>
                            {q.baseline.ap.toFixed(3)}
                          </td>
                          <td style={{ padding: '14px 16px', textAlign: 'center', fontWeight: 600, color: hasImprovement ? 'var(--accent-cyan)' : 'var(--text-secondary)' }}>
                            {q.adapted.ap.toFixed(3)}
                          </td>
                          <td style={{ padding: '14px 16px', textAlign: 'center' }}>
                            {q.baseline.first_rank === -1 ? (
                              <span style={{ color: 'var(--text-muted)' }}>Not Found</span>
                            ) : (
                              <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'center', gap: '6px' }}>
                                <span style={{ color: 'var(--text-secondary)' }}>
                                  {q.baseline.first_rank} → {q.adapted.first_rank}
                                </span>
                                {rankDiff > 0 && (
                                  <span style={{
                                    fontSize: '0.7rem',
                                    color: 'var(--accent-cyan)',
                                    background: 'rgba(0, 229, 255, 0.08)',
                                    padding: '1px 5px',
                                    borderRadius: '3px',
                                    fontWeight: 600
                                  }}>
                                    +{rankDiff}
                                  </span>
                                )}
                              </div>
                            )}
                          </td>
                        </tr>
                      );
                    })}
                    {getFilteredQueries().length === 0 && (
                      <tr>
                        <td colSpan={5} style={{ padding: '30px', textAlign: 'center', color: 'var(--text-muted)', fontStyle: 'italic' }}>
                          No queries match your search query.
                        </td>
                      </tr>
                    )}
                  </tbody>
                </table>
              </div>
            </div>
          )}
        </>
      )}
    </div>
  );
}
