import React, { useState, useEffect, useRef } from 'react';
import { Send, RefreshCw, Sparkles, Loader2, ThumbsUp, ThumbsDown, ShieldCheck, Image as ImageIcon, ExternalLink, ChevronLeft, ChevronRight } from 'lucide-react';

interface ToolStep {
  tool: string;
  input: Record<string, any>;
  result_count?: number;
}

interface ChatMessage {
  role: 'user' | 'assistant';
  content: string;
  assets?: any[];
  tool_steps?: ToolStep[];
}

interface ChatAssistantProps {
  apiBase: string;
  activeQuery: string;
  sessionId: string;
  onResetSession: () => void;
  chatHistory: ChatMessage[];
  setChatHistory: React.Dispatch<React.SetStateAction<ChatMessage[]>>;
  onSelectAsset: (assetId: string) => void;
}

export const ChatAssistant: React.FC<ChatAssistantProps> = ({
  apiBase,
  activeQuery,
  sessionId,
  onResetSession,
  chatHistory,
  setChatHistory,
  onSelectAsset,
}) => {
  const [inputValue, setInputValue] = useState('');
  const [loading, setLoading] = useState(false);
  const [feedback, setFeedback] = useState<Record<number, 'up' | 'down'>>({});
  const [galleryPages, setGalleryPages] = useState<Record<number, number>>({});
  const messagesEndRef = useRef<HTMLDivElement>(null);

  // Auto-scroll to bottom on new messages
  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [chatHistory, loading]);

  // Trigger initial AI answer if history is empty and a query is active
  useEffect(() => {
    if (activeQuery && chatHistory.length === 0) {
      triggerInitialAnalysis();
    }
  }, [activeQuery]);

  const sendChatMessageSync = async (msg: string) => {
    const response = await fetch(`${apiBase}/api/agent/chat`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ message: msg, session_id: sessionId }),
    });

    const resJson = await response.json();
    if (resJson.code !== 200) {
      throw new Error(resJson.message || 'API request failed');
    }

    return resJson.data;
  };

  const sendChatMessageStream = async (msg: string, currentHistory: ChatMessage[]) => {
    const assistantIndex = currentHistory.length;
    let updatedHistory = [
      ...currentHistory,
      { role: 'assistant', content: '', assets: [], tool_steps: [] } as ChatMessage,
    ];
    setChatHistory(updatedHistory);

    try {
      const response = await fetch(`${apiBase}/api/agent/chat/stream`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ message: msg, session_id: sessionId }),
      });

      if (!response.ok || !response.body) {
        throw new Error('Streaming connection failed');
      }

      const reader = response.body.getReader();
      const decoder = new TextDecoder('utf-8');
      let buffer = '';
      let currentEvent = 'message';

      while (true) {
        const { value, done } = await reader.read();
        if (done) break;
        buffer += decoder.decode(value, { stream: true });

        const lines = buffer.split('\n');
        buffer = lines.pop() || '';

        for (const line of lines) {
          const trimmed = line.trim();
          if (!trimmed) continue;

          if (trimmed.startsWith('event:')) {
            currentEvent = trimmed.replace('event:', '').trim();
          } else if (trimmed.startsWith('data:')) {
            const rawData = trimmed.replace('data:', '').trim();
            try {
              const dataObj = JSON.parse(rawData);
              if (currentEvent === 'assets') {
                updatedHistory = updatedHistory.map((m, idx) => {
                  if (idx === assistantIndex) {
                    return {
                      ...m,
                      assets: dataObj.retrieved_assets || m.assets,
                      tool_steps: dataObj.tool_steps || m.tool_steps,
                    };
                  }
                  return m;
                });
                setChatHistory(updatedHistory);
              } else if (currentEvent === 'text') {
                const chunk = dataObj.chunk || '';
                updatedHistory = updatedHistory.map((m, idx) => {
                  if (idx === assistantIndex) {
                    return {
                      ...m,
                      content: (m.content || '') + chunk,
                    };
                  }
                  return m;
                });
                setChatHistory(updatedHistory);
              } else if (currentEvent === 'done') {
                updatedHistory = updatedHistory.map((m, idx) => {
                  if (idx === assistantIndex) {
                    return {
                      ...m,
                      content: dataObj.answer || m.content,
                      assets: dataObj.retrieved_assets || m.assets,
                      tool_steps: dataObj.tool_steps || m.tool_steps,
                    };
                  }
                  return m;
                });
                setChatHistory(updatedHistory);
              }
            } catch (err) {
              console.error('Error parsing SSE payload:', err);
            }
          }
        }
      }
    } catch (streamErr) {
      console.warn('SSE stream unavailable, falling back to sync chat API:', streamErr);
      try {
        const fallbackData = await sendChatMessageSync(msg);
        setChatHistory([
          ...currentHistory,
          {
            role: 'assistant',
            content: fallbackData.answer,
            assets: fallbackData.retrieved_assets,
            tool_steps: fallbackData.tool_steps || [],
          },
        ]);
      } catch (err) {
        setChatHistory([
          ...currentHistory,
          { role: 'assistant', content: 'Sorry, the AI service is currently unavailable. Please try again later.', assets: [] },
        ]);
      }
    }
  };

  const triggerInitialAnalysis = async () => {
    setLoading(true);
    const initialUserMsg: ChatMessage = { role: 'user', content: activeQuery };
    const baseHistory = [initialUserMsg];
    setChatHistory(baseHistory);

    try {
      await sendChatMessageStream(activeQuery, baseHistory);
    } catch (err) {
      setChatHistory([
        initialUserMsg,
        { role: 'assistant', content: 'Sorry, the AI service is currently unavailable. Please try again later.', assets: [] },
      ]);
    } finally {
      setLoading(false);
    }
  };

  const handleSend = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!inputValue.trim() || loading) return;

    const userText = inputValue.trim();
    setInputValue('');
    setLoading(true);

    const updatedHistory = [...chatHistory, { role: 'user', content: userText } as ChatMessage];
    setChatHistory(updatedHistory);

    try {
      await sendChatMessageStream(userText, updatedHistory);
    } catch (err) {
      setChatHistory([
        ...updatedHistory,
        { role: 'assistant', content: 'Sorry, the AI service is currently unavailable. Please try again later.', assets: [] },
      ]);
    } finally {
      setLoading(false);
    }
  };

  const handleFeedback = (idx: number, type: 'up' | 'down') => {
    setFeedback((prev) => ({ ...prev, [idx]: type }));
  };

  const getImageUrl = (url?: string, title?: string) => {
    let target = url || '';
    if (!target && title) {
      target = `/static/dataset/${title.replace(/\s+/g, '_')}.jpg`;
    }
    if (!target) {
      return 'data:image/svg+xml;utf8,<svg xmlns="http://www.w3.org/2000/svg" width="400" height="250" viewBox="0 0 400 250" fill="%230f172a"><rect width="400" height="250" fill="%230f172a"/><path d="M120 180 L200 80 L280 180 Z" fill="%231e293b" stroke="%2338bdf8" stroke-width="2"/><circle cx="200" cy="60" r="12" fill="%2338bdf8" opacity="0.6"/><text x="200" y="210" font-family="sans-serif" font-size="14" fill="%2394a3b8" text-anchor="middle">UKAHT Archive Image</text></svg>';
    }
    if (target.startsWith('http://') || target.startsWith('https://')) {
      return target;
    }
    if (target.startsWith('/')) {
      return `${apiBase}${target}`;
    }
    return `${apiBase}/data/${target}`;
  };

  /**
   * Deep Track Text Citation Parser:
   * Parses bracket citations like [ID: 101], [Photo: Title], [Asset 5], or inline photo titles,
   * rendering them as clickable Image Citation Chips linking directly to DetailDrawer.
   */
  const renderFormattedDeepTrackContent = (content: string, assets: any[] = []) => {
    if (!content) return null;

    const regex = /\[(?:ID:|Photo:|Asset:|Image:)?\s*([^\]]+)\]/gi;
    const parts: React.ReactNode[] = [];
    let lastIndex = 0;
    let match: RegExpExecArray | null;

    while ((match = regex.exec(content)) !== null) {
      const start = match.index;
      const end = regex.lastIndex;
      const rawMatch = match[0];
      const extractedStr = match[1].trim();

      if (start > lastIndex) {
        parts.push(content.substring(lastIndex, start));
      }

      // Match extractedStr against retrieved assets
      const matchedAsset = assets.find((a) =>
        a.id?.toLowerCase() === extractedStr.toLowerCase() ||
        a.title?.toLowerCase().includes(extractedStr.toLowerCase()) ||
        extractedStr.toLowerCase().includes(a.title?.toLowerCase() || '___nomatch')
      ) || (assets.length > 0 && !isNaN(Number(extractedStr)) && assets[Number(extractedStr) - 1] ? assets[Number(extractedStr) - 1] : null);

      const assetIdToUse = matchedAsset ? matchedAsset.id : extractedStr;
      const displayLabel = matchedAsset ? (matchedAsset.title || `Photo ID: ${matchedAsset.id}`) : rawMatch;

      parts.push(
        <span
          key={`cit_${start}`}
          onClick={() => onSelectAsset(assetIdToUse)}
          style={{
            display: 'inline-flex',
            alignItems: 'center',
            gap: '4px',
            background: 'rgba(0, 102, 204, 0.12)',
            color: 'var(--accent-blue)',
            border: '1px solid rgba(0, 102, 204, 0.3)',
            borderRadius: '6px',
            padding: '1px 8px',
            fontSize: '0.8rem',
            fontWeight: 600,
            margin: '0 3px',
            cursor: 'pointer',
            transition: 'all 0.2s ease',
            userSelect: 'none'
          }}
          onMouseEnter={(e) => {
            e.currentTarget.style.background = 'var(--accent-blue)';
            e.currentTarget.style.color = '#ffffff';
          }}
          onMouseLeave={(e) => {
            e.currentTarget.style.background = 'rgba(0, 102, 204, 0.12)';
            e.currentTarget.style.color = 'var(--accent-blue)';
          }}
          title={`Click to view high-resolution photo "${displayLabel}"`}
        >
          <ImageIcon size={12} />
          <span>{displayLabel}</span>
        </span>
      );

      lastIndex = end;
    }

    if (lastIndex < content.length) {
      parts.push(content.substring(lastIndex));
    }

    return parts;
  };

  return (
    <div className="glass-panel" style={{
      display: 'flex',
      flexDirection: 'column',
      height: '100%',
      padding: '24px',
      overflow: 'hidden'
    }}>
      {/* Chat Portal Header */}
      <div style={{
        display: 'flex',
        alignItems: 'center',
        justifyContent: 'space-between',
        borderBottom: '1px solid var(--border-color)',
        paddingBottom: '16px',
        marginBottom: '16px'
      }}>
        <div>
          <h3 style={{
            fontSize: '1.15rem',
            display: 'flex',
            alignItems: 'center',
            gap: '8px',
            color: 'var(--text-primary)',
            fontFamily: 'var(--font-family-title)',
            fontWeight: 700
          }}>
            <Sparkles size={20} style={{ color: 'var(--accent-blue)' }} />
            Fast Track Visual Search & Deep Track AI Portal
          </h3>
          <p style={{ fontSize: '0.78rem', color: 'var(--text-muted)', marginTop: '2px' }}>
            Instant vector visual recall (Fast Track) + ReAct multi-turn AI reasoning (Deep Track) | Session: <span style={{ color: 'var(--text-secondary)', fontFamily: 'monospace' }}>{sessionId}</span>
          </p>
        </div>
        <button
          onClick={onResetSession}
          className="btn btn-secondary"
          style={{
            padding: '8px 14px',
            fontSize: '0.8rem',
            gap: '6px'
          }}
          title="Reset conversation history"
        >
          <RefreshCw size={14} />
          Reset Session
        </button>
      </div>

      {/* Messages Scroll Container */}
      <div style={{
        flex: 1,
        overflowY: 'auto',
        marginBottom: '16px',
        paddingRight: '6px',
        display: 'flex',
        flexDirection: 'column',
        gap: '20px'
      }}>
        {chatHistory.length === 0 && !loading && (
          <div style={{
            textAlign: 'center',
            color: 'var(--text-muted)',
            padding: '60px 20px',
            fontSize: '0.95rem'
          }}>
            <Sparkles size={32} style={{ color: 'var(--accent-blue)', margin: '0 auto 12px auto', opacity: 0.7 }} />
            Ask a question or enter a search query to explore expedition photos and historical insights.
          </div>
        )}

        {chatHistory.map((msg, idx) => {
          const currentPage = galleryPages[idx] || 1;
          const pageSize = 6;
          const totalAssetsCount = msg.assets ? msg.assets.length : 0;
          const totalPages = Math.max(1, Math.ceil(totalAssetsCount / pageSize));
          const startIndex = (currentPage - 1) * pageSize;
          const endIndex = Math.min(startIndex + pageSize, totalAssetsCount);
          const pageAssets = msg.assets ? msg.assets.slice(startIndex, endIndex) : [];

          return (
            <div
              key={idx}
              style={{
                alignSelf: msg.role === 'user' ? 'flex-end' : 'flex-start',
                width: msg.role === 'user' ? 'auto' : '100%',
                maxWidth: msg.role === 'user' ? '80%' : '100%',
                display: 'flex',
                flexDirection: 'column',
                gap: '6px'
              }}
            >
              {/* Sender Label */}
              <div style={{
                fontSize: '0.75rem',
                color: 'var(--text-muted)',
                fontWeight: 600,
                alignSelf: msg.role === 'user' ? 'flex-end' : 'flex-start',
                padding: '0 4px',
                display: 'flex',
                alignItems: 'center',
                gap: '6px'
              }}>
                {msg.role === 'user' ? 'You' : '⚡ Smart Search Assistant'}
              </div>

              {/* User Message Bubble */}
              {msg.role === 'user' && (
                <div style={{
                  background: 'linear-gradient(135deg, rgba(0, 102, 204, 0.1), rgba(0, 102, 204, 0.18))',
                  border: '1px solid rgba(0, 102, 204, 0.3)',
                  color: 'var(--text-primary)',
                  padding: '12px 18px',
                  borderRadius: '16px 16px 2px 16px',
                  fontSize: '0.95rem',
                  lineHeight: '1.5',
                  boxShadow: '0 2px 8px rgba(0,0,0,0.04)'
                }}>
                  {msg.content}
                </div>
              )}

              {/* Assistant Message Layout: FAST TRACK (Gallery with Pagination) ON TOP, DEEP TRACK BELOW */}
              {msg.role === 'assistant' && (
                <div style={{
                  display: 'flex',
                  flexDirection: 'column',
                  gap: '16px',
                  width: '100%',
                  background: '#ffffff',
                  border: '1px solid var(--border-color)',
                  borderRadius: '16px',
                  padding: '20px',
                  boxShadow: '0 4px 20px rgba(0, 0, 0, 0.04)'
                }}>

                  {/* =========================================================================
                     1. FAST TRACK (RANKED VISUAL GALLERY WITH PAGINATION) - POSITIONED ON TOP
                     ========================================================================= */}
                  {msg.assets && msg.assets.length > 0 && (
                    <div style={{ display: 'flex', flexDirection: 'column', gap: '12px' }}>
                      <div style={{
                        fontSize: '0.85rem',
                        fontWeight: 700,
                        color: 'var(--text-secondary)',
                        display: 'flex',
                        alignItems: 'center',
                        justifyContent: 'space-between',
                        borderBottom: '1px solid var(--border-color)',
                        paddingBottom: '8px'
                      }}>
                        <div style={{ display: 'flex', alignItems: 'center', gap: '6px' }}>
                          <ImageIcon size={18} style={{ color: 'var(--accent-blue)' }} />
                          <span style={{ fontSize: '0.92rem', color: 'var(--text-primary)', fontWeight: 700 }}>
                            Fast Track Ranked Visual Gallery (Showing {totalAssetsCount === 0 ? 0 : startIndex + 1}-{endIndex} of {totalAssetsCount} Ranked Photos)
                          </span>
                        </div>
                        <span style={{ fontSize: '0.72rem', color: 'var(--text-muted)', fontWeight: 500 }}>
                          Sorted by Full Similarity Ranking Algorithm
                        </span>
                      </div>

                      {/* Photo Grid with Global Rank Badges (#1, #2, #3...) */}
                      <div style={{
                        display: 'grid',
                        gridTemplateColumns: 'repeat(auto-fill, minmax(210px, 1fr))',
                        gap: '14px',
                        width: '100%'
                      }}>
                        {pageAssets.map((asset) => {
                          return (
                            <div key={asset.id} style={{ position: 'relative' }}>
                              {/* Card Stacking Layers */}
                              {asset.stacked_assets && asset.stacked_assets.length > 0 && (
                                <>
                                  <div style={{
                                    position: 'absolute',
                                    top: '6px',
                                    left: '6px',
                                    right: '-6px',
                                    bottom: '-6px',
                                    background: 'rgba(5, 10, 20, 0.15)',
                                    borderRadius: '12px',
                                    zIndex: 1
                                  }} />
                                  <div style={{
                                    position: 'absolute',
                                    top: '3px',
                                    left: '3px',
                                    right: '-3px',
                                    bottom: '-3px',
                                    background: 'rgba(10, 18, 36, 0.25)',
                                    borderRadius: '12px',
                                    zIndex: 2
                                  }} />
                                </>
                              )}

                              {/* Main Image Card */}
                              <div
                                onClick={() => onSelectAsset(asset.id)}
                                style={{
                                  zIndex: 3,
                                  position: 'relative',
                                  borderRadius: '12px',
                                  background: '#ffffff',
                                  overflow: 'hidden',
                                  cursor: 'pointer',
                                  boxShadow: '0 2px 10px rgba(0, 0, 0, 0.05)',
                                  transition: 'all 0.25s cubic-bezier(0.16, 1, 0.3, 1)',
                                  border: '1px solid var(--border-color)',
                                  display: 'flex',
                                  flexDirection: 'column'
                                }}
                                onMouseEnter={(e) => {
                                  e.currentTarget.style.borderColor = 'var(--accent-blue)';
                                  e.currentTarget.style.transform = 'translateY(-3px) scale(1.02)';
                                  e.currentTarget.style.boxShadow = '0 8px 24px rgba(0, 102, 204, 0.18)';
                                }}
                                onMouseLeave={(e) => {
                                  e.currentTarget.style.borderColor = 'var(--border-color)';
                                  e.currentTarget.style.transform = 'translateY(0) scale(1)';
                                  e.currentTarget.style.boxShadow = '0 2px 10px rgba(0, 0, 0, 0.05)';
                                }}
                              >
                                {/* Clean Image Frame without cluttering overlay badges */}
                                <div style={{ height: '140px', backgroundColor: '#0f172a', overflow: 'hidden', position: 'relative' }}>
                                  <img
                                    src={getImageUrl(asset.url, asset.title)}
                                    alt={asset.title}
                                    onError={(e) => {
                                      const img = e.target as HTMLImageElement;
                                      const currentSrc = img.src || '';
                                      if (currentSrc && currentSrc.startsWith('http') && !currentSrc.includes('/api/image-proxy')) {
                                        img.src = `${apiBase}/api/image-proxy?url=${encodeURIComponent(currentSrc)}`;
                                      } else {
                                        img.src = 'data:image/svg+xml;utf8,<svg xmlns="http://www.w3.org/2000/svg" width="400" height="250" viewBox="0 0 400 250" fill="%230f172a"><rect width="400" height="250" fill="%230f172a"/><path d="M120 180 L200 80 L280 180 Z" fill="%231e293b" stroke="%2338bdf8" stroke-width="2"/><circle cx="200" cy="60" r="12" fill="%2338bdf8" opacity="0.6"/><text x="200" y="210" font-family="sans-serif" font-size="14" fill="%2394a3b8" text-anchor="middle">UKAHT Archive Image</text></svg>';
                                      }
                                    }}
                                    style={{
                                      width: '100%',
                                      height: '100%',
                                      objectFit: 'cover',
                                      transition: 'transform 0.3s ease'
                                    }}
                                  />
                                </div>

                                {/* Footer Info */}
                                <div style={{ padding: '10px 12px', display: 'flex', flexDirection: 'column', gap: '4px' }}>
                                  <div style={{
                                    fontWeight: 700,
                                    fontSize: '0.82rem',
                                    whiteSpace: 'nowrap',
                                    overflow: 'hidden',
                                    textOverflow: 'ellipsis',
                                    color: 'var(--text-primary)'
                                  }}>
                                    {asset.title || 'Untitled Asset'}
                                  </div>
                                  <div style={{
                                    display: 'flex',
                                    alignItems: 'center',
                                    justifyContent: 'space-between',
                                    fontSize: '0.7rem',
                                    color: 'var(--text-muted)'
                                  }}>
                                    <span>ID: {asset.id}</span>
                                    {asset.base_code && (
                                      <span style={{
                                        background: 'rgba(0, 102, 204, 0.08)',
                                        color: 'var(--accent-blue)',
                                        padding: '1px 6px',
                                        borderRadius: '4px',
                                        fontWeight: 600
                                      }}>
                                        Base {asset.base_code}
                                      </span>
                                    )}
                                  </div>
                                </div>
                              </div>
                            </div>
                          );
                        })}
                      </div>

                      {/* Previous Page (上一页) & Next Page (下一页) Pagination Bar */}
                      {totalPages > 1 && (
                        <div style={{
                          display: 'flex',
                          alignItems: 'center',
                          justifyContent: 'center',
                          gap: '16px',
                          marginTop: '8px',
                          paddingTop: '12px',
                          borderTop: '1px dashed var(--border-color)'
                        }}>
                          <button
                            onClick={() => setGalleryPages(prev => ({ ...prev, [idx]: Math.max(1, currentPage - 1) }))}
                            disabled={currentPage <= 1}
                            className="btn btn-secondary"
                            style={{
                              padding: '6px 16px',
                              fontSize: '0.8rem',
                              borderRadius: '20px',
                              display: 'flex',
                              alignItems: 'center',
                              gap: '6px',
                              opacity: currentPage <= 1 ? 0.4 : 1,
                              cursor: currentPage <= 1 ? 'not-allowed' : 'pointer',
                              borderColor: 'var(--accent-blue)',
                              color: currentPage <= 1 ? 'var(--text-muted)' : 'var(--accent-blue)'
                            }}
                          >
                            <ChevronLeft size={16} />
                            <span>上一页 (Previous)</span>
                          </button>

                          <span style={{ fontSize: '0.85rem', fontWeight: 600, color: 'var(--text-secondary)' }}>
                            Page <strong style={{ color: 'var(--accent-blue)' }}>{currentPage}</strong> of {totalPages}
                          </span>

                          <button
                            onClick={() => setGalleryPages(prev => ({ ...prev, [idx]: Math.min(totalPages, currentPage + 1) }))}
                            disabled={currentPage >= totalPages}
                            className="btn btn-secondary"
                            style={{
                              padding: '6px 16px',
                              fontSize: '0.8rem',
                              borderRadius: '20px',
                              display: 'flex',
                              alignItems: 'center',
                              gap: '6px',
                              opacity: currentPage >= totalPages ? 0.4 : 1,
                              cursor: currentPage >= totalPages ? 'not-allowed' : 'pointer',
                              borderColor: 'var(--accent-blue)',
                              color: currentPage >= totalPages ? 'var(--text-muted)' : 'var(--accent-blue)'
                            }}
                          >
                            <span>下一页 (Next)</span>
                            <ChevronRight size={16} />
                          </button>
                        </div>
                      )}
                    </div>
                  )}

                  {/* =========================================================================
                     2. DEEP TRACK (AI EXECUTIVE SUMMARY & REASONING) - POSITIONED BELOW FAST TRACK
                     ========================================================================= */}
                  <div style={{
                    display: 'flex',
                    flexDirection: 'column',
                    gap: '12px',
                    background: 'linear-gradient(180deg, rgba(0, 102, 204, 0.03) 0%, rgba(248, 250, 252, 0.5) 100%)',
                    border: '1px solid rgba(0, 102, 204, 0.15)',
                    borderRadius: '12px',
                    padding: '16px 18px'
                  }}>
                    {/* Deep Track Executive Header */}
                    <div style={{
                      display: 'flex',
                      alignItems: 'center',
                      justifyContent: 'space-between',
                      borderBottom: '1px solid rgba(0, 102, 204, 0.1)',
                      paddingBottom: '10px'
                    }}>
                      <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
                        <span style={{
                          background: 'var(--accent-blue)',
                          color: '#ffffff',
                          padding: '3px 10px',
                          borderRadius: '20px',
                          fontSize: '0.72rem',
                          fontWeight: 700,
                          display: 'flex',
                          alignItems: 'center',
                          gap: '4px'
                        }}>
                          <Sparkles size={12} />
                          Deep Track AI Executive Summary
                        </span>
                        {msg.assets && msg.assets.length > 0 && (
                          <span style={{
                            background: 'rgba(16, 185, 129, 0.1)',
                            color: '#10b981',
                            border: '1px solid rgba(16, 185, 129, 0.25)',
                            padding: '2px 8px',
                            borderRadius: '12px',
                            fontSize: '0.7rem',
                            fontWeight: 600,
                            display: 'flex',
                            alignItems: 'center',
                            gap: '4px'
                          }}>
                            <ShieldCheck size={12} />
                            Verified Grounding ({msg.assets.length} Archive Photos)
                          </span>
                        )}
                      </div>

                      {/* Feedback Loop Buttons (Thumbs Up / Down) */}
                      <div style={{ display: 'flex', alignItems: 'center', gap: '6px' }}>
                        <button
                          onClick={() => handleFeedback(idx, 'up')}
                          style={{
                            background: feedback[idx] === 'up' ? 'rgba(16, 185, 129, 0.15)' : 'transparent',
                            border: feedback[idx] === 'up' ? '1px solid #10b981' : '1px solid var(--border-color)',
                            color: feedback[idx] === 'up' ? '#10b981' : 'var(--text-muted)',
                            borderRadius: '6px',
                            padding: '4px 8px',
                            cursor: 'pointer',
                            display: 'flex',
                            alignItems: 'center',
                            gap: '4px',
                            fontSize: '0.72rem',
                            transition: 'all 0.2s'
                          }}
                          title="Helpful analysis"
                        >
                          <ThumbsUp size={12} />
                          {feedback[idx] === 'up' && <span>Helpful</span>}
                        </button>
                        <button
                          onClick={() => handleFeedback(idx, 'down')}
                          style={{
                            background: feedback[idx] === 'down' ? 'rgba(239, 68, 68, 0.15)' : 'transparent',
                            border: feedback[idx] === 'down' ? '1px solid #ef4444' : '1px solid var(--border-color)',
                            color: feedback[idx] === 'down' ? '#ef4444' : 'var(--text-muted)',
                            borderRadius: '6px',
                            padding: '4px 8px',
                            cursor: 'pointer',
                            display: 'flex',
                            alignItems: 'center',
                            gap: '4px',
                            fontSize: '0.72rem',
                            transition: 'all 0.2s'
                          }}
                          title="Needs improvement"
                        >
                          <ThumbsDown size={12} />
                        </button>
                      </div>
                    </div>

                    {/* Deep Track Natural Language Text Content with Clickable Image Citations */}
                    <div style={{
                      fontSize: '0.94rem',
                      lineHeight: '1.6',
                      color: 'var(--text-primary)',
                      whiteSpace: 'pre-wrap'
                    }}>
                      {msg.content ? (
                        renderFormattedDeepTrackContent(msg.content, msg.assets)
                      ) : (
                        <div style={{ display: 'flex', alignItems: 'center', gap: '8px', color: 'var(--text-muted)', padding: '6px 0' }}>
                          <Loader2 size={16} className="animate-spin" style={{ animation: 'spin 1s linear infinite', color: 'var(--accent-blue)' }} />
                          <span>Deep Track AI is analyzing historical context and generating reasoning...</span>
                        </div>
                      )}
                    </div>

                    {/* Clickable Image Evidence Chips Bar (Photo-based Citation Rail - Limited to top 6 grounded items) */}
                    {msg.assets && msg.assets.length > 0 && (
                      <div style={{
                        marginTop: '6px',
                        paddingTop: '10px',
                        borderTop: '1px dashed rgba(0, 102, 204, 0.15)',
                        display: 'flex',
                        alignItems: 'center',
                        flexWrap: 'wrap',
                        gap: '8px'
                      }}>
                        <span style={{ fontSize: '0.75rem', fontWeight: 700, color: 'var(--text-secondary)', display: 'flex', alignItems: 'center', gap: '4px' }}>
                          <ImageIcon size={14} style={{ color: 'var(--accent-blue)' }} />
                          Grounded Photo Citations:
                        </span>
                        {msg.assets.slice(0, 6).map((asset) => (
                          <button
                            key={asset.id}
                            onClick={() => onSelectAsset(asset.id)}
                            style={{
                              display: 'inline-flex',
                              alignItems: 'center',
                              gap: '6px',
                              background: '#ffffff',
                              border: '1px solid rgba(0, 102, 204, 0.25)',
                              borderRadius: '20px',
                              padding: '3px 10px',
                              fontSize: '0.75rem',
                              fontWeight: 600,
                              color: 'var(--accent-blue)',
                              cursor: 'pointer',
                              boxShadow: '0 2px 4px rgba(0, 102, 204, 0.05)',
                              transition: 'all 0.2s'
                            }}
                            onMouseEnter={(e) => {
                              e.currentTarget.style.borderColor = 'var(--accent-blue)';
                              e.currentTarget.style.transform = 'translateY(-1px)';
                              e.currentTarget.style.boxShadow = '0 4px 8px rgba(0, 102, 204, 0.15)';
                            }}
                            onMouseLeave={(e) => {
                              e.currentTarget.style.borderColor = 'rgba(0, 102, 204, 0.25)';
                              e.currentTarget.style.transform = 'translateY(0)';
                              e.currentTarget.style.boxShadow = '0 2px 4px rgba(0, 102, 204, 0.05)';
                            }}
                            title={`View photo details for ${asset.title}`}
                          >
                            <img
                              src={getImageUrl(asset.url, asset.title)}
                              alt={asset.title}
                              style={{ width: '18px', height: '18px', objectFit: 'cover', borderRadius: '50%' }}
                            />
                            <span>{asset.title || `Photo ${asset.id}`}</span>
                            <ExternalLink size={10} style={{ opacity: 0.7 }} />
                          </button>
                        ))}
                      </div>
                    )}
                  </div>

                  {/* =========================================================================
                     3. TOOL EXECUTION ROUTING RAIL (AGENT PROVENANCE LOGS)
                     ========================================================================= */}
                  {msg.tool_steps && msg.tool_steps.length > 0 && (
                    <div style={{
                      display: 'flex',
                      alignItems: 'center',
                      flexWrap: 'wrap',
                      gap: '6px',
                      paddingTop: '8px',
                      borderTop: '1px solid var(--border-color)',
                      fontSize: '0.72rem',
                      color: 'var(--text-muted)'
                    }}>
                      <span style={{ fontWeight: 600, color: 'var(--text-secondary)' }}>Agent Route Execution:</span>
                      {msg.tool_steps.map((step, stepIdx) => {
                        const toolLabel = step.tool === 'semantic_search' ? 'Semantic Search' : step.tool === 'sql_filter' ? 'SQL Filter' : step.tool;
                        const inputSummary = Object.entries(step.input || {})
                          .filter(([, v]) => v !== null && v !== undefined && v !== '')
                          .map(([k, v]) => `${k}: "${v}"`)
                          .join(', ');
                        return (
                          <div key={stepIdx} style={{
                            display: 'inline-flex',
                            alignItems: 'center',
                            gap: '6px',
                            background: 'rgba(99, 102, 241, 0.06)',
                            border: '1px solid rgba(99, 102, 241, 0.2)',
                            borderRadius: '6px',
                            padding: '3px 8px',
                            color: '#6366f1',
                          }}>
                            <span>⚙</span>
                            <span style={{ fontWeight: 700 }}>{toolLabel}</span>
                            {inputSummary && <span style={{ color: '#94a3b8', fontWeight: 400 }}>({inputSummary})</span>}
                            {step.result_count !== undefined && (
                              <span style={{
                                background: '#6366f1',
                                color: '#fff',
                                borderRadius: '10px',
                                padding: '0px 6px',
                                fontSize: '0.65rem',
                                fontWeight: 700
                              }}>{step.result_count} items</span>
                            )}
                          </div>
                        );
                      })}
                    </div>
                  )}

                </div>
              )}

            </div>
          );
        })}

        {loading && (
          <div style={{
            alignSelf: 'flex-start',
            display: 'flex',
            alignItems: 'center',
            gap: '10px',
            color: 'var(--text-secondary)',
            fontSize: '0.88rem',
            padding: '12px 18px',
            background: '#ffffff',
            border: '1px solid var(--border-color)',
            borderRadius: '12px',
            boxShadow: '0 2px 8px rgba(0,0,0,0.04)'
          }}>
            <Loader2 size={16} className="animate-spin" style={{ animation: 'spin 1s linear infinite', color: 'var(--accent-blue)' }} />
            <span>Deep Track AI is routing query & analyzing archive evidence...</span>
          </div>
        )}
        <div ref={messagesEndRef} />
      </div>

      {/* Interactive Input Form */}
      <form onSubmit={handleSend} style={{ display: 'flex', gap: '10px', position: 'relative' }}>
        <input
          type="text"
          value={inputValue}
          onChange={(e) => setInputValue(e.target.value)}
          placeholder="Ask a follow-up question or refine search parameters..."
          disabled={loading}
          style={{
            flex: 1,
            padding: '12px 16px',
            borderRadius: '10px',
            border: '1px solid var(--border-color)',
            fontSize: '0.95rem'
          }}
        />
        <button
          type="submit"
          className="btn btn-primary"
          disabled={!inputValue.trim() || loading}
          style={{ padding: '12px 20px', borderRadius: '10px', display: 'flex', alignItems: 'center', gap: '6px' }}
        >
          <Send size={16} />
          <span>Send</span>
        </button>
      </form>
    </div>
  );
};
