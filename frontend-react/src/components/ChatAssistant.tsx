import React, { useState, useEffect, useRef } from 'react';
import { Send, RefreshCw, Sparkles, Loader2 } from 'lucide-react';

interface ChatMessage {
  role: 'user' | 'assistant';
  content: string;
  assets?: any[];
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

  const triggerInitialAnalysis = async () => {
    setLoading(true);
    // Add temporary user message
    const initialUserMsg: ChatMessage = { role: 'user', content: activeQuery };
    setChatHistory([initialUserMsg]);

    try {
      const response = await sendChatMessage(activeQuery);
      setChatHistory([
        initialUserMsg,
        {
          role: 'assistant',
          content: response.answer,
          assets: response.retrieved_assets,
        },
      ]);
    } catch (err) {
      // Fallback response in case of overall api error
      const mockResp = getDemoReply(activeQuery);
      setChatHistory([
        initialUserMsg,
        {
          role: 'assistant',
          content: mockResp.answer,
          assets: mockResp.retrieved_assets,
        },
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
      const response = await sendChatMessage(userText);
      setChatHistory([
        ...updatedHistory,
        {
          role: 'assistant',
          content: response.answer,
          assets: response.retrieved_assets,
        },
      ]);
    } catch (err) {
      const mockResp = getDemoReply(userText);
      setChatHistory([
        ...updatedHistory,
        {
          role: 'assistant',
          content: mockResp.answer,
          assets: mockResp.retrieved_assets,
        },
      ]);
    } finally {
      setLoading(false);
    }
  };

  // Helper to send API request to backend chat agent
  const sendChatMessage = async (msg: string) => {
    const response = await fetch(`${apiBase}/api/agent/chat`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ message: msg, session_id: sessionId }),
    });

    const resJson = await response.json();
    if (resJson.code !== 200) {
      throw new Error(resJson.message || 'API request failed');
    }

    const chatData = resJson.data;
    
    // If backend reports LLM is unavailable, trigger fallback mock
    if (chatData.answer && chatData.answer.includes('[LLM unavailable')) {
      return getDemoReply(msg);
    }

    return chatData;
  };

  // Demo fallback reply (matches the backend/streamlit implementation)
  const getDemoReply = (message: string) => {
    const msgLower = message.toLowerCase();
    let answer = '';
    
    if (msgLower.includes('dog') || msgLower.includes('sledge')) {
      answer = `🐕 **[Demo Mode - Simulated Response]**\n\nDogs (specifically huskies) were indispensable partners in British Antarctic expeditions. They were primarily used for pulling sledges loaded with geological equipment and food rations across crevassed terrain where vehicles could not go. The photos show sledge dogs resting near the tents at Base E (Stonington Island) and active dog teams during field surveys.`;
    } else if (msgLower.includes('people') || msgLower.includes('man') || msgLower.includes('crew') || msgLower.includes('person')) {
      answer = `👥 **[Demo Mode - Simulated Response]**\n\nI found several images depicting expedition personnel. These include scientists, station leaders, and radio operators engaged in daily station maintenance, meteorological observations, and recreational activities during the long polar winters.`;
    } else if (msgLower.includes('building') || msgLower.includes('station') || msgLower.includes('base') || msgLower.includes('hut')) {
      answer = `🏠 **[Demo Mode - Simulated Response]**\n\nThe archive contains historical views of early British scientific bases, such as Port Lockroy (Base A) and Horseshoe Island (Base Y). These structures were built with timber panels and served as living quarters, laboratories, and radio stations.`;
    } else {
      answer = `❄️ **[Demo Mode - Simulated Response]**\n\nRegarding your query about **"${message}"**:\nI searched the archive and found matching historical visual resources. The collection documents early mid-20th century expedition life, scientific instrumentation, and polar geography under the Falkland Islands Dependencies Survey (FIDS).\n\n*Note: The Ollama LLM backend is currently offline or unreachable. This is a pre-configured demo reply.*`;
    }

    return {
      answer,
      retrieved_assets: [], // empty list or fetch assets if needed, but keeping simple
      session_id: sessionId
    };
  };

  const getImageUrl = (url: string) => {
    if (url.startsWith('http')) return url;
    if (url.startsWith('/')) return `${apiBase}${url}`;
    return url;
  };

  return (
    <div className="glass-panel" style={{
      display: 'flex',
      flexDirection: 'column',
      height: '100%',
      padding: '24px',
      overflow: 'hidden'
    }}>
      {/* Chat Header */}
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
            fontSize: '1.1rem',
            display: 'flex',
            alignItems: 'center',
            gap: '8px',
            color: 'var(--text-primary)'
          }}>
            <Sparkles size={18} style={{ color: 'var(--accent-blue)' }} />
            AI Assistant Analysis
          </h3>
          <p style={{ fontSize: '0.75rem', color: 'var(--text-muted)', marginTop: '2px' }}>
            Session Context: <span style={{ color: 'var(--text-secondary)' }}>{sessionId}</span>
          </p>
        </div>
        <button
          onClick={onResetSession}
          className="btn btn-secondary"
          style={{
            padding: '8px 12px',
            fontSize: '0.8rem',
            gap: '6px'
          }}
          title="Reset conversational history"
        >
          <RefreshCw size={14} />
          Reset Chat
        </button>
      </div>

      {/* Messages Container */}
      <div style={{
        flex: 1,
        overflowY: 'auto',
        marginBottom: '16px',
        paddingRight: '4px',
        display: 'flex',
        flexDirection: 'column',
        gap: '16px'
      }}>
        {chatHistory.length === 0 && !loading && (
          <div style={{
            textAlign: 'center',
            color: 'var(--text-muted)',
            padding: '40px 20px',
            fontSize: '0.9rem'
          }}>
            Enter a search query or ask a question to start the conversation.
          </div>
        )}

        {chatHistory.map((msg, idx) => (
          <div
            key={idx}
            style={{
              alignSelf: msg.role === 'user' ? 'flex-end' : 'flex-start',
              maxWidth: '85%',
              display: 'flex',
              flexDirection: 'column',
              gap: '4px'
            }}
          >
            <div style={{
              fontSize: '0.75rem',
              color: 'var(--text-muted)',
              alignSelf: msg.role === 'user' ? 'flex-end' : 'flex-start',
              padding: '0 4px'
            }}>
              {msg.role === 'user' ? 'You' : 'Assistant'}
            </div>
            
            <div style={{
              background: msg.role === 'user' ? 'linear-gradient(135deg, rgba(0, 102, 204, 0.08), rgba(0, 102, 204, 0.12))' : '#f1f5f9',
              border: msg.role === 'user' ? '1px solid rgba(0, 102, 204, 0.2)' : '1px solid var(--border-color)',
              color: 'var(--text-primary)',
              padding: '14px 18px',
              borderRadius: msg.role === 'user' ? '12px 12px 2px 12px' : '12px 12px 12px 2px',
              fontSize: '0.92rem',
              lineHeight: '1.5',
              whiteSpace: 'pre-wrap'
            }}>
              {/* Parse basic markdown style manually or display raw */}
              {msg.content}
            </div>

            {/* Inline Assets from Assistant */}
            {msg.role === 'assistant' && msg.assets && msg.assets.length > 0 && (
              <div style={{
                display: 'grid',
                gridTemplateColumns: `repeat(${Math.min(msg.assets.length, 3)}, 1fr)`,
                gap: '8px',
                marginTop: '8px'
              }}>
                {msg.assets.slice(0, 3).map((asset) => (
                  <div 
                    key={asset.id} 
                    className="glass-panel"
                    onClick={() => onSelectAsset(asset.id)}
                    style={{
                      padding: '4px',
                      borderRadius: '6px',
                      background: '#f1f5f9',
                      fontSize: '0.75rem',
                      cursor: 'pointer'
                    }}
                  >
                    <img 
                      src={getImageUrl(asset.url)} 
                      alt={asset.title} 
                      style={{
                        width: '100%',
                        height: '60px',
                        objectFit: 'cover',
                        borderRadius: '4px',
                        marginBottom: '4px'
                      }}
                    />
                    <div style={{
                      whiteSpace: 'nowrap',
                      overflow: 'hidden',
                      textOverflow: 'ellipsis',
                      color: 'var(--text-secondary)',
                      padding: '0 2px'
                    }}>
                      {asset.title}
                    </div>
                  </div>
                ))}
              </div>
            )}
          </div>
        ))}

        {loading && (
          <div style={{
            alignSelf: 'flex-start',
            display: 'flex',
            alignItems: 'center',
            gap: '8px',
            color: 'var(--text-secondary)',
            fontSize: '0.85rem',
            padding: '10px 16px',
            background: '#f1f5f9',
            border: '1px solid var(--border-color)',
            borderRadius: '12px'
          }}>
            <Loader2 size={14} className="animate-spin" style={{ animation: 'spin 1s linear infinite' }} />
            AI is analyzing...
          </div>
        )}
        <div ref={messagesEndRef} />
      </div>

      {/* Input Box */}
      <form onSubmit={handleSend} style={{ display: 'flex', gap: '10px' }}>
        <input
          type="text"
          value={inputValue}
          onChange={(e) => setInputValue(e.target.value)}
          placeholder="Ask a follow-up question..."
          disabled={loading}
          style={{ flex: 1 }}
        />
        <button
          type="submit"
          className="btn btn-primary"
          disabled={!inputValue.trim() || loading}
          style={{ padding: '12px 16px' }}
        >
          <Send size={16} />
        </button>
      </form>
    </div>
  );
};
