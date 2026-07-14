import { useState, useEffect } from 'react';
import { Sidebar } from './components/Sidebar';
import { SearchPortal } from './components/SearchPortal';
import { ChatAssistant } from './components/ChatAssistant';
import { Annotator } from './components/Annotator';

interface Asset {
  id: string;
  url: string;
  title: string;
  category: string;
  description: string;
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
  const [activeTab, setActiveTab] = useState<'search' | 'annotate'>('search');
  
  // Search & Results State
  const [activeQuery, setActiveQuery] = useState('');
  const [activeCategory, setActiveCategory] = useState('');
  const [searchResults, setSearchResults] = useState<Asset[]>([]);
  const [searchLoading, setSearchLoading] = useState(false);
  const [similarForId, setSimilarForId] = useState<string | null>(null);
  
  // Chat Assistant State
  const [chatSessionId, setChatSessionId] = useState(generateSessionId());
  const [chatHistory, setChatHistory] = useState<ChatMessage[]>([]);

  // Initial load: Fetch all assets to populate the gallery
  useEffect(() => {
    fetchSearchResults('', '');
  }, []);

  const fetchSearchResults = async (query: string, category: string) => {
    setSearchLoading(true);
    try {
      const payload: any = { query };
      if (category) {
        payload.category = category;
      }
      
      const response = await fetch(`${API_BASE}/api/search`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(payload)
      });
      
      if (response.ok) {
        const resJson = await response.json();
        if (resJson.code === 200) {
          setSearchResults(resJson.data);
        } else {
          console.error('API Error:', resJson.message);
        }
      }
    } catch (err) {
      console.error('Error searching assets:', err);
    } finally {
      setSearchLoading(false);
    }
  };

  const handleSearchSubmit = (query: string, category: string) => {
    setActiveQuery(query);
    setActiveCategory(category);
    setSimilarForId(null);
    fetchSearchResults(query, category);

    // If query has changed, reset conversational session
    if (query !== activeQuery) {
      resetChatSession();
    }
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



  // Determine if split screen is active (when query is set)
  const isSplitView = activeTab === 'search' && activeQuery.trim() !== '';

  return (
    <div style={{
      display: 'flex',
      minHeight: '100vh',
      width: '100vw',
      backgroundColor: 'var(--bg-primary)',
      overflow: 'hidden'
    }}>
      {/* Sidebar Navigation */}
      <Sidebar activeTab={activeTab} setActiveTab={setActiveTab} />

      {/* Main Content Workspace */}
      <main style={{
        flex: 1,
        height: '100vh',
        overflowY: 'auto',
        padding: '24px 30px',
        display: 'flex',
        flexDirection: 'column'
      }}>
        {activeTab === 'search' ? (
          !isSplitView ? (
            // Full Width Search Portal
            <div style={{ maxWidth: '1200px', width: '100%', margin: '0 auto' }}>
              <SearchPortal
                apiBase={API_BASE}
                activeQuery={activeQuery}
                activeCategory={activeCategory}
                similarForId={similarForId}
                setSimilarForId={setSimilarForId}
                searchResults={searchResults}
                loading={searchLoading}
                onSearchSubmit={handleSearchSubmit}
                isSplitView={false}
              />
            </div>
          ) : (
            // Split Panel View: Search Results on Left, Chat Assistant on Right
            <div style={{
              display: 'flex',
              gap: '24px',
              height: 'calc(100vh - 48px)',
              overflow: 'hidden'
            }}>
              {/* Left Column - Retrieval Grid */}
              <div style={{
                flex: '1.2 1 0',
                overflowY: 'auto',
                paddingRight: '8px'
              }}>
                <SearchPortal
                  apiBase={API_BASE}
                  activeQuery={activeQuery}
                  activeCategory={activeCategory}
                  similarForId={similarForId}
                  setSimilarForId={setSimilarForId}
                  searchResults={searchResults}
                  loading={searchLoading}
                  onSearchSubmit={handleSearchSubmit}
                  isSplitView={true}
                />
              </div>

              {/* Right Column - Chat Assistant */}
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
                />
              </div>
            </div>
          )
        ) : (
          // Caption Annotator Page
          <div style={{ maxWidth: '1200px', width: '100%', margin: '0 auto' }}>
            <Annotator apiBase={API_BASE} />
          </div>
        )}
      </main>
    </div>
  );
}

export default App;
