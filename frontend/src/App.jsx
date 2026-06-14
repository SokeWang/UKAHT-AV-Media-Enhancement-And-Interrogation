import React, { useState, useEffect, useRef } from 'react';
import { 
  Search, Eye, Shield, FileText, Image as ImageIcon, 
  Tag, Activity, ChevronRight, Layers, HelpCircle, RefreshCw, Cpu,
  UploadCloud, X, Sparkles, Heart, Compass, ArrowRight, Info
} from 'lucide-react';

export default function App() {
  const [activeTab, setActiveTab] = useState('search');
  
  // Search state
  const [searchQuery, setSearchQuery] = useState('');
  const [searchResults, setSearchResults] = useState([]);
  const [loadingSearch, setLoadingSearch] = useState(false);
  
  // Selected Asset & Recommendation State
  const [selectedAsset, setSelectedAsset] = useState(null);
  const [recommendations, setRecommendations] = useState([]);
  const [loadingRecommendations, setLoadingRecommendations] = useState(false);

  // Upload and index state
  const [uploadingIndex, setUploadingIndex] = useState(false);
  const [uploadStatus, setUploadStatus] = useState('');
  const [indexResult, setIndexResult] = useState(null);


  
  // Oral history transcripts state
  const [selectedTranscript, setSelectedTranscript] = useState(null);
  const transcripts = [
    {
      id: "tr_001",
      speaker: "Alastair (Base Commander, 1958)",
      text: "The snow was up to the eaves of Bransfield House that winter. We had to dig a tunnel just to open the main entrance door and reach the coal store.",
      keyword: "Bransfield House"
    },
    {
      id: "tr_002",
      speaker: "Janet (Conservator, 2012)",
      text: "Rust on the historical food cans inside the kitchen is our main concern. Moisture inside the building accelerates decay on the old iron labels.",
      keyword: "Artifact"
    },
    {
      id: "tr_003",
      speaker: "Tom (Geologist, 1982)",
      text: "We noticed the glacier boundary receding slowly behind the harbor. You can see the bare mountain rocks now that used to be covered in permanent ice.",
      keyword: "Glacial"
    }
  ];

  // Run search on mount
  useEffect(() => {
    handleSearch('');
  }, []);



  const handleSearch = async (queryStr) => {
    setLoadingSearch(true);
    try {
      const res = await fetch('/api/search', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ query: queryStr })
      });
      const data = await res.json();
      setSearchResults(data);
    } catch (e) {
      console.error(e);
    } finally {
      setLoadingSearch(false);
    }
  };

  const fetchRecommendations = async (assetId) => {
    setLoadingRecommendations(true);
    try {
      const res = await fetch('/api/recommend', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ id: assetId, limit: 4 })
      });
      const data = await res.json();
      setRecommendations(data);
    } catch (e) {
      console.error(e);
    } finally {
      setLoadingRecommendations(false);
    }
  };

  const handleSelectAsset = (asset) => {
    setSelectedAsset(asset);
    fetchRecommendations(asset.id);
  };

  const handleUploadAndIndex = async (file) => {
    if (!file) return;
    setUploadingIndex(true);
    setIndexResult(null);
    setUploadStatus('Uploading file...');
    
    // Simulate multi-stage progress for visual polish
    const step2 = setTimeout(() => setUploadStatus('Running BLIP Captioning...'), 800);
    const step3 = setTimeout(() => setUploadStatus('Extracting CLIP Vision Embeddings...'), 1600);
    const step4 = setTimeout(() => setUploadStatus('Writing vector records to SQLite...'), 2400);

    const formData = new FormData();
    formData.append('file', file);
    try {
      const res = await fetch('/api/upload-and-index', {
        method: 'POST',
        body: formData
      });
      const data = await res.json();
      
      clearTimeout(step2);
      clearTimeout(step3);
      clearTimeout(step4);
      
      setIndexResult(data);
      setUploadStatus('Indexing Completed Successfully!');
      // Immediately refresh search to show the newly added item
      handleSearch(searchQuery);
    } catch (e) {
      console.error(e);
      setUploadStatus('Error: Failed to index file.');
    } finally {
      setUploadingIndex(false);
    }
  };



  return (
    <div className="flex flex-col h-full bg-transparent text-[#1d1d1f] font-sans antialiased relative z-10">
      
      {/* Top Navigation Bar with Light Embossed Border */}
      <header className="flex justify-between items-center px-8 py-4 bg-white/80 backdrop-blur-xl shrink-0 sticky top-0 z-50 border-b border-black/[0.06] shadow-[0_1px_0_0_rgba(255,255,255,0.9)]">
        <div className="flex items-center space-x-4">
          <div className="w-9 h-9 rounded-xl bg-gradient-to-b from-[#f4f4f5] to-[#e4e4e7] flex items-center justify-center border border-black/10 shadow-sm">
            <Cpu className="text-zinc-700 w-4 h-4" />
          </div>
          <div>
            <h1 className="text-sm font-semibold tracking-tight text-zinc-900">UKAHT Antarctic Media Portal</h1>
            <p className="text-[10px] text-zinc-400 font-medium">A/V Media Enhancement and Interrogation</p>
          </div>
        </div>
        <div className="flex items-center space-x-3">
        </div>
      </header>

      {/* Main Container */}
      <div className="flex flex-1 overflow-hidden">
        
        {/* Apple-style Sidebar (白色立体面板) */}
        <aside className="w-64 border-r border-black/[0.05] bg-white/40 p-5 space-y-1.5 shrink-0 flex flex-col justify-between shadow-[inset_-1px_0_0_rgba(255,255,255,0.8)]">
          <div className="space-y-1.5">
            <h3 className="text-[9px] font-bold text-zinc-400 uppercase tracking-widest px-3 mb-3">Modules</h3>
            
            <button
              onClick={() => setActiveTab('search')}
              className="w-full flex items-center space-x-3 px-3.5 py-2.5 rounded-xl text-xs font-semibold bg-gradient-to-b from-[#1d1d1f] to-[#121214] text-white shadow-[0_4px_12px_rgba(0,0,0,0.1)] border-t border-black"
            >
              <Search size={14} className="text-white" />
              <span>Search & Recommend</span>
            </button>
          </div>

          <div className="pt-6 border-t border-black/[0.04]">
            <h3 className="text-[9px] font-bold text-zinc-400 uppercase tracking-widest px-3 mb-2.5">Oral History Index</h3>
            <div className="space-y-1.5 max-h-[220px] overflow-y-auto pr-1">
              {transcripts.map((t) => (
                <div 
                  key={t.id}
                  onClick={() => {
                    setSelectedTranscript(t);
                    setSearchQuery(t.keyword);
                    handleSearch(t.keyword);
                    setActiveTab('search');
                  }}
                  className={`p-3 rounded-xl text-[11px] cursor-pointer transition-all border ${
                    selectedTranscript?.id === t.id 
                      ? 'bg-white border-black/10 text-zinc-800 shadow-sm' 
                      : 'bg-transparent border-transparent text-zinc-500 hover:text-zinc-800'
                  }`}
                >
                  <div className="font-semibold text-zinc-700 flex items-center justify-between mb-1">
                    <span>{t.speaker}</span>
                    <ChevronRight size={10} className="text-zinc-400" />
                  </div>
                  <p className="line-clamp-2 italic font-light leading-relaxed">"{t.text}"</p>
                </div>
              ))}
            </div>
          </div>
        </aside>

        {/* Dynamic Center Work Area */}
        <main className="flex-1 overflow-y-auto p-8">          {/* TAB 1: SEARCH & RECOMMENDATION ENGINE */}
          {activeTab === 'search' && (
            <div className="space-y-8 max-w-6xl">
              <div className="flex justify-between items-end">
                <div>
                  <h2 className="text-xl font-bold tracking-tight text-zinc-900 flex items-center space-x-2">
                    <Compass className="text-sky-500 w-5 h-5 animate-pulse" />
                    <span>Image Search & Recommendation Engine</span>
                  </h2>
                  <p className="text-xs text-zinc-500 mt-1">
                    Retrieve archival photographs using natural language descriptions, or upload new files to automatically caption and index them.
                  </p>
                </div>
                {selectedTranscript && (
                  <button 
                    onClick={() => { setSelectedTranscript(null); setSearchQuery(''); handleSearch(''); }}
                    className="text-[10px] text-sky-600 hover:underline flex items-center"
                  >
                    Clear Oral Link Filter ×
                  </button>
                )}
              </div>

              {/* Two Column Layout: Search Area (Left) and Ingestion Area (Right) */}
              <div className="grid grid-cols-1 lg:grid-cols-3 gap-8 items-start">
                
                {/* Search & Results (2/3 width) */}
                <div className="lg:col-span-2 space-y-6">
                  {/* Apple search bar */}
                  <div className="flex space-x-3">
                    <div className="relative flex-1">
                      <Search className="absolute left-3.5 top-3.5 text-zinc-400" size={16} />
                      <input
                        type="text"
                        value={searchQuery}
                        onChange={(e) => setSearchQuery(e.target.value)}
                        onKeyDown={(e) => e.key === 'Enter' && handleSearch(searchQuery)}
                        placeholder="Search images using human language (e.g., 'a wooden hut near a glacier', 'historical metal cans')..."
                        className="w-full pl-10 pr-4 py-3 apple-input rounded-xl text-xs text-zinc-800 focus:outline-none placeholder-zinc-400 shadow-sm"
                      />
                    </div>
                    <button
                      onClick={() => handleSearch(searchQuery)}
                      disabled={loadingSearch}
                      className="px-5 py-3 apple-btn-primary rounded-xl text-xs font-semibold flex items-center space-x-2"
                    >
                      {loadingSearch && <RefreshCw size={12} className="animate-spin" />}
                      <span>Search</span>
                    </button>
                  </div>

                  {/* Results Grid */}
                  {loadingSearch ? (
                    <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                      {[1, 2, 3, 4].map((i) => (
                        <div 
                          key={`shimmer-search-${i}`} 
                          className="apple-card rounded-2xl overflow-hidden flex flex-col pointer-events-none"
                        >
                          <div className="h-44 skeleton-shimmer relative border-b border-black/[0.04]">
                            <div className="absolute top-3 right-3 bg-white/60 w-16 h-5 rounded-full skeleton-shimmer" />
                          </div>
                          <div className="p-4 flex-1 flex flex-col justify-between bg-white/5">
                            <div>
                              <div className="h-4 bg-zinc-250 skeleton-shimmer rounded w-2/3 mb-2.5" />
                              <div className="space-y-1.5">
                                <div className="h-3 bg-zinc-150 skeleton-shimmer rounded w-full" />
                                <div className="h-3 bg-zinc-150 skeleton-shimmer rounded w-5/6" />
                              </div>
                            </div>
                            <div className="flex justify-between items-center mt-6 pt-3 border-t border-black/[0.04]">
                              <div className="h-3 bg-zinc-150 skeleton-shimmer rounded w-24" />
                              <div className="h-4 bg-zinc-200 skeleton-shimmer rounded-full w-16" />
                            </div>
                          </div>
                        </div>
                      ))}
                    </div>
                  ) : searchResults.length > 0 ? (
                    <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                      {searchResults.map((item) => (
                        <div 
                          key={item.id} 
                          onClick={() => handleSelectAsset(item)}
                          className="apple-card rounded-2xl overflow-hidden flex flex-col cursor-pointer group hover:scale-[1.01] transition-transform duration-300"
                        >
                          <div className="h-44 bg-transparent relative overflow-hidden flex items-center justify-center border-b border-black/[0.04]">
                            <img
                              src={item.url}
                              alt={item.title}
                              className="w-full h-full object-cover group-hover:scale-105 transition-transform duration-500"
                              loading="lazy"
                              onError={(e) => {
                                e.target.src = "https://images.unsplash.com/photo-1516690561799-46d8f74f9abf?q=80&w=600";
                              }}
                            />
                            <span className="absolute top-3 right-3 bg-white/95 backdrop-blur-md px-2.5 py-0.5 rounded-full text-[9px] font-bold text-zinc-550 border border-black/5 shadow-sm">
                              {item.category}
                            </span>
                          </div>
                          <div className="p-4 flex-1 flex flex-col justify-between bg-white/10">
                            <div>
                              <h4 className="font-semibold text-zinc-855 text-xs mb-1 group-hover:text-sky-650 transition-colors">{item.title}</h4>
                              <p className="text-[11px] text-zinc-500 font-light leading-relaxed line-clamp-2">"{item.description}"</p>
                            </div>
                            <div className="flex justify-between items-center mt-4 pt-3 border-t border-black/[0.04]">
                              <span className="text-[10px] text-zinc-400 font-medium flex items-center space-x-1">
                                <Info size={10} className="text-zinc-350" />
                                <span>Relevance Match</span>
                              </span>
                              <span className="text-[10px] font-mono font-bold text-sky-650 bg-sky-50 px-2 py-0.5 rounded-full">
                                {(item.score * 100).toFixed(0)}% Match
                              </span>
                            </div>
                          </div>
                        </div>
                      ))}
                    </div>
                  ) : (
                    <div className="flex flex-col items-center justify-center py-20 bg-zinc-50/50 rounded-2xl border border-dashed border-zinc-200">
                      <ImageIcon className="text-zinc-350 mb-3" size={32} />
                      <p className="text-xs text-zinc-500 font-medium">No assets matching your query were found.</p>
                      <p className="text-[10px] text-zinc-400 mt-1">Try general keywords like 'hut', 'metal', 'mountains' or upload your own image to index it.</p>
                    </div>
                  )}
                </div>

                {/* Media Ingestion & Indexing Panel (1/3 width) */}
                <div className="lg:col-span-1 space-y-6">
                  <div className="apple-panel rounded-2xl p-5 space-y-4 shadow-sm border border-black/5 bg-white/60">
                    <div>
                      <h3 className="text-xs font-bold text-zinc-800 flex items-center space-x-1.5">
                        <UploadCloud size={14} className="text-zinc-650" />
                        <span>Ingest & Index New Media</span>
                      </h3>
                      <p className="text-[10px] text-zinc-400 mt-0.5 font-light leading-relaxed">
                        Uploaded assets will be auto-captioned by BLIP, categorized by CLIP, and instantly added to the searchable index.
                      </p>
                    </div>

                    {/* Interactive Dropzone */}
                    <div className="relative flex flex-col items-center justify-center border border-dashed border-zinc-300 rounded-xl p-6 bg-zinc-50/50 hover:bg-zinc-50 hover:border-zinc-400 transition-all cursor-pointer group">
                      <input
                        type="file"
                        accept="image/*"
                        onChange={(e) => handleUploadAndIndex(e.target.files[0])}
                        disabled={uploadingIndex}
                        className="absolute inset-0 w-full h-full opacity-0 cursor-pointer z-10"
                      />
                      <ImageIcon className="text-zinc-400 mb-2 group-hover:scale-105 transition-transform" size={24} />
                      <span className="text-[11px] text-zinc-600 font-semibold">Click or drag image here</span>
                      <span className="text-[9px] text-zinc-400 mt-1">Supports PNG, JPG, JPEG</span>
                    </div>

                    {/* Progress tracking indicator */}
                    {uploadingIndex && (
                      <div className="p-3 bg-sky-50/50 rounded-xl border border-sky-100 flex items-center space-x-3">
                        <RefreshCw size={14} className="animate-spin text-sky-500 shrink-0" />
                        <div className="flex-1 min-w-0">
                          <span className="text-[10px] font-bold text-sky-650 block">AI Ingestion Active</span>
                          <span className="text-[9px] text-sky-500 font-medium truncate block">{uploadStatus}</span>
                        </div>
                      </div>
                    )}

                    {/* Indexing Output */}
                    {indexResult && !uploadingIndex && (
                      <div className="p-4 bg-emerald-50/40 rounded-xl border border-emerald-100 space-y-3">
                        <div className="flex items-center space-x-2 text-emerald-700">
                          <div className="w-4 h-4 rounded-full bg-emerald-100 flex items-center justify-center">
                            <span className="text-[10px] font-bold font-sans">✓</span>
                          </div>
                          <span className="text-[10px] font-bold">Asset Indexed Successfully</span>
                        </div>

                        {/* Thumbnail details */}
                        <div className="flex space-x-3 items-start bg-white/70 p-2.5 rounded-lg border border-black/[0.03] shadow-sm">
                          <div className="w-12 h-12 rounded overflow-hidden border border-black/5 bg-zinc-100 shrink-0">
                            <img 
                              src={indexResult.url} 
                              alt="Indexed" 
                              className="w-full h-full object-cover"
                              loading="lazy"
                            />
                          </div>
                          <div className="flex-1 min-w-0">
                            <h4 className="text-[10px] font-bold text-zinc-800 truncate">{indexResult.title}</h4>
                            <span className="inline-block px-1.5 py-0.2 bg-zinc-800 text-white rounded text-[7px] font-bold uppercase mt-0.5">
                              {indexResult.category}
                            </span>
                            <p className="text-[9px] text-zinc-500 italic truncate mt-1">"{indexResult.description}"</p>
                          </div>
                        </div>
                      </div>
                    )}
                  </div>
                </div>

              </div>
            </div>
          )}

        </main>
      </div>

      {/* Details Drawer Overlay */}
      {selectedAsset && (
        <div className="fixed inset-0 z-50 flex justify-end bg-black/25 backdrop-blur-sm transition-opacity duration-305">
          {/* Backdrop click to close */}
          <div className="absolute inset-0" onClick={() => setSelectedAsset(null)} />
          
          {/* Slide-out Panel */}
          <div className="relative w-[480px] h-full bg-white/95 backdrop-blur-2xl shadow-[-10px_0_30px_rgba(0,0,0,0.06)] border-l border-black/[0.06] flex flex-col z-10 transition-transform duration-300">
            {/* Header */}
            <div className="flex items-center justify-between px-6 py-5 border-b border-black/[0.04] bg-white/50 shrink-0">
              <div className="flex items-center space-x-2">
                <Sparkles size={14} className="text-sky-500 animate-pulse" />
                <span className="text-[10px] font-bold text-zinc-800 uppercase tracking-widest">Asset interrogation</span>
              </div>
              <button 
                onClick={() => setSelectedAsset(null)}
                className="p-1.5 rounded-full hover:bg-black/5 text-zinc-400 hover:text-zinc-750 transition-colors"
              >
                <X size={14} />
              </button>
            </div>
            
            {/* Content */}
            <div className="flex-1 overflow-y-auto p-6 space-y-6">
              {/* Image Preview */}
              <div className="w-full h-60 rounded-2xl overflow-hidden border border-black/5 shadow-md relative group bg-slate-50">
                <img 
                  src={selectedAsset.url} 
                  alt={selectedAsset.title} 
                  className="w-full h-full object-cover"
                  loading="lazy"
                />
                <span className="absolute bottom-3 right-3 bg-white/90 backdrop-blur-md px-2.5 py-0.5 rounded-full text-[9px] font-bold text-zinc-550 border border-black/5 shadow-sm">
                  {selectedAsset.category}
                </span>
              </div>
              
              {/* Metadata Info */}
              <div className="space-y-4">
                <div>
                  <h3 className="text-sm font-semibold text-zinc-900 leading-snug">{selectedAsset.title}</h3>
                  <p className="text-[10px] text-zinc-400 font-mono mt-1">ID: {selectedAsset.id}</p>
                </div>
                
                <div className="p-4 bg-zinc-50/50 rounded-2xl border border-black/[0.03] space-y-2">
                  <span className="text-[9px] font-bold text-zinc-400 uppercase tracking-wider block">Generated Capture</span>
                  <p className="text-xs text-zinc-700 italic leading-relaxed">"{selectedAsset.description}"</p>
                </div>
                
                {selectedAsset.labels && selectedAsset.labels.length > 0 && (
                  <div className="space-y-1.5">
                    <span className="text-[9px] font-bold text-zinc-400 uppercase tracking-wider block">Vision tags</span>
                    <div className="flex flex-wrap gap-1">
                      {selectedAsset.labels.map((lbl, idx) => (
                        <span key={idx} className="px-2 py-0.5 bg-white text-zinc-650 rounded-md border border-black/[0.04] text-[9px] font-medium shadow-sm">
                          {lbl}
                        </span>
                      ))}
                    </div>
                  </div>
                )}
              </div>
              
              {/* Recommendations Section */}
              <div className="space-y-3 pt-4 border-t border-black/[0.04]">
                <div className="flex justify-between items-center">
                  <span className="text-[10px] font-bold text-zinc-400 uppercase tracking-wider">Recommended Similar Assets</span>
                  {loadingRecommendations && <RefreshCw size={10} className="animate-spin text-zinc-400" />}
                </div>
                
                {loadingRecommendations ? (
                  <div className="grid grid-cols-2 gap-3">
                    {[1, 2, 3, 4].map((i) => (
                      <div key={`shimmer-rec-${i}`} className="border border-black/5 rounded-xl p-2 bg-white/40 flex flex-col space-y-2 pointer-events-none">
                        <div className="w-full h-20 skeleton-shimmer rounded-lg" />
                        <div className="h-3 bg-zinc-200 skeleton-shimmer rounded w-3/4" />
                        <div className="h-2.5 bg-zinc-100 skeleton-shimmer rounded w-1/2" />
                      </div>
                    ))}
                  </div>
                ) : recommendations && recommendations.length > 0 ? (
                  <div className="grid grid-cols-2 gap-3">
                    {recommendations.map((rec) => (
                      <div 
                        key={rec.id} 
                        onClick={() => handleSelectAsset(rec)}
                        className="group cursor-pointer apple-card rounded-xl overflow-hidden border border-black/5 flex flex-col p-2 bg-white/40 hover:bg-white/80 transition-all shadow-sm"
                      >
                        <div className="h-20 bg-zinc-100 rounded-lg overflow-hidden relative mb-1.5">
                          <img 
                            src={rec.url} 
                            alt={rec.title} 
                            className="w-full h-full object-cover group-hover:scale-105 transition-transform duration-300"
                            loading="lazy"
                          />
                          <span className="absolute top-1 right-1 bg-sky-500/90 text-white px-1.5 py-0.5 rounded-full text-[8px] font-mono font-bold">
                            {(rec.score * 100).toFixed(0)}%
                          </span>
                        </div>
                        <h4 className="font-semibold text-zinc-800 text-[10px] truncate">{rec.title}</h4>
                        <p className="text-[9px] text-zinc-400 truncate">{rec.category}</p>
                      </div>
                    ))}
                  </div>
                ) : (
                  <div className="text-[10px] text-zinc-400 italic text-center py-4 bg-zinc-50/50 rounded-xl border border-dashed border-zinc-200">
                    No similar assets found.
                  </div>
                )}
              </div>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
