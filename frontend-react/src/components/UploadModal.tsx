import React, { useState, useRef } from 'react';
import { Upload, X, CheckCircle, Loader2 } from 'lucide-react';

interface UploadModalProps {
  isOpen: boolean;
  onClose: () => void;
  onUploadSuccess: (asset: any) => void;
  apiBase: string;
}

export const UploadModal: React.FC<UploadModalProps> = ({ isOpen, onClose, onUploadSuccess, apiBase }) => {
  const [file, setFile] = useState<File | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [successResult, setSuccessResult] = useState<any | null>(null);
  const fileInputRef = useRef<HTMLInputElement>(null);

  if (!isOpen) return null;

  const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files && e.target.files.length > 0) {
      setFile(e.target.files[0]);
      setError(null);
      setSuccessResult(null);
    }
  };

  const handleDragOver = (e: React.DragEvent) => {
    e.preventDefault();
  };

  const handleDrop = (e: React.DragEvent) => {
    e.preventDefault();
    if (e.dataTransfer.files && e.dataTransfer.files.length > 0) {
      const droppedFile = e.dataTransfer.files[0];
      const validTypes = ['image/jpeg', 'image/png', 'image/jpg'];
      if (validTypes.includes(droppedFile.type)) {
        setFile(droppedFile);
        setError(null);
        setSuccessResult(null);
      } else {
        setError('Only JPEG, JPG, and PNG files are supported.');
      }
    }
  };

  const handleUpload = async () => {
    if (!file) return;

    setLoading(true);
    setError(null);
    setSuccessResult(null);

    const formData = new FormData();
    formData.append('file', file);

    try {
      const response = await fetch(`${apiBase}/api/upload-and-index`, {
        method: 'POST',
        body: formData,
      });

      const resJson = await response.json();
      if (resJson.code !== 200) {
        throw new Error(resJson.message || 'Upload failed');
      }

      const resultData = resJson.data;
      setSuccessResult(resultData);
      onUploadSuccess(resultData);
      setFile(null);
    } catch (err: any) {
      setError(err.message || 'An error occurred during file upload');
    } finally {
      setLoading(false);
    }
  };

  return (
    <div style={{
      position: 'fixed',
      top: 0,
      left: 0,
      right: 0,
      bottom: 0,
      backgroundColor: 'rgba(5, 10, 20, 0.85)',
      display: 'flex',
      alignItems: 'center',
      justifyContent: 'center',
      zIndex: 1000,
      backdropFilter: 'blur(8px)',
      animation: 'fadeIn var(--transition-fast) forwards'
    }}>
      <div className="glass-panel" style={{
        width: '500px',
        maxHeight: '90vh',
        overflowY: 'auto',
        padding: '30px',
        position: 'relative',
        boxShadow: '0 20px 50px rgba(0, 210, 255, 0.15)'
      }}>
        {/* Header */}
        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '20px' }}>
          <h3 style={{ fontSize: '1.25rem', display: 'flex', alignItems: 'center', gap: '8px' }}>
            <Upload size={20} className="glow-text" style={{ color: 'var(--accent-blue)' }} />
            Upload & Index Image
          </h3>
          <button 
            onClick={onClose} 
            className="btn btn-secondary" 
            style={{ padding: '6px', borderRadius: '50%', border: 'none', background: 'transparent' }}
          >
            <X size={20} />
          </button>
        </div>

        {/* Upload Container */}
        {!successResult && (
          <div 
            onDragOver={handleDragOver}
            onDrop={handleDrop}
            onClick={() => fileInputRef.current?.click()}
            style={{
              border: '2px dashed var(--border-color)',
              borderRadius: '10px',
              padding: '40px 20px',
              textAlign: 'center',
              cursor: 'pointer',
              background: 'rgba(13, 22, 39, 0.4)',
              transition: 'all var(--transition-fast)',
              marginBottom: '20px'
            }}
            onMouseEnter={(e) => {
              e.currentTarget.style.borderColor = 'var(--accent-blue)';
              e.currentTarget.style.boxShadow = '0 0 10px var(--accent-glow)';
            }}
            onMouseLeave={(e) => {
              e.currentTarget.style.borderColor = 'var(--border-color)';
              e.currentTarget.style.boxShadow = 'none';
            }}
          >
            <input 
              type="file" 
              ref={fileInputRef} 
              onChange={handleFileChange} 
              style={{ display: 'none' }} 
              accept=".jpg,.jpeg,.png"
            />
            <Upload size={32} style={{ color: 'var(--text-secondary)', marginBottom: '12px' }} />
            <p style={{ fontWeight: '500', marginBottom: '4px' }}>
              {file ? file.name : 'Click or drag image file to upload'}
            </p>
            <p style={{ fontSize: '0.8rem', color: 'var(--text-muted)' }}>
              Supports JPG, JPEG, PNG
            </p>
          </div>
        )}

        {/* Selected File Details */}
        {file && !loading && !successResult && (
          <div style={{ marginBottom: '20px', fontSize: '0.9rem', color: 'var(--text-secondary)' }}>
            <p><strong>Selected File:</strong> {file.name} ({(file.size / 1024).toFixed(1)} KB)</p>
          </div>
        )}

        {/* Success View */}
        {successResult && (
          <div style={{
            background: 'rgba(0, 230, 118, 0.08)',
            border: '1px solid rgba(0, 230, 118, 0.3)',
            borderRadius: '8px',
            padding: '20px',
            marginBottom: '20px'
          }}>
            <div style={{ display: 'flex', alignItems: 'center', gap: '8px', color: 'var(--success)', marginBottom: '10px', fontWeight: '600' }}>
              <CheckCircle size={18} />
              Successfully Indexed!
            </div>
            <p style={{ fontSize: '0.95rem', fontWeight: '500', marginBottom: '8px' }}>
              Title: {successResult.title}
            </p>
            <p style={{ fontSize: '0.85rem', color: 'var(--text-secondary)', background: 'rgba(0,0,0,0.2)', padding: '10px', borderRadius: '4px', fontStyle: 'italic' }}>
              <strong>BLIP Caption:</strong> {successResult.caption}
            </p>
          </div>
        )}

        {/* Error Alert */}
        {error && (
          <div style={{
            background: 'rgba(255, 82, 82, 0.08)',
            border: '1px solid rgba(255, 82, 82, 0.3)',
            color: 'var(--error)',
            padding: '12px 16px',
            borderRadius: '8px',
            fontSize: '0.9rem',
            marginBottom: '20px'
          }}>
            {error}
          </div>
        )}

        {/* Action Buttons */}
        <div style={{ display: 'flex', justifyContent: 'flex-end', gap: '10px' }}>
          <button 
            onClick={onClose} 
            className="btn btn-secondary"
            disabled={loading}
          >
            {successResult ? 'Close' : 'Cancel'}
          </button>
          
          {!successResult && (
            <button 
              onClick={handleUpload} 
              className="btn btn-primary"
              disabled={!file || loading}
              style={{ minWidth: '120px' }}
            >
              {loading ? (
                <>
                  <Loader2 size={16} className="animate-spin" style={{ animation: 'spin 1s linear infinite' }} />
                  <span>Uploading...</span>
                </>
              ) : (
                'Upload & Index'
              )}
            </button>
          )}
        </div>
      </div>
    </div>
  );
};
