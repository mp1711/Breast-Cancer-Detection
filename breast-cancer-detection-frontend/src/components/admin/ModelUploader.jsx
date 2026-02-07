import { useState } from 'react';
import { uploadModels } from '../../api/models';
import Button from '../common/Button';
import './ModelUploader.css';

const ModelUploader = ({ datasetId, onUploadComplete }) => {
    const [selectedFiles, setSelectedFiles] = useState({
        CNN: null,
        VGG16: null,
        VGG19: null,
        ResNet: null
    });
    const [uploading, setUploading] = useState(false);
    const [results, setResults] = useState(null);
    const [error, setError] = useState(null);

    const handleFileSelect = (modelType, file) => {
        if (!file) {
            setSelectedFiles({ ...selectedFiles, [modelType]: null });
            return;
        }

        // Validate file extension
        const validExtensions = ['.keras', '.h5'];
        const fileName = file.name.toLowerCase();
        const isValid = validExtensions.some(ext => fileName.endsWith(ext));

        if (!isValid) {
            setError(`Invalid file format for ${modelType}. Only .keras and .h5 files are supported.`);
            return;
        }

        setError(null);
        setSelectedFiles({ ...selectedFiles, [modelType]: file });
    };

    const handleUploadAll = async () => {
        // Check if at least one file is selected
        const hasFiles = Object.values(selectedFiles).some(file => file !== null);
        if (!hasFiles) {
            setError('Please select at least one model file to upload.');
            return;
        }

        setUploading(true);
        setError(null);
        setResults(null);

        try {
            const result = await uploadModels(datasetId, selectedFiles);
            setResults(result.results);

            // Check if all uploads were successful
            const allSuccessful = result.results.every(r => r.success);
            if (allSuccessful) {
                // Clear selected files on success
                setSelectedFiles({
                    CNN: null,
                    VGG16: null,
                    VGG19: null,
                    ResNet: null
                });
                // Reset file inputs
                const fileInputs = document.querySelectorAll('.model-uploader input[type="file"]');
                fileInputs.forEach(input => input.value = '');
            }

            // Notify parent component
            if (onUploadComplete) {
                onUploadComplete(result);
            }
        } catch (err) {
            setError(`Upload failed: ${err.response?.data?.detail || err.message}`);
        } finally {
            setUploading(false);
        }
    };

    const getFileInputStatus = (modelType) => {
        if (!selectedFiles[modelType]) return null;
        return selectedFiles[modelType].name;
    };

    return (
        <div className="model-uploader">
            <h3>Upload Pre-trained Models</h3>
            <p className="uploader-info">
                Upload one or more pre-trained models (.keras or .h5 format).
                Selected models will be validated and tested automatically.
            </p>

            {error && (
                <div className="error-message">
                    {error}
                </div>
            )}

            <div className="upload-form">
                {['CNN', 'VGG16', 'VGG19', 'ResNet'].map(modelType => (
                    <div key={modelType} className="upload-row">
                        <label className="model-label">{modelType}</label>
                        <div className="file-input-wrapper">
                            <input
                                type="file"
                                accept=".keras,.h5"
                                onChange={(e) => handleFileSelect(modelType, e.target.files[0])}
                                disabled={uploading}
                                className="file-input"
                            />
                            {getFileInputStatus(modelType) && (
                                <span className="file-name">{getFileInputStatus(modelType)}</span>
                            )}
                        </div>
                    </div>
                ))}
            </div>

            <div className="upload-actions">
                <Button
                    onClick={handleUploadAll}
                    disabled={uploading || Object.values(selectedFiles).every(f => f === null)}
                    className="upload-btn"
                >
                    {uploading ? 'Uploading and Testing...' : 'Upload All Selected'}
                </Button>
            </div>

            {results && results.length > 0 && (
                <div className="upload-results">
                    <h4>Upload Results</h4>
                    {results.map((result, index) => (
                        <div
                            key={index}
                            className={`result-item ${result.success ? 'success' : 'error'}`}
                        >
                            <strong>{result.model_type}</strong>
                            {result.success ? (
                                <div className="result-metrics">
                                    <span className="success-icon">✓</span>
                                    <span>Accuracy: {(result.metrics.accuracy * 100).toFixed(2)}%</span>
                                    <span>Loss: {result.metrics.loss.toFixed(4)}</span>
                                    <span>AUC: {result.metrics.auc.toFixed(4)}</span>
                                </div>
                            ) : (
                                <div className="result-error">
                                    <span className="error-icon">✗</span>
                                    <span>{result.error}</span>
                                </div>
                            )}
                        </div>
                    ))}
                </div>
            )}
        </div>
    );
};

export default ModelUploader;
