import React, { useEffect, useState } from "react";
import { getDatasets, deleteDataset, uploadDataset } from "../../api/datasets";
import { Button, Card, Loading, Alert } from "../../components/common";

const DatasetsPage = () => {
  const [datasets, setDatasets] = useState([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);
  const [success, setSuccess] = useState(null);
  const [selectedFile, setSelectedFile] = useState(null);
  const [isUploading, setIsUploading] = useState(false);
  const [description, setDescription] = useState('');

  useEffect(() => {
    loadDatasets();
  }, []);

  const loadDatasets = async () => {
    try {
      setLoading(true);
      const data = await getDatasets();
      setDatasets(data || []);
      setError(null);
    } catch (err) {
      console.error("Failed to load datasets:", err);
      setError("Failed to load datasets. Please try again.");
    } finally {
      setLoading(false);
    }
  };

  const handleFileChange = (e) => {
    setSelectedFile(e.target.files[0]);
  };

  const handleDescriptionChange = (e) => {
    setDescription(e.target.value);
  };

  const handleUpload = async () => {
    if (!selectedFile) {
      setError("Please select a ZIP file to upload");
      return;
    }

    if (!selectedFile.name.endsWith('.zip')) {
      setError("Please select a ZIP file");
      return;
    }

    setIsUploading(true);
    setError(null);
    setSuccess(null);

    try {
      const formData = new FormData();
      formData.append('file', selectedFile);
      
      await uploadDataset(formData, description);
      setSuccess("Dataset uploaded successfully");
      setSelectedFile(null);
      setDescription('');
      // Reset file input
      document.getElementById('dataset-upload').value = "";
      // Reload datasets
      await loadDatasets();
    } catch (err) {
      console.error("Upload error:", err);
      setError("Failed to upload dataset. Please try again.");
    } finally {
      setIsUploading(false);
    }
  };

  const handleDelete = async (datasetId) => {
    if (!window.confirm("Are you sure you want to delete this dataset?")) {
      return;
    }

    try {
      setError(null);
      setSuccess(null);
      await deleteDataset(datasetId);
      setSuccess("Dataset deleted successfully");
      // Update the list without reloading
      setDatasets(datasets.filter(dataset => dataset.id !== datasetId));
    } catch (err) {
      console.error("Delete error:", err);
      setError("Failed to delete dataset");
    }
  };

  if (loading && datasets.length === 0) return <Loading />;

  return (
    <div className="datasets-page">
      <h1>Datasets Management</h1>
      
      <div className="upload-section">
        <h2>Upload New Dataset</h2>
        {error && <Alert message={error} type="error" />}
        {success && <Alert message={success} type="success" />}
        
        <div className="upload-form">
          <input 
            id="dataset-upload"
            type="file" 
            accept=".zip" 
            onChange={handleFileChange} 
            disabled={isUploading}
            className="upload-input"
          />
          
          <div className="description-input">
            <label htmlFor="dataset-description">Description (optional):</label>
            <textarea
              id="dataset-description"
              value={description}
              onChange={handleDescriptionChange}
              disabled={isUploading}
              rows="3"
              placeholder="Enter dataset description"
              className="description-textarea"
            />
          </div>
          
          <Button 
            onClick={handleUpload} 
            disabled={!selectedFile || isUploading}
            className="upload-btn"
          >
            {isUploading ? "Uploading..." : "Upload Dataset"}
          </Button>
        </div>
        
        {selectedFile && (
          <p className="selected-file">
            Selected file: <strong>{selectedFile.name}</strong> ({(selectedFile.size / 1024 / 1024).toFixed(2)} MB)
          </p>
        )}
      </div>
      
      <div className="datasets-list">
        <h2>Available Datasets</h2>
        {datasets.length === 0 ? (
          <p>No datasets available. Upload a new dataset to get started.</p>
        ) : (
          <div className="cards-grid">
            {datasets.map((dataset) => (
              <Card key={dataset.id || dataset.name} title={dataset.name}>
                <div className="dataset-info">
                  {dataset.size && <p>Size: {dataset.size} images</p>}
                  {dataset.created_at && <p>Uploaded: {new Date(dataset.created_at).toLocaleDateString()}</p>}
                  {dataset.description && <p>{dataset.description}</p>}
                </div>
                <div className="card-actions">
                  <Button 
                    onClick={() => handleDelete(dataset.id || dataset.name)}
                    className="delete-btn"
                  >
                    Delete Dataset
                  </Button>
                </div>
              </Card>
            ))}
          </div>
        )}
      </div>
    </div>
  );
};

export default DatasetsPage;
