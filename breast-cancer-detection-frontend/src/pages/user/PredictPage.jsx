import React, { useState, useEffect } from "react";
import { predictWithDataset, getExplanationImage } from "../../api/predictions";
import { fetchDatasets } from "../../api/datasets";
import { Button, Card, Loading, Alert } from "../../components/common";

const PredictPage = () => {
  const [selectedFile, setSelectedFile] = useState(null);
  const [loading, setLoading] = useState(false);
  const [datasetsLoading, setDatasetsLoading] = useState(true);
  const [predictionResult, setPredictionResult] = useState(null);
  const [error, setError] = useState(null);
  const [datasets, setDatasets] = useState([]);
  const [selectedDatasetId, setSelectedDatasetId] = useState("");

  // Load datasets when component mounts
  useEffect(() => {
    const loadDatasets = async () => {
      try {
        const data = await fetchDatasets();
        setDatasets(data || []);
        if (data && data.length > 0) {
          setSelectedDatasetId(data[0].id);
        }
        setError(null);
      } catch (err) {
        console.error("Failed to load datasets:", err);
        setError("Failed to load datasets. Please try again.");
      } finally {
        setDatasetsLoading(false);
      }
    };

    loadDatasets();
  }, []);

  const handleFileChange = (event) => {
    setSelectedFile(event.target.files[0]);
    setPredictionResult(null); // Reset previous prediction
  };

  const handleSubmit = async (event) => {
    event.preventDefault();
    if (!selectedFile) {
      setError("Please select an image to upload.");
      return;
    }

    if (!selectedDatasetId) {
      setError("Please select a dataset for prediction.");
      return;
    }

    setLoading(true);
    setError(null);

    try {
      const result = await predictWithDataset(selectedDatasetId, selectedFile);
      setPredictionResult(result);
    } catch (err) {
      console.error("Prediction error:", err);
      setError("Error processing image. Please try again.");
    } finally {
      setLoading(false);
    }
  };

  if (datasetsLoading) {
    return <Loading />;
  }

  return (
    <div className="predict-page">
      <h1>Breast Cancer Prediction</h1>

      <Card title="Upload Image for Prediction">
        {error && <Alert message={error} type="error" />}

        <form onSubmit={handleSubmit}>
          <div className="dataset-selection">
            <label htmlFor="dataset-select">
              Select Dataset for Prediction:
            </label>
            <select
              id="dataset-select"
              value={selectedDatasetId}
              onChange={(e) => setSelectedDatasetId(e.target.value)}
              disabled={loading}
            >
              {datasets.length === 0 ? (
                <option value="">No datasets available</option>
              ) : (
                datasets.map((dataset) => (
                  <option key={dataset.id} value={dataset.id}>
                    {dataset.name}
                  </option>
                ))
              )}
            </select>
          </div>

          <div className="file-input-container">
            <label htmlFor="image-upload">Select Breast Image:</label>
            <div className="file-upload-wrapper">
              <input
                id="image-upload"
                type="file"
                accept="image/*"
                onChange={handleFileChange}
                disabled={loading}
                className="file-upload-input"
              />
              <div className="file-upload-button">
                <span>Choose File</span>
              </div>
            </div>

            {selectedFile && (
              <div className="selected-file-preview">
                <p>
                  Selected file: <strong>{selectedFile.name}</strong>
                </p>
                <img
                  src={URL.createObjectURL(selectedFile)}
                  alt="Preview"
                  className="image-preview"
                />
              </div>
            )}
          </div>

          <Button
            type="submit"
            disabled={loading || !selectedFile || !selectedDatasetId}
            className="predict-btn"
          >
            {loading ? "Processing..." : "Predict"}
          </Button>
        </form>
      </Card>

      {loading && <Loading />}

      {predictionResult && (
        <div className="prediction-result">
          <h2>Prediction Result</h2>
          <Card>
            <div className="result-content">
              <p className="prediction-class">
                <strong>Diagnosis:</strong>{" "}
                <span className="diagnosis-label">{predictionResult.result.label}</span>
              </p>

              <p className="prediction-confidence">
                <strong>Confidence:</strong>{" "}
                <div className="confidence-bar">
                  <div 
                    className="confidence-fill" 
                    style={{ width: `${(predictionResult.result.confidence * 100).toFixed(2)}%` }}
                  >
                    {(predictionResult.result.confidence * 100).toFixed(2)}%
                  </div>
                </div>
              </p>

              <p className="dataset-info">
                <strong>Dataset:</strong> {predictionResult.dataset_name}
              </p>

              {predictionResult.result.explanation && (
                <div className="explanation">
                  <h3>LIME Explanation</h3>
                  <a 
                    href={getExplanationImage(predictionResult.result.explanation)} 
                    target="_blank" 
                    rel="noreferrer"
                    className="explanation-link"
                  >
                    <img
                      src={getExplanationImage(predictionResult.result.explanation)}
                      alt="LIME Explanation"
                      className="lime-image"
                    />
                    <div className="explanation-overlay">
                      <span>View Full Size</span>
                    </div>
                  </a>
                  <p className="explanation-note">
                    The highlighted areas show regions that influenced the
                    prediction. Click on the image to view in full size.
                  </p>
                </div>
              )}
            </div>
          </Card>
        </div>
      )}
    </div>
  );
};

export default PredictPage;