import React, { useEffect, useState } from "react";
import {
  getDatasetModels,
  setDatasetBestModel,
  trainDatasetModels,
  getDatasetBestModel,
} from "../../api/models";
import { fetchDatasets } from "../../api/datasets";
import { Button, Card, Loading, Alert } from "../../components/common";
import { getDatasetModelPlot } from "../../api/models";

const API_BASE_URL = import.meta.env.VITE_API_URL || "http://localhost:8000";

const ModelsPage = () => {
  const [models, setModels] = useState([]);
  const [datasets, setDatasets] = useState([]);
  const [selectedDatasetId, setSelectedDatasetId] = useState("");
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);
  const [success, setSuccess] = useState(null);
  const [isTraining, setIsTraining] = useState(false);
  const [bestModel, setBestModel] = useState(null);
  const [selectedModelName, setSelectedModelName] = useState("none");

  const plotTypes = [
    "roc_curve",
    "precision_recall",
    "training_history",
    "confusion_matrix",
  ];

  useEffect(() => {
    loadInitialData();
  }, []);

  const loadInitialData = async () => {
    try {
      setLoading(true);
      // Load available datasets
      const datasetsData = await fetchDatasets();
      setDatasets(datasetsData || []);

      // Select first dataset if available and load its models
      if (datasetsData && datasetsData.length > 0) {
        const firstDataset = datasetsData[0];
        setSelectedDatasetId(firstDataset.id);
        await loadModelsForDataset(firstDataset.id);
      } else {
        setLoading(false);
      }
    } catch (error) {
      console.error("Error loading initial data:", error);
      setError("Failed to load datasets. Please try again.");
      setLoading(false);
    }
  };

  const loadModelsForDataset = async (datasetId) => {
    if (!datasetId) {
      setModels([]);
      setBestModel(null);
      setLoading(false);
      return;
    }

    try {
      setLoading(true);
      const modelsData = await getDatasetModels(datasetId);
      setModels(modelsData || []);

      // Also load best model if available
      try {
        const bestModelData = await getDatasetBestModel(datasetId);
        setBestModel(bestModelData);
      } catch (err) {
        console.log("No best model found for this dataset");
        setBestModel(null);
      }

      setError(null);
      setSelectedModelName("none"); // Reset selection when dataset changes
    } catch (error) {
      console.error("Error fetching models:", error);
      setError(`Failed to load models for dataset ${datasetId}`);
      setModels([]);
      setBestModel(null);
    } finally {
      setLoading(false);
    }
  };

  const handleDatasetChange = (e) => {
    const newDatasetId = e.target.value;
    setSelectedDatasetId(newDatasetId);
    loadModelsForDataset(newDatasetId);
  };

  const handleModelSelection = (e) => {
    setSelectedModelName(e.target.value);
  };

  const confirmSetBestModel = async () => {
    if (!selectedModelName || selectedModelName === "none") {
      setError("Please select a model first");
      return;
    }

    try {
      await setDatasetBestModel(selectedDatasetId, selectedModelName);
      setSuccess(`Model "${selectedModelName}" set as best model`);

      // Refresh models and best model
      await loadModelsForDataset(selectedDatasetId);

      setTimeout(() => setSuccess(null), 3000);
    } catch (error) {
      console.error("Error setting best model:", error);
      setError("Failed to set best model");
      setTimeout(() => setError(null), 3000);
    }
  };

  const handleStartTraining = async () => {
    if (!selectedDatasetId) {
      setError("Please select a dataset first");
      return;
    }

    try {
      setIsTraining(true);
      setError(null);
      setSuccess(null);

      await trainDatasetModels(selectedDatasetId);

      setSuccess("Training completed successfully!");
      // Reload models to show the newly trained ones
      await loadModelsForDataset(selectedDatasetId);
    } catch (error) {
      console.error("Error training models:", error);
      setError(`Failed to train models: ${error.message || "Unknown error"}`);
    } finally {
      setIsTraining(false);
    }
  };

  const formatPlotTypeName = (plotType) => {
    return plotType
      .split("_")
      .map((word) => word.charAt(0).toUpperCase() + word.slice(1))
      .join(" ");
  };

  // Function to get the full URL for a plot image
  const getPlotUrl = (datasetId, modelName, plotType) => {
    const plotPath = getDatasetModelPlot(datasetId, modelName, plotType);
    return plotPath;
  };

  if (loading && !datasets.length) {
    return <Loading />;
  }

  return (
    <div className="models-page">
      <h1>Models Management</h1>

      {error && <Alert message={error} type="error" />}
      {success && <Alert message={success} type="success" />}

      <div className="dataset-controls">
        <div className="dataset-selector">
          <label htmlFor="dataset-select">Select Dataset:</label>
          <select
            id="dataset-select"
            value={selectedDatasetId}
            onChange={handleDatasetChange}
            disabled={isTraining}
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

        <div className="training-controls">
          <Button
            onClick={handleStartTraining}
            disabled={!selectedDatasetId || isTraining}
            className="train-btn"
          >
            {isTraining ? "Training in progress..." : "Train Models"}
          </Button>
        </div>
      </div>

      {/* Best Model Selection */}
      {models.length > 0 && (
        <div className="best-model-selector">
          <div className="best-model-header">
            <label htmlFor="best-model-select">Set Best Model:</label>
            {bestModel && (
              <div className="current-best-model">
                Current: <span>{bestModel.name || `Model ${bestModel.id}`}</span>
              </div>
            )}
          </div>
          <div className="best-model-controls">
            <select
              id="best-model-select"
              onChange={handleModelSelection}
              value={selectedModelName}
              disabled={isTraining}
            >
              <option value="none">-- Select a model --</option>
              {models.map((model) => (
                <option key={model.id} value={model.name || model.id}>
                  {model.name || `Model ${model.id}`}
                </option>
              ))}
            </select>
            <Button
              onClick={confirmSetBestModel}
              disabled={selectedModelName === "none" || isTraining}
              className="confirm-best-btn"
            >
              Confirm
            </Button>
          </div>
        </div>
      )}

      {/* Best Model Display */}
      {bestModel && (
        <div className="best-model-display">
          <h2>Best Model</h2>
          <Card
            key={bestModel.id}
            title={bestModel.name || `Model ${bestModel.id}`}
            className="best-model-card"
          >
            {bestModel.description && <p>{bestModel.description}</p>}
            <div className="model-badge">✓ Best Model</div>

            {selectedDatasetId && (
              <div className="best-model-visuals">
                <h3>Model Visualizations</h3>
                <div className="plots-grid">
                  {plotTypes.map((plotType) => (
                    <div key={plotType} className="plot-container">
                      <h4>{formatPlotTypeName(plotType)}</h4>
                      <a
                        href={getPlotUrl(
                          selectedDatasetId,
                          bestModel.name || `Model_${bestModel.id}`,
                          plotType
                        )}
                        target="_blank"
                        rel="noreferrer"
                        className="plot-link"
                      >
                        <img
                          src={getPlotUrl(
                            selectedDatasetId,
                            bestModel.name || `Model_${bestModel.id}`,
                            plotType
                          )}
                          alt={`${
                            bestModel.name || "Best Model"
                          } ${formatPlotTypeName(plotType)}`}
                          className="model-plot"
                          onError={(e) => {
                            e.target.onerror = null;
                            e.target.src = "/placeholder-chart.png";
                            e.target.alt = "Plot not available";
                          }}
                        />
                        <div className="plot-overlay">
                          <span>View Full Size</span>
                        </div>
                      </a>
                    </div>
                  ))}
                </div>
              </div>
            )}
          </Card>
        </div>
      )}

      <h2>Available Models</h2>
      <div className="models-list">
        {models.length === 0 ? (
          <p className="no-models">
            No models available for this dataset. Train models to get started.
          </p>
        ) : (
          <div className="cards-grid">
            {models.map((model) => (
              <Card
                key={model.id}
                title={model.name || `Model ${model.id}`}
                className={
                  bestModel && bestModel.id === model.id
                    ? "model-card best-model"
                    : "model-card"
                }
              >
                {model.description && <p>{model.description}</p>}

                {model.metrics && (
                  <div className="model-metrics">
                    {model.metrics.accuracy !== undefined && (
                      <p>
                        <strong>Accuracy:</strong>{" "}
                        {(model.metrics.accuracy * 100).toFixed(2)}%
                      </p>
                    )}
                    {model.metrics.loss !== undefined && (
                      <p>
                        <strong>Loss:</strong> {model.metrics.loss.toFixed(4)}
                      </p>
                    )}
                    {model.metrics.auc !== undefined && (
                      <p>
                        <strong>AUC:</strong> {model.metrics.auc.toFixed(4)}
                      </p>
                    )}
                  </div>
                )}

                {bestModel && bestModel.id === model.id && (
                  <div className="best-model-indicator">✓ Best Model</div>
                )}

                {model.plots && Object.keys(model.plots).length > 0 && (
                  <div className="model-plots">
                    <h3>Available Plots</h3>
                    <ul className="plots-list">
                      {Object.keys(model.plots).map((plotKey) => (
                        <li key={plotKey}>
                          <a
                            href={`http://localhost:8000${model.plots[plotKey]}`}
                            target="_blank"
                            rel="noreferrer"
                            className="plot-link-text"
                          >
                            <i className="plot-icon"></i>
                            {formatPlotTypeName(plotKey)}
                          </a>
                        </li>
                      ))}
                    </ul>
                  </div>
                )}
              </Card>
            ))}
          </div>
        )}
      </div>
    </div>
  );
};

export default ModelsPage;