import React, { useState, useEffect } from 'react';
import { 
    trainDatasetModels, 
    getDatasetModels, 
    getDatasetModelPlot,
    setDatasetBestModel
} from '../../api/models';
import { fetchDatasets } from '../../api/datasets';
import { Button, Card, Alert, Loading } from '../common';

const ModelTrainer = () => {
    const [datasets, setDatasets] = useState([]);
    const [selectedDatasetId, setSelectedDatasetId] = useState('');
    const [loading, setLoading] = useState(true);
    const [trainingInProgress, setTrainingInProgress] = useState(false);
    const [error, setError] = useState(null);
    const [success, setSuccess] = useState(null);
    const [showResults, setShowResults] = useState(false);
    const [selectedModel, setSelectedModel] = useState('');
    const [datasetModels, setDatasetModels] = useState([]);

    // Common plot types
    const plotTypes = ['roc_curve', 'precision_recall', 'training_history', 'confusion_matrix'];

    // Load datasets when component mounts
    useEffect(() => {
        const loadDatasets = async () => {
            try {
                const data = await fetchDatasets();
                setDatasets(data || []);
                if (data && data.length > 0) {
                    setSelectedDatasetId(data[0].id);
                    loadModelsForDataset(data[0].id);
                }
            } catch (err) {
                setError('Failed to load datasets');
            } finally {
                setLoading(false);
            }
        };

        loadDatasets();
    }, []);

    // Load models when dataset changes
    useEffect(() => {
        if (selectedDatasetId) {
            loadModelsForDataset(selectedDatasetId);
        }
    }, [selectedDatasetId]);

    const loadModelsForDataset = async (datasetId) => {
        try {
            setLoading(true);
            const models = await getDatasetModels(datasetId);
            setDatasetModels(models || []);
            
            if (models && models.length > 0) {
                setSelectedModel(models[0].id.toString());
                setShowResults(true);
            } else {
                setShowResults(false);
            }
        } catch (err) {
            console.error('Failed to load models:', err);
            setError('Failed to load models for selected dataset');
            setShowResults(false);
        } finally {
            setLoading(false);
        }
    };

    const handleStartTraining = async () => {
        if (!selectedDatasetId) {
            setError('Please select a dataset first');
            return;
        }

        try {
            setError(null);
            setSuccess(null);
            setTrainingInProgress(true);

            // The API doesn't support tracking progress, so we just initiate training
            const response = await trainDatasetModels(selectedDatasetId);
            
            // Since training is synchronous in the new API, we can immediately reload models
            await loadModelsForDataset(selectedDatasetId);
            setSuccess('Training completed successfully!');
            setShowResults(true);
        } catch (err) {
            setError('Failed to train models: ' + (err.message || 'Unknown error'));
        } finally {
            setTrainingInProgress(false);
        }
    };

    const handleSetBestModel = async (modelName) => {
        if (!selectedDatasetId || !modelName) {
            setError('Dataset and model must be selected');
            return;
        }

        try {
            setError(null);
            await setDatasetBestModel(selectedDatasetId, modelName);
            setSuccess(`Model ${modelName} set as best model for this dataset`);
            // Reload models to update UI
            await loadModelsForDataset(selectedDatasetId);
        } catch (err) {
            setError('Failed to set best model: ' + (err.message || 'Unknown error'));
        }
    };

    const formatPlotTypeName = (plotType) => {
        return plotType
            .split('_')
            .map(word => word.charAt(0).toUpperCase() + word.slice(1))
            .join(' ');
    };

    const getModelNameById = (modelId) => {
        const model = datasetModels.find(m => m.id.toString() === modelId.toString());
        return model ? (model.name || `Model ${model.id}`) : modelId;
    };

    if (loading && !datasetModels.length) {
        return <Loading />;
    }

    return (
        <div className="model-trainer">
            <h2>Train Models</h2>

            {error && <Alert message={error} type="error" />}
            {success && <Alert message={success} type="success" />}

            <div className="dataset-selection">
                <label htmlFor="dataset-select">Select Dataset:</label>
                <select
                    id="dataset-select"
                    value={selectedDatasetId}
                    onChange={(e) => setSelectedDatasetId(e.target.value)}
                    disabled={trainingInProgress}
                >
                    {datasets.length === 0 ? (
                        <option value="">No datasets available</option>
                    ) : (
                        datasets.map(dataset => (
                            <option key={dataset.id} value={dataset.id}>
                                {dataset.name}
                            </option>
                        ))
                    )}
                </select>
            </div>

            <Button
                onClick={handleStartTraining}
                disabled={trainingInProgress || !selectedDatasetId}
                className="start-training-btn"
            >
                {trainingInProgress ? 'Training in Progress...' : 'Start Model Training'}
            </Button>

            {showResults && datasetModels.length > 0 && (
                <div className="model-results">
                    <h3>Training Results</h3>
                    
                    <div className="model-selector">
                        <label>Select Model:</label>
                        <div className="model-buttons">
                            {datasetModels.map(model => (
                                <button
                                    key={model.id}
                                    className={`model-button ${selectedModel === model.id.toString() ? 'active' : ''}`}
                                    onClick={() => setSelectedModel(model.id.toString())}
                                >
                                    {model.name || `Model ${model.id}`}
                                </button>
                            ))}
                        </div>
                    </div>

                    <div className="model-actions">
                        <Button
                            onClick={() => handleSetBestModel(getModelNameById(selectedModel))}
                            className="set-best-model-btn"
                        >
                            Set as Best Model
                        </Button>
                    </div>
                    
                    <div className="plots-container">
                        {plotTypes.map(plotType => (
                            <Card key={plotType} title={formatPlotTypeName(plotType)}>
                                <img
                                    src={getDatasetModelPlot(selectedDatasetId, getModelNameById(selectedModel), plotType)}
                                    alt={`${getModelNameById(selectedModel)} ${plotType}`}
                                    className="model-plot"
                                    onError={(e) => {
                                        e.target.onerror = null;
                                        e.target.src = '/placeholder-chart.png';
                                        e.target.alt = 'Plot not available';
                                    }}
                                />
                            </Card>
                        ))}
                    </div>
                </div>
            )}
            
            {showResults && datasetModels.length === 0 && (
                <p>No models available for this dataset. Start training to see results.</p>
            )}
        </div>
    );
};

export default ModelTrainer;
