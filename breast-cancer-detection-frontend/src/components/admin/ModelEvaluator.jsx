import React, { useEffect, useState } from 'react';
import { getDatasetModels, getDatasetBestModel, getDatasetModelPlot } from '../../api/models';
import { fetchDatasets } from '../../api/datasets';
import Card from '../common/Card';
import Loading from '../common/Loading';
import Alert from '../common/Alert';

const ModelEvaluator = () => {
    const [models, setModels] = useState([]);
    const [bestModel, setBestModel] = useState(null);
    const [loading, setLoading] = useState(true);
    const [error, setError] = useState(null);
    const [datasets, setDatasets] = useState([]);
    const [selectedDatasetId, setSelectedDatasetId] = useState(null);

    useEffect(() => {
        const loadDatasets = async () => {
            try {
                const datasetsData = await fetchDatasets();
                setDatasets(datasetsData || []);
                
                if (datasetsData && datasetsData.length > 0) {
                    setSelectedDatasetId(datasetsData[0].id);
                }
            } catch (err) {
                console.error('Error loading datasets:', err);
                setError('Failed to load datasets');
            }
        };
        
        loadDatasets();
    }, []);

    useEffect(() => {
        if (selectedDatasetId) {
            getDataForDataset(selectedDatasetId);
        }
    }, [selectedDatasetId]);

    const getDataForDataset = async (datasetId) => {
        try {
            setLoading(true);
            const modelsData = await getDatasetModels(datasetId);
            setModels(modelsData);
            
            try {
                const bestModelData = await getDatasetBestModel(datasetId);
                setBestModel(bestModelData);
            } catch (error) {
                console.log('No best model set for this dataset');
                setBestModel(null);
            }
            
            setError(null);
        } catch (error) {
            console.error("Error fetching dataset models:", error);
            setError("Failed to load model data");
            setModels([]);
            setBestModel(null);
        } finally {
            setLoading(false);
        }
    };

    const handleDatasetChange = (e) => {
        setSelectedDatasetId(e.target.value);
    };

    if (loading && !datasets.length) {
        return <Loading />;
    }

    return (
        <div className="model-evaluator">
            <h2>Model Evaluation</h2>
            
            {error && <Alert message={error} type="error" />}
            
            <div className="dataset-selector">
                <label htmlFor="dataset-select">Select Dataset:</label>
                <select
                    id="dataset-select"
                    value={selectedDatasetId || ''}
                    onChange={handleDatasetChange}
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
            
            {bestModel && (
                <Card title={`Best Model`}>
                    <p><strong>Model ID:</strong> {bestModel.id}</p>
                    {bestModel.description && <p><strong>Description:</strong> {bestModel.description}</p>}
                    {selectedDatasetId && (
                        <div className="best-model-plots">
                            <h4>Model Visualization</h4>
                            <img 
                                src={getDatasetModelPlot(selectedDatasetId, bestModel.name || `Model_${bestModel.id}`, 'roc_curve')} 
                                alt="ROC Curve" 
                                className="metric-plot"
                                onError={(e) => {
                                    e.target.onerror = null;
                                    e.target.style.display = 'none';
                                    const placeholder = document.createElement('p');
                                    placeholder.textContent = 'Plot not available';
                                    e.target.parentElement.appendChild(placeholder);
                                }}
                            />
                        </div>
                    )}
                </Card>
            )}
            
            {models && models.length > 0 && (
                <div className="metrics-section">
                    <h3>Model Performance</h3>
                    <div className="metrics-grid">
                        {models.map((model) => (
                            <Card key={model.id} title={model.name || `Model ${model.id}`}>
                                <div className="model-info">
                                    <p><strong>ID:</strong> {model.id}</p>
                                    {model.description && <p><strong>Description:</strong> {model.description}</p>}
                                    {model.metrics?.accuracy !== undefined && (
                                        <p><strong>Accuracy:</strong> {(model.metrics.accuracy * 100).toFixed(2)}%</p>
                                    )}
                                    {selectedDatasetId && (
                                        <div className="model-plots">
                                            <h4>Plots</h4>
                                            {model.plots && Object.entries(model.plots).map(([key, url]) => (
                                                <p key={key}>{key}: <a href={url} target="_blank" rel="noreferrer">View</a></p>
                                            ))}
                                        </div>
                                    )}
                                </div>
                            </Card>
                        ))}
                    </div>
                </div>
            )}
            
            {(!models || models.length === 0) && !error && (
                <p>No models available for this dataset. Train models to see their performance.</p>
            )}

            {!selectedDatasetId && (
                <p>Please select a dataset to view model evaluations.</p>
            )}
        </div>
    );
};

export default ModelEvaluator;