import React, { useEffect, useState } from 'react';
import { fetchModels, deleteModel, uploadModel } from '../../api/models';
import Button from '../common/Button';
import Alert from '../common/Alert';

const ModelManager = () => {
    const [models, setModels] = useState([]);
    const [error, setError] = useState(null);
    const [success, setSuccess] = useState(null);
    const [loading, setLoading] = useState(true);
    const [selectedFile, setSelectedFile] = useState(null);

    useEffect(() => {
        const loadModels = async () => {
            try {
                const response = await fetchModels();
                setModels(response.data || []);
            } catch (err) {
                setError('Failed to fetch models');
            } finally {
                setLoading(false);
            }
        };
        loadModels();
    }, []);

    const handleFileChange = (event) => {
        setSelectedFile(event.target.files[0]);
    };

    const handleUpload = async () => {
        if (!selectedFile) {
            setError('Please select a model file to upload');
            return;
        }
        try {
            await uploadModel(selectedFile);
            setSuccess('Model uploaded successfully');
            setSelectedFile(null);
            // Refresh models list
            const data = await fetchModels();
            setModels(data);
        } catch (err) {
            setError('Failed to upload model');
        }
    };

    const handleDelete = async (modelId) => {
        try {
            await deleteModel(modelId);
            setSuccess('Model deleted successfully');
            // Refresh models list
            const data = await fetchModels();
            setModels(data);
        } catch (err) {
            setError('Failed to delete model');
        }
    };

    return (
        <div>
            <h2>Model Manager</h2>
            {loading && <p>Loading models...</p>}
            {error && <Alert message={error} type="error" />}
            {success && <Alert message={success} type="success" />}
            <input type="file" onChange={handleFileChange} />
            <Button onClick={handleUpload}>Upload Model</Button>
            <h3>Available Models</h3>
            <ul>
                {models.map((model) => (
                    <li key={model.id}>
                        {model.name}
                        <Button onClick={() => handleDelete(model.id)}>Delete</Button>
                    </li>
                ))}
            </ul>
        </div>
    );
};

export default ModelManager;