import React, { useState, useEffect } from 'react';
import { uploadDataset, fetchDatasets, deleteDataset } from '../../api/datasets';
import Alert from '../common/Alert';

const DatasetManager = () => {
    const [datasets, setDatasets] = useState([]);
    const [selectedFile, setSelectedFile] = useState(null);
    const [message, setMessage] = useState('');
    const [error, setError] = useState('');

    useEffect(() => {
        loadDatasets();
    }, []);

    const loadDatasets = async () => {
        try {
            const response = await fetchDatasets();
            setDatasets(response.data);
        } catch (err) {
            setError('Failed to load datasets');
        }
    };

    const handleFileChange = (event) => {
        setSelectedFile(event.target.files[0]);
    };

    const handleUpload = async () => {
        if (!selectedFile) {
            setError('Please select a file to upload');
            return;
        }

        const formData = new FormData();
        formData.append('file', selectedFile);

        try {
            await uploadDataset(formData);
            setMessage('Dataset uploaded successfully');
            setError('');
            loadDatasets();
        } catch (err) {
            setError('Failed to upload dataset');
        }
    };

    const handleDelete = async (id) => {
        try {
            await deleteDataset(id);
            setMessage('Dataset deleted successfully');
            loadDatasets();
        } catch (err) {
            setError('Failed to delete dataset');
        }
    };

    return (
        <div>
            <h2>Dataset Manager</h2>
            {message && <Alert message={message} type="success" />}
            {error && <Alert message={error} type="error" />}
            <input type="file" onChange={handleFileChange} />
            <button onClick={handleUpload}>Upload Dataset</button>
            <h3>Existing Datasets</h3>
            <ul>
                {datasets.map(dataset => (
                    <li key={dataset.id}>
                        {dataset.name}
                        <button onClick={() => handleDelete(dataset.id)}>Delete</button>
                    </li>
                ))}
            </ul>
        </div>
    );
};

export default DatasetManager;