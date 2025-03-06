import React, { useState } from 'react';
import apiClient from '../../utils/apiClient'
import Alert from '../common/Alert';
import Button from '../common/Button';

const ImagePredictor = () => {
    const [selectedFile, setSelectedFile] = useState(null);
    const [prediction, setPrediction] = useState(null);
    const [loading, setLoading] = useState(false);
    const [error, setError] = useState(null);

    const handleFileChange = (event) => {
        setSelectedFile(event.target.files[0]);
    };

    const handleSubmit = async (event) => {
        event.preventDefault();
        if (!selectedFile) {
            setError('Please select an image to upload.');
            return;
        }

        setLoading(true);
        setError(null);

        const formData = new FormData();
        formData.append('file', selectedFile);

        try {
            const response = await apiClient.post('/api/predictions', formData, {
                headers: {
                    'Content-Type': 'multipart/form-data',
                },
            });
            setPrediction(response.data);
        } catch (err) {
            setError('Error uploading image. Please try again.');
        } finally {
            setLoading(false);
        }
    };

    return (
        <div>
            <h2>Image Predictor</h2>
            <form onSubmit={handleSubmit}>
                <input type="file" accept="image/*" onChange={handleFileChange} />
                <Button type="submit" disabled={loading}>
                    {loading ? 'Uploading...' : 'Upload Image'}
                </Button>
            </form>
            {error && <Alert message={error} />}
            {prediction && (
                <div>
                    <h3>Prediction Result</h3>
                    <p>{prediction.result}</p>
                    <img src={prediction.lime_explanation} alt="LIME Explanation" />
                </div>
            )}
        </div>
    );
};

export default ImagePredictor;