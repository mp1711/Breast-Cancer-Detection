import apiClient from '../utils/apiClient';

const API_URL = import.meta.env.VITE_API_URL || 'http://localhost:8000/api';

export const predictWithDataset = async (datasetId, imageData) => {
    try {
        const formData = new FormData();
        formData.append('file', imageData);
        
        const response = await apiClient.post(`${API_URL}/predictions/datasets/${datasetId}/predict`, formData, {
            headers: {
                'Content-Type': 'multipart/form-data',
            },
        });
        return response.data;
    } catch (error) {
        console.error('Error making prediction:', error);
        throw error;
    }
};

export const getExplanationImage = (filename) => {
    return `${API_URL}/predictions/explanation/${filename}`;
};