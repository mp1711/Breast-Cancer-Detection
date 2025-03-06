import apiClient from '../utils/apiClient';

const API_URL = import.meta.env.VITE_API_URL || 'http://localhost:8000/api';

export const fetchDatasets = async (skip = 0, limit = 10) => {
    try {
        const response = await apiClient.get(`${API_URL}/datasets/`, {
            params: { skip, limit }
        });
        
        return response.data;
    } catch (error) {
        console.error('Error fetching datasets:', error);
        throw error;
    }
};

export const getDatasets = fetchDatasets;

export const deleteDataset = async (datasetId) => {
    try {
        const response = await apiClient.delete(`${API_URL}/datasets/${datasetId}`);
        return response.data;
    } catch (error) {
        console.error('Error deleting dataset:', error);
        throw error;
    }
};

export const uploadDataset = async (formData, description) => {
    try {
        const config = {};
        if (description) {
            config.params = { description };
        }

        const response = await apiClient.post(`${API_URL}/datasets/upload`, formData, {
            headers: {
                'Content-Type': 'multipart/form-data',
            },
            ...config
        });
        return response.data;
    } catch (error) {
        console.error('Error uploading dataset:', error);
        throw error;
    }
};