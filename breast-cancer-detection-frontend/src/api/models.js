import apiClient from '../utils/apiClient';

const API_URL = import.meta.env.VITE_API_URL || 'http://localhost:8000/api';

export const getDatasetModels = async (datasetId) => {
    try {
        const response = await apiClient.get(`${API_URL}/models/datasets/${datasetId}/models`);
        return response.data;
    } catch (error) {
        console.error(`Error fetching models for dataset ${datasetId}:`, error);
        throw error;
    }
};

export const getDatasetModel = async (datasetId, modelName) => {
    try {
        const response = await apiClient.get(`${API_URL}/models/datasets/${datasetId}/models/${modelName}`);
        return response.data;
    } catch (error) {
        console.error(`Error fetching model ${modelName} for dataset ${datasetId}:`, error);
        throw error;
    }
};

export const trainDatasetModels = async (datasetId) => {
    try {
        const response = await apiClient.post(`${API_URL}/models/datasets/${datasetId}/train`);
        return response.data;
    } catch (error) {
        console.error(`Error training models for dataset ${datasetId}:`, error);
        throw error;
    }
};

export const getDatasetBestModel = async (datasetId) => {
    try {
        const response = await apiClient.get(`${API_URL}/models/datasets/${datasetId}/best`);
        return response.data;
    } catch (error) {
        console.error(`Error fetching best model for dataset ${datasetId}:`, error);
        throw error;
    }
};

export const setDatasetBestModel = async (datasetId, modelName) => {
    try {
        const response = await apiClient.post(`${API_URL}/models/datasets/${datasetId}/models/${modelName}/set-as-best`);
        return response.data;
    } catch (error) {
        console.error(`Error setting best model for dataset ${datasetId}:`, error);
        throw error;
    }
};

export const getDatasetModelPlot = (datasetId, modelName, plotType) => {
    return `${API_URL}/models/datasets/${datasetId}/results/${modelName}/${plotType}`;
};