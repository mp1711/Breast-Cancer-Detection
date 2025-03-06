import axios from 'axios';

const API_URL = import.meta.env.VITE_API_URL || 'http://localhost:8000/api';

export const registerUser = async (userData) => {
    const response = await axios.post(`${API_URL}/auth/register`, userData);
    return response.data;
};

export const login = async (credentials) => {
    const response = await axios.post(`${API_URL}/auth/login`, {
        username: credentials.username,
        password: credentials.password
    });
    
    // Store token and user data
    if (response.data && response.data.access_token) {
        localStorage.setItem('access_token', response.data.access_token);
        const userData = {
            username: response.data.username,
            email: response.data.email,
            is_admin: response.data.is_admin
        };
        localStorage.setItem('user', JSON.stringify(userData));
    }
    
    return response.data;
};

export const logout = () => {
    localStorage.removeItem('access_token');
    localStorage.removeItem('user');
};

export const getCurrentUser = () => {
    const userStr = localStorage.getItem('user');
    if (!userStr) return null;
    
    try {
        return JSON.parse(userStr);
    } catch (e) {
        console.error('Error parsing user data from localStorage:', e);
        return null;
    }
};

export const getAuthHeaders = () => {
    const token = localStorage.getItem('access_token');
    return token ? { Authorization: `Bearer ${token}` } : {};
};

export const isAuthenticated = () => {
    return !!localStorage.getItem('access_token');
};

export const register = registerUser;
export const loginUser = login;
export const logoutUser = logout;