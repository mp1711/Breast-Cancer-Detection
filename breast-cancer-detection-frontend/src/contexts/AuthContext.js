import React, { createContext, useContext, useState, useEffect } from 'react';
import { useNavigate } from 'react-router-dom';
import { loginUser, registerUser, logoutUser } from '../api/auth';

const AuthContext = createContext();

export const AuthProvider = ({ children }) => {
    const [user, setUser] = useState(null);
    const [loading, setLoading] = useState(true);
    const navigate = useNavigate();
    
    useEffect(() => {
        const storedUser = JSON.parse(localStorage.getItem('user'));
        const token = localStorage.getItem('token');
        
        if (storedUser && token) {
            setUser(storedUser);
        }
        setLoading(false);
    }, []);

    const login = async (credentials) => {
        const response = await loginUser(credentials);
        if (response.token) {
            const userData = response.user || { token: response.token };
            setUser(userData);
            localStorage.setItem('token', response.token);
            localStorage.setItem('user', JSON.stringify(userData));
            navigate('/user/home'); // Redirect to home after login
        }
        return response;
    };

    const register = async (userData) => {
        const response = await registerUser(userData);
        if (response.token) {
            const user = response.user || { token: response.token };
            setUser(user);
            localStorage.setItem('token', response.token);
            localStorage.setItem('user', JSON.stringify(user));
            navigate('/user/home'); // Redirect to home after registration
        }
        return response;
    };

    const logout = () => {
        logoutUser();
        setUser(null);
        localStorage.removeItem('token');
        localStorage.removeItem('user');
        navigate('/auth/login'); // Redirect to login after logout
    };

    return (
        <AuthContext.Provider value={{ user, loading, login, register, logout }}>
            {children}
        </AuthContext.Provider>
    );
};

export const useAuth = () => {
    return useContext(AuthContext);
};