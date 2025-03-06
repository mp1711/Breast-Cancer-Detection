import { useState, useEffect } from 'react';
import { useNavigate } from 'react-router-dom';
import { loginUser, registerUser, logoutUser } from '../api/auth';

export const useAuth = () => {
    const [user, setUser] = useState(null);
    const [loading, setLoading] = useState(true);
    const navigate = useNavigate();

    useEffect(() => {
        const storedUser = JSON.parse(localStorage.getItem('user'));
        if (storedUser) {
            setUser(storedUser);
        }
        setLoading(false);
    }, []);

    const login = async (credentials) => {
        const response = await loginUser(credentials);
        if (response.user) {
            setUser(response.user);
            localStorage.setItem('user', JSON.stringify(response.user));
            navigate('/'); // Redirect to home after login
        }
    };

    const register = async (userData) => {
        const response = await registerUser(userData);
        if (response.user) {
            setUser(response.user);
            localStorage.setItem('user', JSON.stringify(response.user));
            navigate('/'); // Redirect to home after registration
        }
    };

    const logout = () => {
        logoutUser();
        setUser(null);
        localStorage.removeItem('user');
        navigate('/auth/login'); // Redirect to login after logout
    };

    return { user, loading, login, register, logout };
};

export default useAuth; // Keep the default export for backward compatibility