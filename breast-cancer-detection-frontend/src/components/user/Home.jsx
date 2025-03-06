import React from 'react';
import { useAuth } from '../../hooks/useAuth'; // Make sure this matches the export type
import { Link } from 'react-router-dom';

const Home = () => {
    const { user } = useAuth();

    return (
        <div className="home-container">
            <h1>Welcome to the Breast Cancer Detection App</h1>
            {user ? (
                <div>
                    <h2>Hello, {user.username}!</h2>
                    <p>You can upload an image for prediction.</p>
                    <Link to="/predict" className="btn">Go to Prediction</Link>
                </div>
            ) : (
                <div>
                    <h2>Please log in to access the prediction features.</h2>
                    <Link to="/auth/login" className="btn">Login</Link>
                </div>
            )}
        </div>
    );
};

export default Home;