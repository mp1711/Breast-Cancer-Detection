import React from 'react';
import { Link } from 'react-router-dom';
import './HomePage.css'; 

const HomePage = () => {
    return (
        <div className="home-container">
            <h1>Welcome to the Breast Cancer Detection App</h1>
            <p>
                This application allows you to upload images for breast cancer prediction using our trained models.
            </p>
            <div className="button-container">
                <Link to="/predict" className="button">
                    Upload Image for Prediction
                </Link>
            </div>
        </div>
    );
};

export default HomePage;