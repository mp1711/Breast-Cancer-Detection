import React from 'react';
import { useLocation } from 'react-router-dom';

const PredictionResult = () => {
    const location = useLocation();
    const { prediction, confidence, limeImage } = location.state || {};

    return (
        <div className="prediction-result">
            <h2>Prediction Result</h2>
            {prediction !== undefined ? (
                <div>
                    <p>
                        <strong>Prediction:</strong> {prediction ? 'Malignant' : 'Benign'}
                    </p>
                    <p>
                        <strong>Confidence:</strong> {Math.round(confidence * 100)}%
                    </p>
                    {limeImage && (
                        <div>
                            <h3>LIME Explanation</h3>
                            <img src={limeImage} alt="LIME Explanation" />
                        </div>
                    )}
                </div>
            ) : (
                <p>No prediction available. Please try again.</p>
            )}
        </div>
    );
};

export default PredictionResult;