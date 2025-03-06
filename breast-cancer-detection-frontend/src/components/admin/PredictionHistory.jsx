import React, { useEffect, useState } from 'react';
import { fetchPredictionHistory } from '../../api/predictions';
// Remove antd import if you don't have it installed
// import { Table } from 'antd';

const PredictionHistory = () => {
    const [predictions, setPredictions] = useState([]);
    const [loading, setLoading] = useState(true);

    useEffect(() => {
        const getPredictionHistory = async () => {
            try {
                const data = await fetchPredictionHistory();
                setPredictions(data);
            } catch (error) {
                console.error("Error fetching prediction history:", error);
            } finally {
                setLoading(false);
            }
        };

        getPredictionHistory();
    }, []);

    // Since Antd might not be installed, create a simple table
    return (
        <div>
            <h2>Prediction History</h2>
            {loading ? (
                <p>Loading...</p>
            ) : (
                <table className="predictions-table">
                    <thead>
                        <tr>
                            <th>Image</th>
                            <th>Prediction</th>
                            <th>Confidence</th>
                            <th>Date</th>
                        </tr>
                    </thead>
                    <tbody>
                        {predictions.map((prediction) => (
                            <tr key={prediction.id}>
                                <td>
                                    {prediction.image && (
                                        <img 
                                            src={prediction.image} 
                                            alt="Prediction" 
                                            style={{ width: 100 }} 
                                        />
                                    )}
                                </td>
                                <td>{prediction.prediction}</td>
                                <td>{prediction.confidence ? `${(prediction.confidence * 100).toFixed(2)}%` : 'N/A'}</td>
                                <td>{prediction.date}</td>
                            </tr>
                        ))}
                    </tbody>
                </table>
            )}
        </div>
    );
};

export default PredictionHistory;