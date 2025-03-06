import React, { useEffect, useState } from 'react';
import { fetchPredictions } from '../../api/predictions';
import Loading from '../../components/common/Loading';
import Card from '../../components/common/Card';

const PredictionsPage = () => {
    const [predictions, setPredictions] = useState([]);
    const [loading, setLoading] = useState(true);

    useEffect(() => {
        const fetchPredictionsData = async () => {
            try {
                const data = await fetchPredictions();
                setPredictions(data);
            } catch (error) {
                console.error("Error fetching predictions:", error);
            } finally {
                setLoading(false);
            }
        };

        fetchPredictionsData();
    }, []);

    if (loading) {
        return <Loading />;
    }

    return (
        <div>
            <h1>Predictions History</h1>
            <div className="predictions-list">
                {predictions.map((prediction) => (
                    <Card key={prediction.id} title={`Prediction ID: ${prediction.id}`}>
                        <p>Image: {prediction.image}</p>
                        <p>Result: {prediction.result}</p>
                        <p>Confidence: {prediction.confidence}%</p>
                        <p>Explanation: <a href={prediction.explanationUrl}>View LIME Explanation</a></p>
                    </Card>
                ))}
            </div>
        </div>
    );
};

export default PredictionsPage;