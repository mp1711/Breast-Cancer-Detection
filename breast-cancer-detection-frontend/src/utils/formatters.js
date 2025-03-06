const formatPredictionResult = (prediction) => {
    return prediction >= 0.5 ? 'Malignant' : 'Benign';
};

const formatConfidenceScore = (score) => {
    return `${(score * 100).toFixed(2)}%`;
};

const formatDate = (dateString) => {
    const options = { year: 'numeric', month: 'long', day: 'numeric' };
    return new Date(dateString).toLocaleDateString(undefined, options);
};

export { formatPredictionResult, formatConfidenceScore, formatDate };