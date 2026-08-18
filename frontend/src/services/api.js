const BASE_URL = 'http://localhost:8000';

/**
 * Helper to perform HTTP GET requests.
 */
async function get(path) {
  try {
    const response = await fetch(`${BASE_URL}${path}`);
    if (!response.ok) {
      const errData = await response.json().catch(() => ({}));
      throw new Error(errData.detail || `HTTP error! status: ${response.status}`);
    }
    return await response.json();
  } catch (error) {
    console.error(`API GET error on ${path}:`, error);
    throw error;
  }
}

/**
 * Helper to perform HTTP POST requests.
 */
async function post(path, body) {
  try {
    const response = await fetch(`${BASE_URL}${path}`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify(body),
    });
    if (!response.ok) {
      const errData = await response.json().catch(() => ({}));
      throw new Error(errData.detail || `HTTP error! status: ${response.status}`);
    }
    return await response.json();
  } catch (error) {
    console.error(`API POST error on ${path}:`, error);
    throw error;
  }
}

export const api = {
  getHealth: () => get('/api/health'),
  getModelInfo: () => get('/api/model-info'),
  getMetrics: (modelKey) => get(`/api/metrics?model_key=${modelKey}`),
  getThresholdAnalysis: (modelKey) => get(`/api/threshold-analysis?model_key=${modelKey}`),
  getFeatureImportance: (modelKey) => get(`/api/feature-importance?model_key=${modelKey}`),
  getDataProfile: () => get('/api/data-profile'),
  predictRisk: (applicantData, modelKey = 'gradient_boosting', threshold = 0.5) => 
    post(`/api/predict?model_key=${modelKey}&threshold=${threshold}`, applicantData)
};
