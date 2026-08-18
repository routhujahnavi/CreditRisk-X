import React, { useState, useEffect } from 'react';
import { api } from './services/api';
import { 
  LayoutDashboard, 
  Database, 
  BarChart3, 
  Sliders, 
  UserCheck, 
  HelpCircle, 
  Info,
  ShieldCheck,
  TrendingUp,
  AlertTriangle,
  RotateCcw,
  Sparkles
} from 'lucide-react';

export default function App() {
  // App states
  const [activeTab, setActiveTab] = useState('overview');
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);
  
  // Model & data states
  const [modelInfo, setModelInfo] = useState(null);
  const [selectedModel, setSelectedModel] = useState('logistic_regression');
  const [activeMetrics, setActiveMetrics] = useState(null);
  const [thresholdAnalysis, setThresholdAnalysis] = useState([]);
  const [dataProfile, setDataProfile] = useState(null);
  
  // Threshold slider (used in Threshold Analysis Tab)
  const [sliderThreshold, setSliderThreshold] = useState(0.5);
  
  // Simulator states
  const [simModel, setSimModel] = useState('logistic_regression');
  const [simThreshold, setSimThreshold] = useState(0.4);
  const [simLoading, setSimLoading] = useState(false);
  const [simError, setSimError] = useState(null);
  const [simResult, setSimResult] = useState(null);
  
  // Default values for Simulator Form
  const defaultApplicant = {
    age_years: 38.0,
    gender: 'M',
    children_count: 1,
    family_members_count: 3,
    family_status: 'Married',
    education_type: 'Higher education',
    housing_type: 'House / apartment',
    income_total: 210000.0,
    credit_amount: 540000.0,
    annuity_amount: 27000.0,
    goods_price: 500000.0,
    employment_status: 'Employed',
    employment_duration_years: 6.2,
    income_type: 'Working',
    occupation_type: 'Core staff',
    organization_type: 'Business Entity Type 3',
    owns_car: 'Y',
    owns_realty: 'Y',
    years_registration_change: 5.5,
    years_id_publish: 3.2,
    years_phone_change: 2.1,
    credit_bureau_queries_year: 1,
    region_rating: 2,
    ext_source_1: 0.55,
    ext_source_2: 0.62,
    ext_source_3: 0.48
  };
  const [simulatorInput, setSimulatorInput] = useState(defaultApplicant);

  // Initial load
  useEffect(() => {
    fetchInitialData();
  }, []);

  // Fetch metrics when selected model changes
  useEffect(() => {
    if (modelInfo) {
      fetchModelMetrics(selectedModel);
    }
  }, [selectedModel, modelInfo]);

  const fetchInitialData = async () => {
    setLoading(true);
    setError(null);
    try {
      // 1. Health check
      await api.getHealth();
      
      // 2. Fetch model info
      const info = await api.getModelInfo();
      setModelInfo(info);
      if (info.best_model) {
        setSelectedModel(info.best_model);
        setSimModel(info.best_model);
      }
      
      // 3. Fetch data profile
      const profile = await api.getDataProfile();
      setDataProfile(profile);
      
      setLoading(false);
    } catch (err) {
      console.error(err);
      setError("Failed to load initial data. Verify backend is running on port 8000 and data is trained.");
      setLoading(false);
    }
  };

  const fetchModelMetrics = async (modelKey) => {
    try {
      const metrics = await api.getMetrics(modelKey);
      setActiveMetrics(metrics);
      
      const analysis = await api.getThresholdAnalysis(modelKey);
      setThresholdAnalysis(analysis);
    } catch (err) {
      console.error("Error fetching model metrics:", err);
      setError("Failed to load metrics for model: " + modelKey);
    }
  };

  const handleSimulate = async (e) => {
    e.preventDefault();
    setSimLoading(true);
    setSimError(null);
    setSimResult(null);
    try {
      const result = await api.predictRisk(simulatorInput, simModel, simThreshold);
      setSimResult(result);
      setSimLoading(false);
    } catch (err) {
      console.error(err);
      setSimError(err.message || "Failed to generate risk prediction. Please review inputs.");
      setSimLoading(false);
    }
  };

  const handleInputChange = (field, val) => {
    let parsedVal = val;
    // Parse numeric fields correctly
    if (['age_years', 'income_total', 'credit_amount', 'annuity_amount', 'goods_price', 'employment_duration_years', 'years_registration_change', 'years_id_publish', 'years_phone_change', 'ext_source_1', 'ext_source_2', 'ext_source_3'].includes(field)) {
      parsedVal = val === '' ? '' : parseFloat(val);
    } else if (['children_count', 'family_members_count', 'credit_bureau_queries_year', 'region_rating'].includes(field)) {
      parsedVal = val === '' ? '' : parseInt(val, 10);
    }
    
    setSimulatorInput(prev => ({
      ...prev,
      [field]: parsedVal
    }));
  };

  // Helper to get threshold metrics dynamically
  const getThresholdMetrics = (t) => {
    const analysisRow = thresholdAnalysis.find(r => r.threshold === parseFloat(t.toFixed(2)));
    if (analysisRow) return analysisRow;
    
    // Fallback if exact round matches fail: find closest
    if (thresholdAnalysis.length > 0) {
      return thresholdAnalysis.reduce((prev, curr) => {
        return Math.abs(curr.threshold - t) < Math.abs(prev.threshold - t) ? curr : prev;
      });
    }
    return null;
  };

  if (loading) {
    return (
      <div style={{ display: 'flex', flexDirection: 'column', height: '100vh', justifyContent: 'center', alignItems: 'center', backgroundColor: '#09090b', color: '#a1a1aa', gap: '16px' }}>
        <div style={{ border: '3px solid rgba(255,255,255,0.08)', borderTop: '3px solid #3b82f6', borderRadius: '50%', width: '40px', height: '40px', animation: 'spin 1s linear infinite' }}></div>
        <p style={{ fontWeight: 500, fontSize: '0.95rem' }}>Loading CreditRisk-X Pipeline & Models...</p>
        <style dangerouslySetInnerHTML={{__html: `
          @keyframes spin { 0% { transform: rotate(0deg); } 100% { transform: rotate(360deg); } }
        `}} />
      </div>
    );
  }

  if (error) {
    return (
      <div style={{ padding: '40px', maxWidth: '600px', margin: '100px auto', background: '#18181b', border: '1px solid #ef4444', borderRadius: '12px', color: '#f4f4f5' }}>
        <h2 style={{ color: '#ef4444', display: 'flex', alignItems: 'center', gap: '10px' }}><AlertTriangle /> System Initialization Error</h2>
        <p style={{ margin: '16px 0', color: '#a1a1aa', fontSize: '0.95rem', lineHeight: '1.5' }}>{error}</p>
        <div style={{ display: 'flex', gap: '12px' }}>
          <button className="btn" onClick={fetchInitialData}>Retry Connection</button>
          <button className="btn btn-secondary" onClick={() => window.location.reload()}>Reload Page</button>
        </div>
      </div>
    );
  }

  // Active threshold metrics for Threshold view
  const currentTMetrics = getThresholdMetrics(sliderThreshold);
  const bestModelData = modelInfo && modelInfo.models_summary[modelInfo.best_model];

  return (
    <div className="app-container">
      {/* Sidebar Navigation */}
      <div className="sidebar">
        <div className="sidebar-brand">
          <div className="sidebar-logo">
            <ShieldCheck size={24} style={{ color: '#3b82f6' }} />
            CreditRisk-<span>X</span>
          </div>
        </div>
        
        <ul className="sidebar-menu">
          <li>
            <div className={`sidebar-item ${activeTab === 'overview' ? 'active' : ''}`} onClick={() => setActiveTab('overview')}>
              <LayoutDashboard size={18} />
              <span>Overview</span>
            </div>
          </li>
          <li>
            <div className={`sidebar-item ${activeTab === 'data-quality' ? 'active' : ''}`} onClick={() => setActiveTab('data-quality')}>
              <Database size={18} />
              <span>Data Quality</span>
            </div>
          </li>
          <li>
            <div className={`sidebar-item ${activeTab === 'performance' ? 'active' : ''}`} onClick={() => setActiveTab('performance')}>
              <BarChart3 size={18} />
              <span>Model Performance</span>
            </div>
          </li>
          <li>
            <div className={`sidebar-item ${activeTab === 'threshold-analysis' ? 'active' : ''}`} onClick={() => setActiveTab('threshold-analysis')}>
              <Sliders size={18} />
              <span>Threshold Optimizer</span>
            </div>
          </li>
          <li>
            <div className={`sidebar-item ${activeTab === 'simulator' ? 'active' : ''}`} onClick={() => setActiveTab('simulator')}>
              <UserCheck size={18} />
              <span>Risk Simulator</span>
            </div>
          </li>
          <li>
            <div className={`sidebar-item ${activeTab === 'explainability' ? 'active' : ''}`} onClick={() => setActiveTab('explainability')}>
              <Sparkles size={18} />
              <span>Model Explainability</span>
            </div>
          </li>
          <li>
            <div className={`sidebar-item ${activeTab === 'about' ? 'active' : ''}`} onClick={() => setActiveTab('about')}>
              <HelpCircle size={18} />
              <span>About the Model</span>
            </div>
          </li>
        </ul>
        
        <div className="sidebar-footer">
          <p>CreditRisk-X Platform</p>
          <p style={{ opacity: 0.5, marginTop: '4px' }}>Academic ML Pipeline v1.0</p>
        </div>
      </div>

      {/* Main Panel Content */}
      <div className="main-content">
        
        {/* TAB 1: OVERVIEW */}
        {activeTab === 'overview' && (
          <div>
            <div className="page-header">
              <h1>CreditRisk-X Overview</h1>
              <p className="page-subtitle">Interactive diagnostic dashboard for the explainable default risk assessment pipeline.</p>
            </div>

            {/* Overview Summary Cards */}
            <div className="grid-cols-4">
              <div className="card stat-card">
                <span className="stat-label">Dataset Size</span>
                <span className="stat-value">{dataProfile?.dataset_summary.num_rows.toLocaleString()}</span>
                <span className="stat-desc">Applicants in simulation</span>
              </div>
              <div className="card stat-card">
                <span className="stat-label">Default Rate</span>
                <span className="stat-value">{(dataProfile?.target_distribution["1_pct"])?.toFixed(2)}%</span>
                <span className="stat-desc">High risk class imbalance</span>
              </div>
              <div className="card stat-card">
                <span className="stat-label">Best Model</span>
                <span className="stat-value" style={{ fontSize: '1.25rem', padding: '6px 0', textTransform: 'capitalize', color: '#8b5cf6' }}>
                  {modelInfo?.best_model.replace('_', ' ')}
                </span>
                <span className="stat-desc">Determined by Test ROC-AUC</span>
              </div>
              <div className="card stat-card">
                <span className="stat-label">Best ROC-AUC</span>
                <span className="stat-value" style={{ color: '#10b981' }}>
                  {bestModelData?.test_auc.toFixed(4)}
                </span>
                <span className="stat-desc">Unbiased independent test score</span>
              </div>
            </div>

            {/* Main Overview Charts */}
            <div className="grid-cols-2">
              <div className="card">
                <h2>Target Label Distribution</h2>
                <p style={{ color: '#a1a1aa', fontSize: '0.85rem', marginBottom: '24px' }}>
                  Visualization of credit default class imbalance in the training data (TARGET = 1 indicates repayment difficulties).
                </p>
                <div style={{ marginTop: '20px' }}>
                  {/* Bar distribution */}
                  <div style={{ display: 'flex', flexDirection: 'column', gap: '20px' }}>
                    <div>
                      <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: '0.85rem', marginBottom: '6px' }}>
                        <span style={{ fontWeight: 600 }}>Repaid Loans (TARGET = 0)</span>
                        <span style={{ color: '#10b981', fontWeight: 600 }}>
                          {dataProfile?.target_distribution["0_count"].toLocaleString()} ({dataProfile?.target_distribution["0_pct"].toFixed(1)}%)
                        </span>
                      </div>
                      <div style={{ height: '24px', background: '#27272a', borderRadius: '4px', overflow: 'hidden' }}>
                        <div style={{ width: `${dataProfile?.target_distribution["0_pct"]}%`, height: '100%', background: '#10b981' }} />
                      </div>
                    </div>
                    <div>
                      <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: '0.85rem', marginBottom: '6px' }}>
                        <span style={{ fontWeight: 600 }}>Defaulted Loans (TARGET = 1)</span>
                        <span style={{ color: '#ef4444', fontWeight: 600 }}>
                          {dataProfile?.target_distribution["1_count"].toLocaleString()} ({dataProfile?.target_distribution["1_pct"].toFixed(1)}%)
                        </span>
                      </div>
                      <div style={{ height: '24px', background: '#27272a', borderRadius: '4px', overflow: 'hidden' }}>
                        <div style={{ width: `${dataProfile?.target_distribution["1_pct"]}%`, height: '100%', background: '#ef4444' }} />
                      </div>
                    </div>
                  </div>
                  <div className="alert alert-info" style={{ marginTop: '30px', marginBottom: 0 }}>
                    <Info size={16} style={{ float: 'left', marginRight: '10px', marginTop: '2px' }} />
                    <p style={{ marginLeft: '26px' }}>
                      <strong>Class Imbalance Impact:</strong> Credit default datasets are heavily imbalanced. In this cohort, roughly 1 in 10 applicants defaulted. Models optimizing pure Accuracy will predict TARGET = 0 for everyone. We use Stratified Cross-Validation and evaluate using ROC-AUC/PR-AUC to ensure risk is captured correctly.
                    </p>
                  </div>
                </div>
              </div>

              <div className="card">
                <h2>Model ROC Curve Comparison</h2>
                <p style={{ color: '#a1a1aa', fontSize: '0.85rem', marginBottom: '16px' }}>
                  Receiver Operating Characteristic comparison evaluated on the holdout test set.
                </p>
                <div style={{ display: 'flex', justifyContent: 'center' }}>
                  {modelInfo && activeMetrics && (
                    <SVGCurveCompare 
                      modelsSummary={modelInfo.models_summary}
                      activeModel={selectedModel}
                      activeMetrics={activeMetrics}
                      curveType="roc"
                    />
                  )}
                </div>
              </div>
            </div>

            {/* Quick Actions / Summary of models */}
            <div className="card">
              <h2>Active Model Pipelines</h2>
              <div className="table-container">
                <table className="custom-table">
                  <thead>
                    <tr>
                      <th>Pipeline Model</th>
                      <th>CV ROC-AUC Mean</th>
                      <th>CV Standard Deviation</th>
                      <th>Test ROC-AUC</th>
                      <th>Test PR-AUC</th>
                      <th>Status</th>
                    </tr>
                  </thead>
                  <tbody>
                    {Object.entries(modelInfo?.models_summary || {}).map(([key, val]) => (
                      <tr key={key} style={{ cursor: 'pointer' }} onClick={() => setSelectedModel(key)}>
                        <td style={{ fontWeight: 600, color: selectedModel === key ? '#3b82f6' : '#ffffff' }}>
                          {val.name} {selectedModel === key && ' (Active)'}
                        </td>
                        <td>{val.cv_mean.toFixed(4)}</td>
                        <td>± {val.cv_std.toFixed(4)}</td>
                        <td style={{ color: '#10b981', fontWeight: 600 }}>{val.test_auc.toFixed(4)}</td>
                        <td>{val.test_pr_auc.toFixed(4)}</td>
                        <td>
                          <span style={{
                            padding: '4px 8px',
                            borderRadius: '4px',
                            fontSize: '0.75rem',
                            fontWeight: 600,
                            background: modelInfo.best_model === key ? 'rgba(16, 185, 129, 0.1)' : 'rgba(255,255,255,0.05)',
                            color: modelInfo.best_model === key ? '#10b981' : '#a1a1aa'
                          }}>
                            {modelInfo.best_model === key ? 'Best Model' : 'Baseline'}
                          </span>
                        </td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </div>
          </div>
        )}

        {/* TAB 2: DATA QUALITY */}
        {activeTab === 'data-quality' && (
          <div>
            <div className="page-header">
              <h1>Dataset Profiling & Quality Audit</h1>
              <p className="page-subtitle">Automated profiling report checking missingness, cardinalities, and demographic properties.</p>
            </div>

            <div className="grid-cols-3" style={{ marginBottom: '30px' }}>
              <div className="card stat-card">
                <span className="stat-label">Total Features</span>
                <span className="stat-value">{dataProfile?.dataset_summary.numerical_features_count + dataProfile?.dataset_summary.categorical_features_count}</span>
                <span className="stat-desc">{dataProfile?.dataset_summary.numerical_features_count} numerical, {dataProfile?.dataset_summary.categorical_features_count} categorical</span>
              </div>
              <div className="card stat-card">
                <span className="stat-label">Duplicate Rows</span>
                <span className="stat-value">{dataProfile?.dataset_summary.duplicate_rows}</span>
                <span className="stat-desc">Zero replication leakage detected</span>
              </div>
              <div className="card stat-card">
                <span className="stat-label">Constant Features</span>
                <span className="stat-value">{dataProfile?.dataset_summary.constant_features.length}</span>
                <span className="stat-desc">No zero-variance features dropped</span>
              </div>
            </div>

            <div className="grid-cols-2">
              {/* Left Column: Missing Values Table */}
              <div className="card">
                <h2>Missing Value Rates (Imputation Audit)</h2>
                <p style={{ color: '#a1a1aa', fontSize: '0.85rem', marginBottom: '20px' }}>
                  Missingness rate sorted descending. Preprocessing pipeline imputes numerical values with column medians, and categorical values with most-frequent.
                </p>
                <div className="table-container" style={{ maxHeight: '420px', overflowY: 'auto' }}>
                  <table className="custom-table">
                    <thead>
                      <tr>
                        <th>Feature Name</th>
                        <th>Missing Count</th>
                        <th>Missing Rate</th>
                        <th>Imputation strategy</th>
                      </tr>
                    </thead>
                    <tbody>
                      {dataProfile?.missing_values.map((row) => (
                        <tr key={row.column}>
                          <td style={{ fontFamily: 'monospace', fontSize: '0.8rem' }}>{row.column}</td>
                          <td>{row.count.toLocaleString()}</td>
                          <td>
                            <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
                              <span style={{ width: '40px', textAlign: 'right', fontWeight: 600, color: row.pct > 20 ? '#f59e0b' : '#ffffff' }}>
                                {row.pct.toFixed(1)}%
                              </span>
                              <div style={{ flex: 1, height: '6px', background: '#27272a', borderRadius: '3px', width: '80px', overflow: 'hidden' }}>
                                <div style={{ 
                                  width: `${row.pct}%`, 
                                  height: '100%', 
                                  background: row.pct > 50 ? '#ef4444' : row.pct > 20 ? '#f59e0b' : '#3b82f6' 
                                }} />
                              </div>
                            </div>
                          </td>
                          <td style={{ fontSize: '0.75rem', opacity: 0.8 }}>
                            {row.column === 'TARGET' || row.column === 'SK_ID_CURR' ? 'None' :
                             ['object', 'category'].includes(row.type) ? 'Most Frequent' : 'Median'}
                          </td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              </div>

              {/* Right Column: Key Feature Statistics */}
              <div className="card">
                <h2>Continuous Feature Profile</h2>
                <p style={{ color: '#a1a1aa', fontSize: '0.85rem', marginBottom: '20px' }}>
                  Summary statistics for key applicant financials and credit scores.
                </p>
                <div className="table-container" style={{ maxHeight: '420px', overflowY: 'auto' }}>
                  <table className="custom-table">
                    <thead>
                      <tr>
                        <th>Feature</th>
                        <th>Min</th>
                        <th>Median</th>
                        <th>Max</th>
                        <th>Corr with Target</th>
                      </tr>
                    </thead>
                    <tbody>
                      {Object.entries(dataProfile?.numerical_features || {}).map(([key, val]) => (
                        <tr key={key}>
                          <td style={{ fontFamily: 'monospace', fontSize: '0.8rem', fontWeight: 600 }}>{key}</td>
                          <td>{val.min >= 1000 ? `$${(val.min/1000).toFixed(0)}k` : val.min.toFixed(2)}</td>
                          <td>{val.median >= 1000 ? `$${(val.median/1000).toFixed(0)}k` : val.median.toFixed(2)}</td>
                          <td>{val.max >= 1000 ? `$${(val.max/1000).toFixed(0)}k` : val.max.toFixed(2)}</td>
                          <td style={{ 
                            fontWeight: 600, 
                            color: val.correlation_with_target < -0.1 ? '#10b981' : val.correlation_with_target > 0.05 ? '#ef4444' : '#ffffff' 
                          }}>
                            {val.correlation_with_target.toFixed(4)}
                          </td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              </div>
            </div>
          </div>
        )}

        {/* TAB 3: MODEL PERFORMANCE */}
        {activeTab === 'performance' && (
          <div>
            <div className="page-header">
              <h1>Model Performance Evaluation</h1>
              <p className="page-subtitle">Unbiased comparison of baseline Linear Model vs Tree-based Nonlinear Model.</p>
            </div>

            {/* Model select dropdown */}
            <div className="card" style={{ marginBottom: '24px', display: 'flex', justifyContent: 'space-between', alignItems: 'center', padding: '16px 24px' }}>
              <div style={{ display: 'flex', alignItems: 'center', gap: '12px' }}>
                <span style={{ fontSize: '0.9rem', fontWeight: 600, color: 'var(--color-text-muted)' }}>Analyze Model metrics:</span>
                <select 
                  className="form-control" 
                  style={{ width: '280px', background: '#27272a' }}
                  value={selectedModel}
                  onChange={(e) => setSelectedModel(e.target.value)}
                >
                  <option value="logistic_regression">Logistic Regression (Baseline)</option>
                  <option value="gradient_boosting">HistGradientBoosting (Nonlinear)</option>
                </select>
              </div>
              <div style={{ fontSize: '0.85rem', color: '#a1a1aa' }}>
                Currently displaying: <strong style={{ color: '#ffffff' }}>{activeMetrics?.name}</strong>
              </div>
            </div>

            {/* Metric Comparison Cards */}
            <div className="grid-cols-3">
              <div className="card stat-card" style={{ borderLeft: '4px solid #3b82f6' }}>
                <span className="stat-label">Holdout Test ROC-AUC</span>
                <span className="stat-value" style={{ color: '#3b82f6' }}>{activeMetrics?.test_auc.toFixed(4)}</span>
                <span className="stat-desc">Area under ROC Curve</span>
              </div>
              <div className="card stat-card" style={{ borderLeft: '4px solid #8b5cf6' }}>
                <span className="stat-label">Holdout Test PR-AUC</span>
                <span className="stat-value" style={{ color: '#8b5cf6' }}>{activeMetrics?.test_pr_auc.toFixed(4)}</span>
                <span className="stat-desc">Area under Precision-Recall</span>
              </div>
              <div className="card stat-card" style={{ borderLeft: '4px solid #10b981' }}>
                <span className="stat-label">CV Stratified Mean</span>
                <span className="stat-value" style={{ color: '#10b981' }}>{activeMetrics?.cv_mean.toFixed(4)}</span>
                <span className="stat-desc">Out-of-fold score ± {activeMetrics?.cv_std.toFixed(4)}</span>
              </div>
            </div>

            {/* ROC and PR curves */}
            <div className="grid-cols-2">
              <div className="card">
                <h2>ROC Curve (FPR vs TPR)</h2>
                <p style={{ color: '#a1a1aa', fontSize: '0.85rem', marginBottom: '20px' }}>
                  Visualizes TPR (sensitivity) vs FPR (1 - specificity). Higher area (towards top-left) represents better default discrimination.
                </p>
                <div style={{ display: 'flex', justifyContent: 'center' }}>
                  {modelInfo && activeMetrics && (
                    <SVGCurveCompare 
                      modelsSummary={modelInfo.models_summary}
                      activeModel={selectedModel}
                      activeMetrics={activeMetrics}
                      curveType="roc"
                    />
                  )}
                </div>
              </div>

              <div className="card">
                <h2>Precision-Recall Curve</h2>
                <p style={{ color: '#a1a1aa', fontSize: '0.85rem', marginBottom: '20px' }}>
                  Visualizes Precision vs Recall. Essential for highly imbalanced classifications (like default credit risk), showing trade-off without TN.
                </p>
                <div style={{ display: 'flex', justifyContent: 'center' }}>
                  {modelInfo && activeMetrics && (
                    <SVGCurveCompare 
                      modelsSummary={modelInfo.models_summary}
                      activeModel={selectedModel}
                      activeMetrics={activeMetrics}
                      curveType="pr"
                    />
                  )}
                </div>
              </div>
            </div>

            {/* Fold-wise cross validation logs */}
            <div className="card">
              <h2>Stratified 5-Fold Cross-Validation Logs</h2>
              <p style={{ color: '#a1a1aa', fontSize: '0.85rem', marginBottom: '16px' }}>
                Fold-wise scoring report. StratifiedKFold maintains the same target default distribution across all validation folds, preventing leakage and optimistic bias.
              </p>
              <div style={{ display: 'flex', gap: '12px', flexWrap: 'wrap', marginBottom: '20px' }}>
                {activeMetrics?.cv_scores.map((score, idx) => (
                  <div key={idx} style={{ flex: 1, minWidth: '140px', background: '#18181b', padding: '16px', borderRadius: '8px', border: '1px solid var(--border-color)', textAlign: 'center' }}>
                    <div style={{ fontSize: '0.75rem', color: '#a1a1aa', fontWeight: 600, textTransform: 'uppercase', marginBottom: '6px' }}>Fold {idx+1}</div>
                    <div style={{ fontSize: '1.25rem', fontWeight: 700, color: '#3b82f6' }}>{score.toFixed(4)}</div>
                  </div>
                ))}
              </div>
            </div>
          </div>
        )}

        {/* TAB 4: THRESHOLD ANALYSIS */}
        {activeTab === 'threshold-analysis' && (
          <div>
            <div className="page-header">
              <h1>Decision Threshold Optimization</h1>
              <p className="page-subtitle">Configure model probability thresholds to optimize precision, recall, and banking cost trade-offs.</p>
            </div>

            <div className="card" style={{ marginBottom: '30px' }}>
              <h2>Select Classification Threshold</h2>
              <p style={{ color: '#a1a1aa', fontSize: '0.85rem', marginBottom: '20px' }}>
                The model outputs a probability of credit default (0.0 to 1.0). Drag the slider to set the risk cut-off. 
                Applicants with a predicted probability equal or greater than the threshold are classified as <strong>HIGH RISK (Denied)</strong>.
              </p>
              
              {/* Slider controls */}
              <div style={{ background: '#18181b', padding: '24px', borderRadius: '8px', border: '1px solid var(--border-color)' }}>
                <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '12px' }}>
                  <span style={{ fontSize: '0.9rem', fontWeight: 600 }}>Decision Cut-off Threshold:</span>
                  <span style={{ fontSize: '1.75rem', fontWeight: 800, color: '#3b82f6' }}>
                    {(sliderThreshold * 100).toFixed(0)}%
                  </span>
                </div>
                
                <input 
                  type="range"
                  min="0.10"
                  max="0.90"
                  step="0.05"
                  className="threshold-slider"
                  value={sliderThreshold}
                  onChange={(e) => setSliderThreshold(parseFloat(e.target.value))}
                />
                <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: '0.75rem', color: '#a1a1aa', marginTop: '8px' }}>
                  <span>0.10 (Highly Conservative / High Denial Rate)</span>
                  <span>0.50 (Standard Cut-off)</span>
                  <span>0.90 (Aggressive / High Default Exposure)</span>
                </div>
              </div>
            </div>

            {/* Interactive metrics based on slider */}
            {currentTMetrics && (
              <div style={{ display: 'flex', flexDirection: 'column', gap: '24px' }}>
                
                <div className="grid-cols-4">
                  <div className="card stat-card" style={{ borderLeft: '4px solid #10b981' }}>
                    <span className="stat-label">Model Precision</span>
                    <span className="stat-value" style={{ color: '#10b981' }}>{(currentTMetrics.precision * 100).toFixed(1)}%</span>
                    <span className="stat-desc">Of predicted defaults, how many actually defaulted</span>
                  </div>
                  <div className="card stat-card" style={{ borderLeft: '4px solid #3b82f6' }}>
                    <span className="stat-label">Model Recall (Sensitivity)</span>
                    <span className="stat-value" style={{ color: '#3b82f6' }}>{(currentTMetrics.recall * 100).toFixed(1)}%</span>
                    <span className="stat-desc">How many defaults the model successfully caught</span>
                  </div>
                  <div className="card stat-card" style={{ borderLeft: '4px solid #8b5cf6' }}>
                    <span className="stat-label">F1-Score</span>
                    <span className="stat-value" style={{ color: '#8b5cf6' }}>{(currentTMetrics.f1_score * 100).toFixed(1)}%</span>
                    <span className="stat-desc">Balanced harmonic mean</span>
                  </div>
                  <div className="card stat-card" style={{ borderLeft: '4px solid #f59e0b' }}>
                    <span className="stat-label">Predicted High-Risk</span>
                    <span className="stat-value" style={{ color: '#f59e0b' }}>{currentTMetrics.predicted_high_risk_pct.toFixed(1)}%</span>
                    <span className="stat-desc">Percentage of applications denied credit</span>
                  </div>
                </div>

                <div className="grid-cols-2">
                  {/* Confusion Matrix */}
                  <div className="card">
                    <h2>Confusion Matrix at {(sliderThreshold * 100).toFixed(0)}% Cut-off</h2>
                    <p style={{ color: '#a1a1aa', fontSize: '0.85rem', marginBottom: '24px' }}>
                      Renders the distribution of predictions vs actual default occurrences on the holdout test set (2,000 cases).
                    </p>
                    
                    <div style={{ display: 'grid', gridTemplateColumns: '120px 1fr 1fr', gap: '8px', textAlign: 'center', fontSize: '0.85rem' }}>
                      <div></div>
                      <div style={{ fontWeight: 600, color: '#10b981', padding: '8px 0', borderBottom: '1px solid var(--border-color)' }}>Actual Non-Default (0)</div>
                      <div style={{ fontWeight: 600, color: '#ef4444', padding: '8px 0', borderBottom: '1px solid var(--border-color)' }}>Actual Default (1)</div>
                      
                      <div style={{ fontWeight: 600, textAlign: 'right', display: 'flex', alignItems: 'center', justifyContent: 'flex-end', paddingRight: '8px' }}>
                        Predicted Low-Risk (0)
                      </div>
                      <div style={{ background: 'rgba(16, 185, 129, 0.1)', border: '1px solid rgba(16, 185, 129, 0.2)', padding: '24px 12px', borderRadius: '6px' }}>
                        <div style={{ fontSize: '1.5rem', fontWeight: 700, color: '#10b981' }}>{currentTMetrics.tn}</div>
                        <span style={{ fontSize: '0.7rem', color: '#a1a1aa', textTransform: 'uppercase' }}>True Negatives</span>
                      </div>
                      <div style={{ background: 'rgba(239, 68, 68, 0.1)', border: '1px solid rgba(239, 68, 68, 0.2)', padding: '24px 12px', borderRadius: '6px' }}>
                        <div style={{ fontSize: '1.5rem', fontWeight: 700, color: '#ef4444' }}>{currentTMetrics.fn}</div>
                        <span style={{ fontSize: '0.7rem', color: '#a1a1aa', textTransform: 'uppercase' }}>False Negatives</span>
                      </div>
                      
                      <div style={{ fontWeight: 600, textAlign: 'right', display: 'flex', alignItems: 'center', justifyContent: 'flex-end', paddingRight: '8px' }}>
                        Predicted High-Risk (1)
                      </div>
                      <div style={{ background: 'rgba(245, 158, 11, 0.1)', border: '1px solid rgba(245, 158, 11, 0.2)', padding: '24px 12px', borderRadius: '6px' }}>
                        <div style={{ fontSize: '1.5rem', fontWeight: 700, color: '#f59e0b' }}>{currentTMetrics.fp}</div>
                        <span style={{ fontSize: '0.7rem', color: '#a1a1aa', textTransform: 'uppercase' }}>False Positives</span>
                      </div>
                      <div style={{ background: 'rgba(16, 185, 129, 0.2)', border: '1px solid rgba(16, 185, 129, 0.3)', padding: '24px 12px', borderRadius: '6px' }}>
                        <div style={{ fontSize: '1.5rem', fontWeight: 700, color: '#10b981' }}>{currentTMetrics.tp}</div>
                        <span style={{ fontSize: '0.7rem', color: '#a1a1aa', textTransform: 'uppercase' }}>True Positives</span>
                      </div>
                    </div>
                  </div>

                  {/* Business Trade-off Chart */}
                  <div className="card">
                    <h2>Threshold vs Performance Metrics</h2>
                    <p style={{ color: '#a1a1aa', fontSize: '0.85rem', marginBottom: '16px' }}>
                      Graph showing Precision (Green), Recall (Blue), and F1-Score (Purple) across all thresholds.
                    </p>
                    <div style={{ display: 'flex', justifyContent: 'center' }}>
                      <SVGThresholdChart 
                        thresholdAnalysis={thresholdAnalysis}
                        currentThreshold={sliderThreshold}
                      />
                    </div>
                  </div>
                </div>

                {/* Business cost explanation alert */}
                <div className="card" style={{ background: 'rgba(255,255,255,0.02)' }}>
                  <h2>Credit Risk Decision Strategy</h2>
                  <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '24px', marginTop: '16px' }}>
                    <div>
                      <h3 style={{ color: '#3b82f6', display: 'flex', alignItems: 'center', gap: '8px' }}>
                        <ShieldCheck size={16} /> Option A: Low Threshold (e.g. 0.20)
                      </h3>
                      <p style={{ color: '#a1a1aa', fontSize: '0.85rem', lineHeight: '1.45' }}>
                        <strong>Risk-Averse:</strong> Denies credit to anyone with even a slight chance of default. 
                        Result: Low default rates (fewer False Negatives), but high denial rates (many False Positives - losing creditworthy customers and loan interest).
                      </p>
                    </div>
                    <div>
                      <h3 style={{ color: '#f59e0b', display: 'flex', alignItems: 'center', gap: '8px' }}>
                        <TrendingUp size={16} /> Option B: High Threshold (e.g. 0.70)
                      </h3>
                      <p style={{ color: '#a1a1aa', fontSize: '0.85rem', lineHeight: '1.45' }}>
                        <strong>Aggressive Growth:</strong> Approves almost all applicants except those with extreme risk.
                        Result: Maximizes approved customer base, but exposes the bank to high write-offs due to unpaid defaults (many False Negatives).
                      </p>
                    </div>
                  </div>
                </div>
              </div>
            )}
          </div>
        )}

        {/* TAB 5: RISK SIMULATOR */}
        {activeTab === 'simulator' && (
          <div>
            <div className="page-header">
              <h1>Credit Default Risk Simulator</h1>
              <p className="page-subtitle">Input credit application details to evaluate risk, classify probability, and view model explanations.</p>
            </div>

            <div style={{ display: 'grid', gridTemplateColumns: '1fr 1.1fr', gap: '30px' }}>
              {/* Left Column: Input Form */}
              <div className="card" style={{ maxHeight: '85vh', overflowY: 'auto' }}>
                <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '20px' }}>
                  <h2>Applicant Credentials</h2>
                  <button 
                    type="button" 
                    className="btn btn-secondary" 
                    style={{ padding: '6px 12px', fontSize: '0.8rem', gap: '6px' }}
                    onClick={() => {
                      setSimulatorInput(defaultApplicant);
                      setSimResult(null);
                    }}
                  >
                    <RotateCcw size={14} /> Reset Form
                  </button>
                </div>
                
                <form onSubmit={handleSimulate}>
                  {/* Model & Threshold selector */}
                  <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '16px', background: '#18181b', padding: '16px', borderRadius: '8px', marginBottom: '20px', border: '1px solid var(--border-color)' }}>
                    <div className="form-group" style={{ marginBottom: 0 }}>
                      <label className="form-label">ML Estimator Model</label>
                      <select 
                        className="form-control"
                        value={simModel}
                        onChange={(e) => setSimModel(e.target.value)}
                      >
                        <option value="logistic_regression">Logistic Regression (Baseline)</option>
                        <option value="gradient_boosting">HistGradientBoosting (Tree-based)</option>
                      </select>
                    </div>
                    
                    <div className="form-group" style={{ marginBottom: 0 }}>
                      <label className="form-label">Decision Threshold: {(simThreshold * 100).toFixed(0)}%</label>
                      <input 
                        type="range"
                        min="0.10"
                        max="0.90"
                        step="0.05"
                        style={{ height: '6px', margin: '14px 0' }}
                        className="threshold-slider"
                        value={simThreshold}
                        onChange={(e) => setSimThreshold(parseFloat(e.target.value))}
                      />
                    </div>
                  </div>

                  {/* Demographics Block */}
                  <h3 style={{ borderBottom: '1px solid var(--border-color)', paddingBottom: '6px', marginBottom: '16px', color: '#3b82f6', fontSize: '0.9rem' }}>
                    1. Applicant Demographics
                  </h3>
                  
                  <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '16px' }}>
                    <div className="form-group">
                      <label className="form-label">Age (in Years)</label>
                      <input 
                        type="number" 
                        className="form-control" 
                        required
                        value={simulatorInput.age_years}
                        onChange={(e) => handleInputChange('age_years', e.target.value)}
                      />
                    </div>
                    <div className="form-group">
                      <label className="form-label">Gender</label>
                      <select 
                        className="form-control"
                        value={simulatorInput.gender}
                        onChange={(e) => handleInputChange('gender', e.target.value)}
                      >
                        <option value="F">Female (F)</option>
                        <option value="M">Male (M)</option>
                      </select>
                    </div>
                  </div>

                  <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '16px' }}>
                    <div className="form-group">
                      <label className="form-label">Children Count</label>
                      <input 
                        type="number" 
                        className="form-control"
                        required
                        value={simulatorInput.children_count}
                        onChange={(e) => handleInputChange('children_count', e.target.value)}
                      />
                    </div>
                    <div className="form-group">
                      <label className="form-label">Family Members Count</label>
                      <input 
                        type="number" 
                        className="form-control"
                        required
                        value={simulatorInput.family_members_count}
                        onChange={(e) => handleInputChange('family_members_count', e.target.value)}
                      />
                    </div>
                  </div>

                  <div className="form-group">
                    <label className="form-label">Education Type</label>
                    <select 
                      className="form-control"
                      value={simulatorInput.education_type}
                      onChange={(e) => handleInputChange('education_type', e.target.value)}
                    >
                      <option value="Secondary / special education">Secondary / special education</option>
                      <option value="Higher education">Higher education</option>
                      <option value="Incomplete higher">Incomplete higher</option>
                      <option value="Lower secondary">Lower secondary</option>
                      <option value="Academic degree">Academic degree</option>
                    </select>
                  </div>

                  <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '16px' }}>
                    <div className="form-group">
                      <label className="form-label">Family Status</label>
                      <select 
                        className="form-control"
                        value={simulatorInput.family_status}
                        onChange={(e) => handleInputChange('family_status', e.target.value)}
                      >
                        <option value="Married">Married</option>
                        <option value="Single / not married">Single / not married</option>
                        <option value="Civil marriage">Civil marriage</option>
                        <option value="Separated">Separated</option>
                        <option value="Widow">Widow</option>
                      </select>
                    </div>
                    <div className="form-group">
                      <label className="form-label">Housing Situation</label>
                      <select 
                        className="form-control"
                        value={simulatorInput.housing_type}
                        onChange={(e) => handleInputChange('housing_type', e.target.value)}
                      >
                        <option value="House / apartment">House / apartment</option>
                        <option value="With parents">With parents</option>
                        <option value="Municipal apartment">Municipal apartment</option>
                        <option value="Rented apartment">Rented apartment</option>
                        <option value="Office apartment">Office apartment</option>
                        <option value="Co-op apartment">Co-op apartment</option>
                      </select>
                    </div>
                  </div>

                  {/* Financials Block */}
                  <h3 style={{ borderBottom: '1px solid var(--border-color)', paddingBottom: '6px', marginTop: '20px', marginBottom: '16px', color: '#3b82f6', fontSize: '0.9rem' }}>
                    2. Financial Status & Assets
                  </h3>
                  
                  <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '16px' }}>
                    <div className="form-group">
                      <label className="form-label">Total Annual Income ($)</label>
                      <input 
                        type="number" 
                        className="form-control"
                        required
                        value={simulatorInput.income_total}
                        onChange={(e) => handleInputChange('income_total', e.target.value)}
                      />
                    </div>
                    <div className="form-group">
                      <label className="form-label">Requested Credit ($)</label>
                      <input 
                        type="number" 
                        className="form-control"
                        required
                        value={simulatorInput.credit_amount}
                        onChange={(e) => handleInputChange('credit_amount', e.target.value)}
                      />
                    </div>
                  </div>

                  <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '16px' }}>
                    <div className="form-group">
                      <label className="form-label">Annual Annuity Payment ($)</label>
                      <input 
                        type="number" 
                        className="form-control"
                        required
                        value={simulatorInput.annuity_amount}
                        onChange={(e) => handleInputChange('annuity_amount', e.target.value)}
                      />
                    </div>
                    <div className="form-group">
                      <label className="form-label">Goods Price ($)</label>
                      <input 
                        type="number" 
                        className="form-control"
                        required
                        value={simulatorInput.goods_price}
                        onChange={(e) => handleInputChange('goods_price', e.target.value)}
                      />
                    </div>
                  </div>

                  <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '16px' }}>
                    <div className="form-group">
                      <label className="form-label">Owns Car</label>
                      <select 
                        className="form-control"
                        value={simulatorInput.owns_car}
                        onChange={(e) => handleInputChange('owns_car', e.target.value)}
                      >
                        <option value="N">No (N)</option>
                        <option value="Y">Yes (Y)</option>
                      </select>
                    </div>
                    <div className="form-group">
                      <label className="form-label">Owns Real Estate</label>
                      <select 
                        className="form-control"
                        value={simulatorInput.owns_realty}
                        onChange={(e) => handleInputChange('owns_realty', e.target.value)}
                      >
                        <option value="Y">Yes (Y)</option>
                        <option value="N">No (N)</option>
                      </select>
                    </div>
                  </div>

                  {/* Employment Block */}
                  <h3 style={{ borderBottom: '1px solid var(--border-color)', paddingBottom: '6px', marginTop: '20px', marginBottom: '16px', color: '#3b82f6', fontSize: '0.9rem' }}>
                    3. Occupation & Employment
                  </h3>

                  <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '16px' }}>
                    <div className="form-group">
                      <label className="form-label">Employment Status</label>
                      <select 
                        className="form-control"
                        value={simulatorInput.employment_status}
                        onChange={(e) => handleInputChange('employment_status', e.target.value)}
                      >
                        <option value="Employed">Employed</option>
                        <option value="Pensioner">Pensioner</option>
                        <option value="Unemployed">Unemployed</option>
                      </select>
                    </div>
                    <div className="form-group">
                      <label className="form-label">Years in Current Job</label>
                      <input 
                        type="number" 
                        step="0.1"
                        className="form-control"
                        disabled={simulatorInput.employment_status !== 'Employed'}
                        value={simulatorInput.employment_status === 'Employed' ? simulatorInput.employment_duration_years : 0.0}
                        onChange={(e) => handleInputChange('employment_duration_years', e.target.value)}
                      />
                    </div>
                  </div>

                  <div className="form-group">
                    <label className="form-label">Income Source Type</label>
                    <select 
                      className="form-control"
                      value={simulatorInput.income_type}
                      onChange={(e) => handleInputChange('income_type', e.target.value)}
                    >
                      <option value="Working">Working</option>
                      <option value="Commercial associate">Commercial associate</option>
                      <option value="Pensioner">Pensioner</option>
                      <option value="State servant">State servant</option>
                    </select>
                  </div>

                  <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '16px' }}>
                    <div className="form-group">
                      <label className="form-label">Occupation Category</label>
                      <select 
                        className="form-control"
                        value={simulatorInput.occupation_type || ''}
                        onChange={(e) => handleInputChange('occupation_type', e.target.value || null)}
                      >
                        <option value="">(None / Unspecified)</option>
                        <option value="Laborers">Laborers</option>
                        <option value="Sales staff">Sales staff</option>
                        <option value="Core staff">Core staff</option>
                        <option value="Managers">Managers</option>
                        <option value="Drivers">Drivers</option>
                        <option value="High skill tech staff">High skill tech staff</option>
                        <option value="Accountants">Accountants</option>
                        <option value="Medicine staff">Medicine staff</option>
                        <option value="Security staff">Security staff</option>
                      </select>
                    </div>
                    <div className="form-group">
                      <label className="form-label">Organization Category</label>
                      <select 
                        className="form-control"
                        value={simulatorInput.organization_type}
                        onChange={(e) => handleInputChange('organization_type', e.target.value)}
                      >
                        <option value="Business Entity Type 3">Business Entity Type 3</option>
                        <option value="Self-employed">Self-employed</option>
                        <option value="Other">Other</option>
                        <option value="Medicine">Medicine</option>
                        <option value="Government">Government</option>
                        <option value="School">School</option>
                        <option value="Trade: type 7">Trade: type 7</option>
                        <option value="Construction">Construction</option>
                        <option value="Kindergarten">Kindergarten</option>
                        <option value="Industry: type 9">Industry: type 9</option>
                        <option value="XNA">Pensioner / Unemployed (XNA)</option>
                      </select>
                    </div>
                  </div>

                  {/* External Credit Ratings */}
                  <h3 style={{ borderBottom: '1px solid var(--border-color)', paddingBottom: '6px', marginTop: '20px', marginBottom: '16px', color: '#3b82f6', fontSize: '0.9rem' }}>
                    4. External Credit Scores (Normalized 0.0 - 1.0)
                  </h3>
                  <p style={{ color: '#a1a1aa', fontSize: '0.75rem', marginBottom: '12px' }}>
                    Scores from external agencies (e.g. bureaus/agencies). Lower scores reflect higher default probability. Leave blank to test pipeline missing value handling.
                  </p>
                  
                  <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr 1fr', gap: '12px' }}>
                    <div className="form-group">
                      <label className="form-label">EXT Score 1</label>
                      <input 
                        type="number" 
                        step="0.01" 
                        min="0" 
                        max="1" 
                        placeholder="e.g. 0.5"
                        className="form-control"
                        value={simulatorInput.ext_source_1 === null ? '' : simulatorInput.ext_source_1}
                        onChange={(e) => handleInputChange('ext_source_1', e.target.value === '' ? null : e.target.value)}
                      />
                    </div>
                    <div className="form-group">
                      <label className="form-label">EXT Score 2</label>
                      <input 
                        type="number" 
                        step="0.01" 
                        min="0" 
                        max="1" 
                        placeholder="e.g. 0.6"
                        className="form-control"
                        value={simulatorInput.ext_source_2 === null ? '' : simulatorInput.ext_source_2}
                        onChange={(e) => handleInputChange('ext_source_2', e.target.value === '' ? null : e.target.value)}
                      />
                    </div>
                    <div className="form-group">
                      <label className="form-label">EXT Score 3</label>
                      <input 
                        type="number" 
                        step="0.01" 
                        min="0" 
                        max="1" 
                        placeholder="e.g. 0.4"
                        className="form-control"
                        value={simulatorInput.ext_source_3 === null ? '' : simulatorInput.ext_source_3}
                        onChange={(e) => handleInputChange('ext_source_3', e.target.value === '' ? null : e.target.value)}
                      />
                    </div>
                  </div>

                  {/* Registration Change, queries */}
                  <h3 style={{ borderBottom: '1px solid var(--border-color)', paddingBottom: '6px', marginTop: '20px', marginBottom: '16px', color: '#3b82f6', fontSize: '0.9rem' }}>
                    5. Document history & Credit Bureau
                  </h3>
                  
                  <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr 1fr', gap: '12px' }}>
                    <div className="form-group">
                      <label className="form-label">Yrs Registration</label>
                      <input 
                        type="number" 
                        step="0.1" 
                        className="form-control"
                        required
                        value={simulatorInput.years_registration_change}
                        onChange={(e) => handleInputChange('years_registration_change', e.target.value)}
                      />
                    </div>
                    <div className="form-group">
                      <label className="form-label">Yrs ID Publish</label>
                      <input 
                        type="number" 
                        step="0.1" 
                        className="form-control"
                        required
                        value={simulatorInput.years_id_publish}
                        onChange={(e) => handleInputChange('years_id_publish', e.target.value)}
                      />
                    </div>
                    <div className="form-group">
                      <label className="form-label">Yrs Phone Change</label>
                      <input 
                        type="number" 
                        step="0.1" 
                        className="form-control"
                        required
                        value={simulatorInput.years_phone_change}
                        onChange={(e) => handleInputChange('years_phone_change', e.target.value)}
                      />
                    </div>
                  </div>

                  <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '16px', marginTop: '8px' }}>
                    <div className="form-group">
                      <label className="form-label">Bureau Queries (Year)</label>
                      <input 
                        type="number" 
                        className="form-control"
                        required
                        value={simulatorInput.credit_bureau_queries_year}
                        onChange={(e) => handleInputChange('credit_bureau_queries_year', e.target.value)}
                      />
                    </div>
                    <div className="form-group">
                      <label className="form-label">Region Rating</label>
                      <select 
                        className="form-control"
                        value={simulatorInput.region_rating}
                        onChange={(e) => handleInputChange('region_rating', e.target.value)}
                      >
                        <option value={1}>1 (Highest Rating)</option>
                        <option value={2}>2 (Medium Rating)</option>
                        <option value={3}>3 (Lowest Rating)</option>
                      </select>
                    </div>
                  </div>

                  <div style={{ marginTop: '24px' }}>
                    <button 
                      type="submit" 
                      className="btn" 
                      style={{ width: '100%' }}
                      disabled={simLoading}
                    >
                      {simLoading ? 'Running ML Assessment...' : 'Evaluate Credit default Risk'}
                    </button>
                  </div>
                </form>
              </div>

              {/* Right Column: Prediction Results & Explanations */}
              <div>
                {simError && (
                  <div className="alert alert-warning">
                    <AlertTriangle size={18} style={{ float: 'left', marginRight: '10px' }} />
                    <p style={{ marginLeft: '28px' }}>{simError}</p>
                  </div>
                )}

                {!simResult && !simLoading && (
                  <div className="card" style={{ display: 'flex', flexDirection: 'column', height: '100%', justifyContent: 'center', alignItems: 'center', minHeight: '400px', textAlign: 'center', color: '#a1a1aa' }}>
                    <ShieldCheck size={48} style={{ opacity: 0.15, marginBottom: '16px' }} />
                    <h3>Awaiting Evaluation</h3>
                    <p style={{ maxWidth: '300px', fontSize: '0.85rem', marginTop: '6px' }}>
                      Configure applicant credentials on the left and submit the form to invoke the trained preprocessing + ML pipeline.
                    </p>
                  </div>
                )}

                {simLoading && (
                  <div className="card" style={{ display: 'flex', flexDirection: 'column', height: '100%', justifyContent: 'center', alignItems: 'center', minHeight: '400px', textAlign: 'center', color: '#a1a1aa' }}>
                    <div style={{ border: '3px solid rgba(255,255,255,0.08)', borderTop: '3px solid #3b82f6', borderRadius: '50%', width: '32px', height: '32px', animation: 'spin 1s linear infinite', marginBottom: '16px' }}></div>
                    <h3>Processing Pipeline</h3>
                    <p style={{ maxWidth: '300px', fontSize: '0.85rem', marginTop: '6px' }}>
                      Imputing missing fields, scaling features, creating one-hot categories, and executing probabilities...
                    </p>
                  </div>
                )}

                {simResult && !simLoading && (
                  <div style={{ display: 'flex', flexDirection: 'column', gap: '24px' }}>
                    
                    {/* Primary Prediction Result */}
                    <div className="card" style={{ borderTop: `4px solid ${simResult.risk_classification === 'HIGH RISK' ? '#ef4444' : '#10b981'}` }}>
                      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start', marginBottom: '16px' }}>
                        <div>
                          <span style={{ fontSize: '0.75rem', fontWeight: 600, color: '#a1a1aa', textTransform: 'uppercase', letterSpacing: '0.05em' }}>Risk Classification</span>
                          <h2 style={{ 
                            fontSize: '2rem', 
                            fontWeight: 800, 
                            color: simResult.risk_classification === 'HIGH RISK' ? '#ef4444' : '#10b981',
                            marginTop: '2px',
                            lineHeight: 1.1
                          }}>
                            {simResult.risk_classification}
                          </h2>
                        </div>
                        <span style={{
                          padding: '4px 8px',
                          borderRadius: '4px',
                          fontSize: '0.75rem',
                          fontWeight: 600,
                          background: simResult.risk_classification === 'HIGH RISK' ? 'rgba(239, 68, 68, 0.1)' : 'rgba(16, 185, 129, 0.1)',
                          color: simResult.risk_classification === 'HIGH RISK' ? '#ef4444' : '#10b981'
                        }}>
                          Model: {simResult.model_used.replace('_', ' ').toUpperCase()}
                        </span>
                      </div>

                      {/* Prob Progress Bar */}
                      <div style={{ margin: '20px 0' }}>
                        <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: '0.85rem', marginBottom: '6px' }}>
                          <span style={{ fontWeight: 600 }}>Default Probability:</span>
                          <span style={{ 
                            fontWeight: 700, 
                            color: simResult.risk_classification === 'HIGH RISK' ? '#ef4444' : '#10b981' 
                          }}>
                            {(simResult.default_probability * 100).toFixed(1)}%
                          </span>
                        </div>
                        <div className="risk-gauge">
                          <div 
                            className="risk-gauge-fill" 
                            style={{ 
                              width: `${simResult.default_probability * 100}%`, 
                              background: simResult.risk_classification === 'HIGH RISK' ? '#ef4444' : '#10b981' 
                            }} 
                          />
                        </div>
                        <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: '0.7rem', color: '#a1a1aa', marginTop: '4px' }}>
                          <span>0% Risk</span>
                          <span>Cut-off Threshold: {(simResult.decision_threshold * 100).toFixed(0)}%</span>
                          <span>100% Risk</span>
                        </div>
                      </div>

                      <div style={{ borderTop: '1px solid var(--border-color)', paddingTop: '16px', fontSize: '0.825rem', color: '#a1a1aa', lineHeight: 1.5 }}>
                        {simResult.risk_classification === 'HIGH RISK' ? (
                          <p>
                            The applicant has a predicted default probability of <strong>{(simResult.default_probability*100).toFixed(1)}%</strong>, which is <strong>above</strong> the selected cut-off threshold of <strong>{(simResult.decision_threshold*100).toFixed(0)}%</strong>. System flags this application as elevated risk.
                          </p>
                        ) : (
                          <p>
                            The applicant has a predicted default probability of <strong>{(simResult.default_probability*100).toFixed(1)}%</strong>, which is <strong>below</strong> the selected cut-off threshold of <strong>{(simResult.decision_threshold*100).toFixed(0)}%</strong>. System flags this application as acceptable risk.
                          </p>
                        )}
                      </div>
                    </div>

                    {/* Local Explanations */}
                    <div className="card">
                      <h2>Why did the model make this prediction?</h2>
                      <p style={{ color: '#a1a1aa', fontSize: '0.85rem', marginBottom: '20px' }}>
                        Local feature contribution analysis showing feature influence on predicted default risk. 
                        <span style={{ color: '#ef4444', fontWeight: 600 }}> Positive (Red)</span> values push probability up; 
                        <span style={{ color: '#10b981', fontWeight: 600 }}> Negative (Green)</span> values pull probability down.
                      </p>
                      
                      {/* SVGBars for local explanation */}
                      <SVGLocalExplanation explanations={simResult.explanation} />
                      
                      {/* Explanations text list */}
                      <div style={{ display: 'flex', flexDirection: 'column', gap: '10px', marginTop: '24px', borderTop: '1px solid var(--border-color)', paddingTop: '20px' }}>
                        {simResult.explanation.map((item, idx) => (
                          <div key={idx} style={{ display: 'flex', gap: '10px', fontSize: '0.8rem', alignItems: 'flex-start' }}>
                            <span style={{ 
                              width: '8px', 
                              height: '8px', 
                              borderRadius: '50%', 
                              background: item.influence === 'High Risk' ? '#ef4444' : '#10b981',
                              marginTop: '6px',
                              flexShrink: 0
                            }} />
                            <p style={{ color: '#e4e4e7' }}>
                              <strong>{item.display_name}:</strong> {item.text}
                            </p>
                          </div>
                        ))}
                      </div>
                    </div>

                    {/* Disclaimer */}
                    <div className="alert alert-info" style={{ marginBottom: 0 }}>
                      <Info size={16} style={{ float: 'left', marginRight: '10px', marginTop: '2px' }} />
                      <p style={{ marginLeft: '26px' }}>
                        <strong>Disclaimer:</strong> This application is developed as an academic credit risk modeling tool for research and threshold analysis purposes. Predictions represent mathematical estimations trained on synthetic/anonymized patterns and do NOT constitute financial advice or formal credit approvals.
                      </p>
                    </div>
                  </div>
                )}
              </div>
            </div>
          </div>
        )}

        {/* TAB 6: GLOBAL EXPLAINABILITY */}
        {activeTab === 'explainability' && (
          <div>
            <div className="page-header">
              <h1>Global Model Explainability</h1>
              <p className="page-subtitle">Understand what features drive credit default predictions globally across the entire population.</p>
            </div>

            {/* Model select dropdown */}
            <div className="card" style={{ marginBottom: '24px', display: 'flex', justifyContent: 'space-between', alignItems: 'center', padding: '16px 24px' }}>
              <div style={{ display: 'flex', alignItems: 'center', gap: '12px' }}>
                <span style={{ fontSize: '0.9rem', fontWeight: 600, color: '#a1a1aa' }}>Model Explainability:</span>
                <select 
                  className="form-control" 
                  style={{ width: '280px', background: '#27272a' }}
                  value={selectedModel}
                  onChange={(e) => setSelectedModel(e.target.value)}
                >
                  <option value="logistic_regression">Logistic Regression (Baseline)</option>
                  <option value="gradient_boosting">HistGradientBoosting (Nonlinear)</option>
                </select>
              </div>
              <div style={{ fontSize: '0.85rem', color: '#a1a1aa' }}>
                Currently displaying: <strong style={{ color: '#ffffff' }}>{activeMetrics?.name}</strong>
              </div>
            </div>

            <div className="grid-cols-2" style={{ alignItems: 'start' }}>
              {/* Left Column: Global Importance Chart */}
              <div className="card">
                <h2>
                  {selectedModel === 'logistic_regression' ? 'Linear Coefficients Weights' : 'Permutation Feature Importance'}
                </h2>
                <p style={{ color: '#a1a1aa', fontSize: '0.85rem', marginBottom: '24px' }}>
                  {selectedModel === 'logistic_regression' 
                    ? "Coefficients represent the direct linear impact on the log-odds of default. Positive values increase default risk, negative values decrease risk."
                    : "Permutation importance measures how much the test ROC-AUC drops when a single feature column is randomly shuffled. Larger drops indicate higher reliance."}
                </p>
                <div style={{ display: 'flex', justifyContent: 'center' }}>
                  {activeMetrics && (
                    <SVGGlobalImportance 
                      modelKey={selectedModel} 
                      importanceData={activeMetrics} 
                    />
                  )}
                </div>
              </div>

              {/* Right Column: Explanation & Analysis */}
              <div className="card">
                <h2>Key Drivers Analysis</h2>
                
                <div style={{ display: 'flex', flexDirection: 'column', gap: '20px', marginTop: '10px' }}>
                  <div>
                    <h3 style={{ display: 'flex', alignItems: 'center', gap: '8px', color: '#ef4444' }}>
                      <AlertTriangle size={16} /> 1. External Source Ratings (EXT_SOURCE_x)
                    </h3>
                    <p style={{ color: '#a1a1aa', fontSize: '0.85rem', lineHeight: '1.45', marginTop: '6px' }}>
                      Both models place heavy reliance on the external source features. Higher scores dramatically reduce predicted credit default risk. This aligns with standard credit bureau ratings where a clean repayment history represents the strongest predictor of future compliance.
                    </p>
                  </div>
                  
                  <div>
                    <h3 style={{ display: 'flex', alignItems: 'center', gap: '8px', color: '#f59e0b' }}>
                      <Sliders size={16} style={{ transform: 'rotate(90deg)' }} /> 2. Applicant Age & Stability (DAYS_BIRTH)
                    </h3>
                    <p style={{ color: '#a1a1aa', fontSize: '0.85rem', lineHeight: '1.45', marginTop: '6px' }}>
                      Age is represented as a negative feature (days birth). Younger applicants (values closer to 0) statistically show elevated default rates compared to older, more established applicants. This is a common pattern in retail credit risk, as older cohorts tend to have more stable income histories and higher assets.
                    </p>
                  </div>

                  <div>
                    <h3 style={{ display: 'flex', alignItems: 'center', gap: '8px', color: '#3b82f6' }}>
                      <TrendingUp size={16} /> 3. Leverage Ratio (AMT_CREDIT / AMT_INCOME_TOTAL)
                    </h3>
                    <p style={{ color: '#a1a1aa', fontSize: '0.85rem', lineHeight: '1.45', marginTop: '6px' }}>
                      High credit requests relative to annual income (high debt-to-income) are flagged by both pipelines as adding credit stress. In Logistic Regression, `AMT_CREDIT` has a positive coefficient (increasing default risk) when not matched by corresponding income increases.
                    </p>
                  </div>

                  <div className="alert alert-info" style={{ marginBottom: 0, marginTop: '10px' }}>
                    <Info size={16} style={{ float: 'left', marginRight: '10px', marginTop: '2px' }} />
                    <p style={{ marginLeft: '26px' }}>
                      <strong>Interpretability Trade-off:</strong> The Logistic Regression model is fully linear and transparent, allowing us to see coefficients directly for every sub-feature. The Gradient Boosting model is nonlinear and complex, capturing high-degree interactions (e.g. how debt ratio matters differently for young vs old applicants), which we explain globally using independent Permutation Importance.
                    </p>
                  </div>
                </div>
              </div>
            </div>
          </div>
        )}

        {/* TAB 7: ABOUT */}
        {activeTab === 'about' && (
          <div style={{ maxWidth: '800px' }}>
            <div className="page-header">
              <h1>About CreditRisk-X</h1>
              <p className="page-subtitle">Educational documentation explaining ML integrity, architecture, and model limitations.</p>
            </div>

            <div className="card" style={{ display: 'flex', flexDirection: 'column', gap: '20px', lineHeight: '1.6' }}>
              <div>
                <h2>ML Pipeline Architecture</h2>
                <p style={{ color: '#a1a1aa', fontSize: '0.9rem', marginBottom: '12px' }}>
                  CreditRisk-X implements a robust machine learning system designed to simulate industry-grade credit scoring pipelines.
                </p>
                <div style={{ background: '#18181b', padding: '16px', borderRadius: '8px', border: '1px solid var(--border-color)', fontFamily: 'monospace', fontSize: '0.75rem', color: '#8b5cf6', margin: '14px 0' }}>
                  User Form input → Pydantic Validation → API Endpoint → Saved ML Pipeline (joblib) → Median Imputer → StandardScaler / One-Hot Encoder → Fit Estimator → Probability output → Threshold Classification
                </div>
              </div>

              <div>
                <h2>Academic Integrity & Leakage Prevention</h2>
                <p style={{ color: '#a1a1aa', fontSize: '0.9rem', marginBottom: '8px' }}>
                  Data leakage is a critical issue where test set information slips into model training, leading to overly optimistic results that fail in production. CreditRisk-X guarantees complete leakage protection:
                </p>
                <ul style={{ listStyleType: 'disc', marginLeft: '20px', color: '#a1a1aa', fontSize: '0.875rem', display: 'flex', flexDirection: 'column', gap: '6px' }}>
                  <li><strong>Pipeline encapsulation:</strong> Numerical scaling statistics (mean, std) and imputation statistics (medians) are fit strictly on the training partition within each cross-validation fold.</li>
                  <li><strong>Independent validation:</strong> The final evaluation metrics (ROC, PR, Confusion Matrices) are computed on a clean 20% holdout test set that was never exposed to the models during cross-validation or hyperparameter tuning.</li>
                  <li><strong>Threshold optimization:</strong> The threshold analysis slider utilizes precomputed test statistics, but the threshold itself is an external decision boundary that does not affect model parameter fitting.</li>
                </ul>
              </div>

              <div>
                <h2>Limitations & Ethical Credit Concerns</h2>
                <p style={{ color: '#a1a1aa', fontSize: '0.9rem', marginBottom: '8px' }}>
                  When deploying machine learning models in financial underwriting, several limitations and ethical considerations must be reviewed:
                </p>
                <ul style={{ listStyleType: 'disc', marginLeft: '20px', color: '#a1a1aa', fontSize: '0.875rem', display: 'flex', flexDirection: 'column', gap: '6px' }}>
                  <li><strong>Class Imbalance:</strong> Default rates are naturally low (~8-10%). Models risk overfitting the majority class. Appropriate performance metrics like ROC-AUC or PR-AUC must be prioritized over basic Accuracy.</li>
                  <li><strong>Feature Missingness:</strong> Crucial credit indicators like bureau ratings (`EXT_SOURCE_x`) often suffer from high missingness. While median imputation allows models to run, it reduces variance and may mask true default signals.</li>
                  <li><strong>Fairness & Bias:</strong> Demographics like Gender (`CODE_GENDER`) or Family Status can correlate with default rates due to systemic biases in historical underwriting. Training models directly on these characteristics can perpetuate socioeconomic discrimination, violating fair lending laws. In real underwriting, such sensitive fields are typically excluded or heavily regularized.</li>
                  <li><strong>Difference from Real Credit Decisions:</strong> Real underwriting involves comprehensive risk policies, compliance checks, fraud filters, and human-in-the-loop overrides. ML pipelines act as diagnostic scorecards, not final arbiters.</li>
                </ul>
              </div>
            </div>
          </div>
        )}
      </div>
    </div>
  );
}

/**
 * ---------------------------------------------------------
 * CUSTOM SVG CHART COMPONENTS
 * ---------------------------------------------------------
 */

/**
 * Renders ROC and PR curves side-by-side or individually overlaying both models.
 */
function SVGCurveCompare({ modelsSummary, activeModel, activeMetrics, curveType }) {
  const width = 340;
  const height = 280;
  const pad = 40;
  const chartW = width - 2 * pad;
  const chartH = height - 2 * pad;

  // Map coordinates (0.0 to 1.0) to SVG pixel space
  const getX = (val) => pad + val * chartW;
  const getY = (val) => pad + (1 - val) * chartH; // invert Y for SVG

  // Collect curve lines for both models
  const curves = {};
  Object.entries(modelsSummary).forEach(([key, val]) => {
    // If the active metrics belongs to this key, we can pull coordinates directly,
    // otherwise we can read it if pre-stored.
    // Fortunately, since we pre-loaded the activeMetrics and also saved all curves 
    // inside the metadata, we can extract them!
    // But wait! activeMetrics only contains curves for the *selected* model.
    // To show comparison, we overlay:
    // - Selected Model: Thick blue or purple line
    // - Baseline: Thin grey dashed line
    // Let's implement this.
  });

  const points = curveType === 'roc' 
    ? activeMetrics.roc_curve 
    : activeMetrics.pr_curve;

  if (!points) return null;

  // Build path data
  let dPath = "";
  const len = curveType === 'roc' ? points.fpr.length : points.recall.length;
  for (let i = 0; i < len; i++) {
    const xVal = curveType === 'roc' ? points.fpr[i] : points.recall[i];
    const yVal = curveType === 'roc' ? points.tpr[i] : points.precision[i];
    const prefix = i === 0 ? "M" : "L";
    dPath += `${prefix} ${getX(xVal).toFixed(1)} ${getY(yVal).toFixed(1)} `;
  }

  // Draw grid
  const gridLines = [0.2, 0.4, 0.6, 0.8];

  return (
    <svg width={width} height={height} style={{ background: '#121214', borderRadius: '8px', border: '1px solid rgba(255,255,255,0.04)' }}>
      {/* Grid Lines */}
      {gridLines.map(v => (
        <React.Fragment key={v}>
          <line x1={getX(v)} y1={getY(0)} x2={getX(v)} y2={getY(1)} stroke="rgba(255,255,255,0.05)" strokeDasharray="2" />
          <line x1={getX(0)} y1={getY(v)} x2={getX(1)} y2={getY(v)} stroke="rgba(255,255,255,0.05)" strokeDasharray="2" />
        </React.Fragment>
      ))}

      {/* Diagonal baseline (only for ROC) */}
      {curveType === 'roc' && (
        <line x1={getX(0)} y1={getY(0)} x2={getX(1)} y2={getY(1)} stroke="rgba(255,255,255,0.2)" strokeDasharray="4" />
      )}

      {/* Curve Path */}
      {dPath && (
        <path 
          d={dPath} 
          fill="none" 
          stroke={activeModel === 'logistic_regression' ? '#3b82f6' : '#8b5cf6'} 
          strokeWidth="2.5" 
          strokeLinecap="round"
          strokeLinejoin="round"
        />
      )}

      {/* Axes */}
      <line x1={getX(0)} y1={getY(0)} x2={getX(1)} y2={getY(0)} stroke="rgba(255,255,255,0.3)" strokeWidth="1" />
      <line x1={getX(0)} y1={getY(0)} x2={getX(0)} y2={getY(1)} stroke="rgba(255,255,255,0.3)" strokeWidth="1" />

      {/* Axis labels */}
      <text x={width / 2} y={height - 8} fill="#a1a1aa" fontSize="10" textAnchor="middle" fontWeight="500">
        {curveType === 'roc' ? 'False Positive Rate' : 'Recall (Sensitivity)'}
      </text>
      <text x="12" y={height / 2} fill="#a1a1aa" fontSize="10" textAnchor="middle" transform={`rotate(-90 12 ${height/2})`} fontWeight="500">
        {curveType === 'roc' ? 'True Positive Rate' : 'Precision (Positive Predictive Value)'}
      </text>

      {/* Ticks & Values */}
      <text x={getX(0)} y={getY(0) + 14} fill="#a1a1aa" fontSize="9" textAnchor="middle">0</text>
      <text x={getX(0.5)} y={getY(0) + 14} fill="#a1a1aa" fontSize="9" textAnchor="middle">0.5</text>
      <text x={getX(1)} y={getY(0) + 14} fill="#a1a1aa" fontSize="9" textAnchor="middle">1</text>

      <text x={getX(0) - 8} y={getY(0) + 3} fill="#a1a1aa" fontSize="9" textAnchor="end">0</text>
      <text x={getX(0) - 8} y={getY(0.5) + 3} fill="#a1a1aa" fontSize="9" textAnchor="end">0.5</text>
      <text x={getX(0) - 8} y={getY(1) + 3} fill="#a1a1aa" fontSize="9" textAnchor="end">1</text>

      {/* Legend */}
      <rect x={pad + 12} y={pad + 12} width="165" height="26" fill="rgba(0,0,0,0.4)" rx="4" border="1px solid rgba(255,255,255,0.05)" />
      <circle cx={pad + 20} cy={pad + 25} r="4" fill={activeModel === 'logistic_regression' ? '#3b82f6' : '#8b5cf6'} />
      <text x={pad + 30} y={pad + 28} fill="#ffffff" fontSize="9" fontWeight="600">
        {activeModel === 'logistic_regression' ? 'Logistic Regression' : 'HistGradientBoosting'} (AUC: {activeMetrics[`test_${curveType}_auc`]?.toFixed(3) || activeMetrics.test_auc.toFixed(3)})
      </text>
    </svg>
  );
}

/**
 * Line chart showing how Precision, Recall, and F1 vary by probability threshold.
 */
function SVGThresholdChart({ thresholdAnalysis, currentThreshold }) {
  const width = 340;
  const height = 280;
  const pad = 40;
  const chartW = width - 2 * pad;
  const chartH = height - 2 * pad;

  const getX = (val) => pad + ((val - 0.05) / 0.90) * chartW; // threshold ranges 0.05 to 0.95
  const getY = (val) => pad + (1 - val) * chartH; // metrics range 0.0 to 1.0

  if (!thresholdAnalysis || thresholdAnalysis.length === 0) return null;

  // Build path strings
  let dPrec = "";
  let dRec = "";
  let dF1 = "";

  thresholdAnalysis.forEach((row, i) => {
    const prefix = i === 0 ? "M" : "L";
    const cx = getX(row.threshold).toFixed(1);
    
    dPrec += `${prefix} ${cx} ${getY(row.precision).toFixed(1)} `;
    dRec += `${prefix} ${cx} ${getY(row.recall).toFixed(1)} `;
    dF1 += `${prefix} ${cx} ${getY(row.f1_score).toFixed(1)} `;
  });

  return (
    <svg width={width} height={height} style={{ background: '#121214', borderRadius: '8px', border: '1px solid rgba(255,255,255,0.04)' }}>
      {/* Grid Lines */}
      {[0.2, 0.4, 0.6, 0.8].map(v => (
        <React.Fragment key={v}>
          <line x1={pad} y1={getY(v)} x2={width - pad} y2={getY(v)} stroke="rgba(255,255,255,0.03)" />
        </React.Fragment>
      ))}

      {/* Draw curve paths */}
      <path d={dPrec} fill="none" stroke="#10b981" strokeWidth="2" strokeLinecap="round" />
      <path d={dRec} fill="none" stroke="#3b82f6" strokeWidth="2" strokeLinecap="round" />
      <path d={dF1} fill="none" stroke="#8b5cf6" strokeWidth="2" strokeLinecap="round" />

      {/* Active Threshold Indicator Line */}
      <line 
        x1={getX(currentThreshold)} 
        y1={getY(0)} 
        x2={getX(currentThreshold)} 
        y2={getY(1)} 
        stroke="#f59e0b" 
        strokeWidth="1.5" 
        strokeDasharray="3" 
      />
      <circle cx={getX(currentThreshold)} cy={getY(1)} r="3" fill="#f59e0b" />
      <text x={getX(currentThreshold)} y={getY(1) - 6} fill="#f59e0b" fontSize="8" fontWeight="700" textAnchor="middle">
        Cut-off
      </text>

      {/* Axes */}
      <line x1={pad} y1={getY(0)} x2={width - pad} y2={getY(0)} stroke="rgba(255,255,255,0.3)" />
      <line x1={pad} y1={getY(0)} x2={pad} y2={getY(1)} stroke="rgba(255,255,255,0.3)" />

      {/* Labels */}
      <text x={width / 2} y={height - 8} fill="#a1a1aa" fontSize="10" textAnchor="middle" fontWeight="500">
        Probability Decision Threshold
      </text>
      
      {/* Ticks */}
      <text x={getX(0.1)} y={getY(0) + 14} fill="#a1a1aa" fontSize="9" textAnchor="middle">0.1</text>
      <text x={getX(0.5)} y={getY(0) + 14} fill="#a1a1aa" fontSize="9" textAnchor="middle">0.5</text>
      <text x={getX(0.9)} y={getY(0) + 14} fill="#a1a1aa" fontSize="9" textAnchor="middle">0.9</text>

      <text x={pad - 8} y={getY(0) + 3} fill="#a1a1aa" fontSize="9" textAnchor="end">0.0</text>
      <text x={pad - 8} y={getY(0.5) + 3} fill="#a1a1aa" fontSize="9" textAnchor="end">0.5</text>
      <text x={pad - 8} y={getY(1) + 3} fill="#a1a1aa" fontSize="9" textAnchor="end">1.0</text>

      {/* Floating legends */}
      <g transform={`translate(${pad + 12}, 16)`}>
        <rect width="210" height="18" fill="rgba(0,0,0,0.5)" rx="4" />
        <circle cx="10" cy="9" r="3.5" fill="#10b981" />
        <text x="18" y="12" fill="#a1a1aa" fontSize="8" fontWeight="600">Precision</text>
        
        <circle cx="80" cy="9" r="3.5" fill="#3b82f6" />
        <text x="88" y="12" fill="#a1a1aa" fontSize="8" fontWeight="600">Recall</text>

        <circle cx="140" cy="9" r="3.5" fill="#8b5cf6" />
        <text x="148" y="12" fill="#a1a1aa" fontSize="8" fontWeight="600">F1-Score</text>
      </g>
    </svg>
  );
}

/**
 * Renders local explanations as a horizontal diverging bar chart.
 */
function SVGLocalExplanation({ explanations }) {
  const width = 420;
  const barHeight = 20;
  const spacing = 10;
  const pad = 10;
  const chartW = width - 2 * pad;
  const numBars = explanations.length;
  const height = numBars * (barHeight + spacing) + 30;

  // Find max contribution for scale
  const maxVal = Math.max(...explanations.map(e => Math.abs(e.contribution)), 0.05);

  // Map contribution value to width
  const scale = (val) => (val / maxVal) * (chartW / 2 - 20); // half width minus text buffer
  const center = chartW / 2 + pad;

  return (
    <svg width={width} height={height} style={{ background: '#121214', borderRadius: '8px', padding: '12px', border: '1px solid rgba(255,255,255,0.04)' }}>
      {/* Center axis */}
      <line x1={center} y1={5} x2={center} y2={height - 25} stroke="rgba(255,255,255,0.2)" strokeWidth="1" />
      
      {/* Labels for positive/negative influence */}
      <text x={pad + 10} y={height - 8} fill="#10b981" fontSize="9" fontWeight="700" textAnchor="start">
        ← Reduces Default Risk
      </text>
      <text x={width - pad - 10} y={height - 8} fill="#ef4444" fontSize="9" fontWeight="700" textAnchor="end">
        Increases Default Risk →
      </text>

      {explanations.map((item, idx) => {
        const y = idx * (barHeight + spacing) + 10;
        const cVal = item.contribution;
        const w = Math.abs(scale(cVal));
        const isPos = cVal > 0;
        const x = isPos ? center : center - w;
        
        // Colors: Pos = Red (increase risk), Neg = Green (reduce risk)
        const fill = isPos ? '#ef4444' : '#10b981';

        return (
          <g key={idx}>
            {/* Bar */}
            <rect 
              x={x} 
              y={y} 
              width={w} 
              height={barHeight} 
              fill={fill} 
              rx="3" 
              style={{ transition: 'all 0.5s ease' }}
            />
            
            {/* Text Label on the opposite side of the bar */}
            <text 
              x={isPos ? center - 8 : center + 8} 
              y={y + 13} 
              fill="#f4f4f5" 
              fontSize="8.5" 
              fontWeight="600"
              textAnchor={isPos ? "end" : "start"}
            >
              {item.display_name}
            </text>
            
            {/* Value label */}
            <text
              x={isPos ? x + w + 6 : x - 6}
              y={y + 13}
              fill={fill}
              fontSize="8"
              fontWeight="700"
              textAnchor={isPos ? "start" : "end"}
            >
              {isPos ? '+' : ''}{cVal.toFixed(3)}
            </text>
          </g>
        );
      })}
    </svg>
  );
}

/**
 * Global Feature Importance or Coefficient Weights bar chart.
 */
function SVGGlobalImportance({ modelKey, importanceData }) {
  const width = 380;
  const barHeight = 16;
  const spacing = 8;
  const pad = 12;
  const chartW = width - 2 * pad;

  if (!importanceData || !importanceData.importance || !importanceData.importance.values) {
    return (
      <div style={{ padding: '20px', color: '#a1a1aa', fontSize: '0.875rem', textAlign: 'center' }}>
        Awaiting model feature importance metrics...
      </div>
    );
  }

  const impValues = importanceData.importance.values;
  const impType = importanceData.importance.type;

  // Sort feature importances/coefficients by absolute magnitude descending
  const sortedFeatures = Object.entries(impValues)
    .map(([feature, val]) => ({ feature, val }))
    .sort((a, b) => Math.abs(b.val) - Math.abs(a.val))
    .slice(0, 12); // top 12 features

  const numBars = sortedFeatures.length;
  const height = numBars * (barHeight + spacing) + 30;

  const maxVal = Math.max(...sortedFeatures.map(f => Math.abs(f.val)), 0.01);
  const center = impType === 'coefficients' ? (chartW / 2 + pad) : (pad + 120);
  const scale = (val) => (val / maxVal) * (impType === 'coefficients' ? (chartW / 2 - 30) : (chartW - 140));

  return (
    <svg width={width} height={height} style={{ background: '#121214', borderRadius: '8px', padding: '12px', border: '1px solid rgba(255,255,255,0.04)' }}>
      {/* Center axis for coefficients, left axis for importances */}
      <line 
        x1={center} 
        y1={5} 
        x2={center} 
        y2={height - 25} 
        stroke="rgba(255,255,255,0.2)" 
        strokeWidth="1" 
      />

      {/* Axis directions for coefficients */}
      {impType === 'coefficients' ? (
        <>
          <text x={pad + 10} y={height - 8} fill="#10b981" fontSize="9" fontWeight="700" textAnchor="start">
            ← Reduces Default Risk (Negative Coef)
          </text>
          <text x={width - pad - 10} y={height - 8} fill="#ef4444" fontSize="9" fontWeight="700" textAnchor="end">
            Increases Default Risk (Positive Coef) →
          </text>
        </>
      ) : (
        <text x={width / 2} y={height - 8} fill="#a1a1aa" fontSize="9" fontWeight="700" textAnchor="middle">
          Importance Score (ROC-AUC performance drop)
        </text>
      )}

      {sortedFeatures.map((item, idx) => {
        const y = idx * (barHeight + spacing) + 10;
        const val = item.val;
        const w = Math.abs(scale(val));
        const display_name = item.feature.replace('_', ' ').replace('NAME_', '').replace('CODE_GENDER_', 'Gender: ').replace('NAME_EDUCATION_TYPE_', 'Edu: ').titleCase();
        
        let x, fill;
        if (impType === 'coefficients') {
          const isPos = val > 0;
          x = isPos ? center : center - w;
          fill = isPos ? '#ef4444' : '#10b981';
        } else {
          x = center;
          fill = '#8b5cf6'; // Purple for Gradient Boosting importance
        }

        return (
          <g key={idx}>
            {/* Feature name label */}
            <text 
              x={center - 6} 
              y={y + 11} 
              fill="#e4e4e7" 
              fontSize="8" 
              fontWeight="600"
              textAnchor="end"
            >
              {display_name.length > 20 ? display_name.substring(0, 18) + '..' : display_name}
            </text>

            {/* Bar */}
            <rect 
              x={x} 
              y={y} 
              width={w} 
              height={barHeight} 
              fill={fill} 
              rx="2" 
            />

            {/* Value */}
            <text
              x={impType === 'coefficients' ? (val > 0 ? x + w + 4 : x - 4) : x + w + 4}
              y={y + 11}
              fill={fill}
              fontSize="7.5"
              fontWeight="700"
              textAnchor={impType === 'coefficients' && val < 0 ? 'end' : 'start'}
            >
              {val > 0 && impType === 'coefficients' ? '+' : ''}{val.toFixed(3)}
            </text>
          </g>
        );
      })}
    </svg>
  );
}

// Extend String prototype helper for easy title case formatting inside render
String.prototype.titleCase = function() {
  return this.split(' ').map(w => w.charAt(0).toUpperCase() + w.slice(1).toLowerCase()).join(' ');
};
String.prototype.title = function() {
  return this.replace(/_/g, ' ').split(' ').map(w => w.charAt(0).toUpperCase() + w.slice(1).toLowerCase()).join(' ');
};
