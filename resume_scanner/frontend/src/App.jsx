import { useState, useRef, useEffect, useCallback } from 'react';
import './index.css';

// ── Debounce hook ─────────────────────────────────────────────
function useDebounce(value, delay) {
  const [debouncedValue, setDebouncedValue] = useState(value);
  useEffect(() => {
    const handler = setTimeout(() => setDebouncedValue(value), delay);
    return () => clearTimeout(handler);
  }, [value, delay]);
  return debouncedValue;
}

function App() {
  const [file, setFile] = useState(null);
  const [isScanning, setIsScanning] = useState(false);
  const [result, setResult] = useState(null);
  const [error, setError] = useState('');
  const fileInputRef = useRef(null);

  // ── Evaluator state ───────────────────────────────────────
  const [isEvaluating, setIsEvaluating] = useState(false);
  const [evaluation, setEvaluation] = useState(null);
  const [evalError, setEvalError] = useState('');

  // ── JD Builder state ──────────────────────────────────────
  const [roleTitle, setRoleTitle]           = useState('');
  const [experienceYears, setExperienceYears] = useState('');
  const [degreeRequired, setDegreeRequired]   = useState('Any');
  const [suggestedSkills, setSuggestedSkills] = useState([]);   // from AI
  const [checkedSkills, setCheckedSkills]     = useState({});   // { skill: bool }
  const [customSkill, setCustomSkill]         = useState('');
  const [isFetchingSkills, setIsFetchingSkills] = useState(false);
  const [customSkills, setCustomSkills]       = useState([]);   // user-added

  const debouncedRole = useDebounce(roleTitle, 700);

  // Auto-fetch skills when role title changes (debounced)
  useEffect(() => {
    if (!debouncedRole.trim() || debouncedRole.trim().length < 3) return;
    let cancelled = false;
    setIsFetchingSkills(true);
    fetch('http://localhost:8000/api/suggest-skills', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ role_title: debouncedRole.trim() }),
    })
      .then(r => r.json())
      .then(data => {
        if (cancelled) return;
        const skills = data.skills || [];
        setSuggestedSkills(skills);
        // Pre-check all suggested skills by default
        const init = {};
        skills.forEach(s => { init[s] = true; });
        customSkills.forEach(s => { init[s] = true; });
        setCheckedSkills(init);
      })
      .catch(() => {})
      .finally(() => { if (!cancelled) setIsFetchingSkills(false); });
    return () => { cancelled = true; };
  }, [debouncedRole]);

  // Toggle a skill checkbox
  const toggleSkill = (skill) => {
    setCheckedSkills(prev => ({ ...prev, [skill]: !prev[skill] }));
  };

  // Add a custom skill
  const addCustomSkill = () => {
    const s = customSkill.trim();
    if (!s || customSkills.includes(s) || suggestedSkills.includes(s)) {
      setCustomSkill('');
      return;
    }
    setCustomSkills(prev => [...prev, s]);
    setCheckedSkills(prev => ({ ...prev, [s]: true }));
    setCustomSkill('');
  };

  // Remove a custom skill
  const removeCustomSkill = (skill) => {
    setCustomSkills(prev => prev.filter(s => s !== skill));
    setCheckedSkills(prev => { const n = { ...prev }; delete n[skill]; return n; });
  };

  // Build the final JD text from the builder fields
  const buildJDText = () => {
    const selectedSkills = [
      ...suggestedSkills.filter(s => checkedSkills[s]),
      ...customSkills.filter(s => checkedSkills[s]),
    ];
    let jd = `Job Title: ${roleTitle}`;
    if (experienceYears) jd += `\nRequired Experience: ${experienceYears}+ years`;
    if (degreeRequired !== 'Any') jd += `\nRequired Degree: ${degreeRequired}`;
    if (selectedSkills.length > 0)
      jd += `\nRequired Skills: ${selectedSkills.join(', ')}`;
    return jd;
  };

  // How many skills are currently checked
  const checkedCount = Object.values(checkedSkills).filter(Boolean).length;

  // ── File handlers ─────────────────────────────────────────
  const handleDragOver = (e) => e.preventDefault();

  const handleDrop = (e) => {
    e.preventDefault();
    if (e.dataTransfer.files?.length > 0)
      validateAndSetFile(e.dataTransfer.files[0]);
  };

  const handleFileChange = (e) => {
    if (e.target.files?.length > 0) validateAndSetFile(e.target.files[0]);
  };

  const validateAndSetFile = (selectedFile) => {
    const validTypes = ['application/pdf', 'image/png', 'image/jpeg', 'image/jpg'];
    if (!validTypes.includes(selectedFile.type)) {
      setError('Please select a valid PDF or Image file (PNG/JPG).');
      return;
    }
    setError('');
    setFile(selectedFile);
    setResult(null);
  };

  // ── Scan handler ──────────────────────────────────────────
  const handleScan = async () => {
    if (!file) return;
    setIsScanning(true);
    setError('');
    setResult(null);

    const formData = new FormData();
    formData.append('file', file);

    try {
      const response = await fetch('http://localhost:8000/api/scan-resume', {
        method: 'POST',
        body: formData,
      });
      if (!response.ok) {
        const errData = await response.json();
        throw new Error(errData.detail || 'Failed to scan resume');
      }
      const data = await response.json();
      const jdText = roleTitle ? buildJDText() : null;
      setResult({
        sanitizedText: data.sanitized_text,
        structuredData: data.structured_data,
        candidateRole: data.candidate_role || null,
        jdText,
        roleTitle: roleTitle || null,
      });
      // Reset evaluation when a new scan is done
      setEvaluation(null);
      setEvalError('');
    } catch (err) {
      setError(err.message);
    } finally {
      setIsScanning(false);
    }
  };

  // ── Evaluate handler ─────────────────────────────────────
  const handleEvaluate = async () => {
    if (!result?.structuredData || !result?.jdText) return;
    setIsEvaluating(true);
    setEvalError('');
    setEvaluation(null);

    try {
      const response = await fetch('http://localhost:8000/api/evaluate', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          jd_text: result.jdText,
          structured_data: result.structuredData,
        }),
      });
      if (!response.ok) {
        const errData = await response.json();
        throw new Error(errData.detail || 'Evaluation failed');
      }
      const data = await response.json();
      setEvaluation(data);
    } catch (err) {
      setEvalError(err.message);
    } finally {
      setIsEvaluating(false);
    }
  };

  const sd = result?.structuredData;

  return (
    <div className="app-container">
      <div className="background-glow"></div>
      <div className="background-glow highlight"></div>

      <header className="header">
        <h1>Aura<span className="text-primary">Parse</span></h1>
        <p>Next-Gen Resume Intelligence</p>
      </header>

      <main className="main-content">
        {/* ═══ Upload Panel ══════════════════════════════════ */}
        <div className={`upload-panel ${result ? 'has-result' : ''}`}>
          <div className="panel-header">
            <h2>Upload Resume</h2>
            <div className="pdf-badge">PDF / IMAGE</div>
          </div>

          {/* ── JD Builder ────────────────────────────────── */}
          <div className="jd-builder">
            <div className="jd-builder-header">
              <span className="jd-builder-icon">🎯</span>
              <span className="jd-builder-title">Job Description Builder</span>
              <span className="jd-builder-subtitle">AI fills in the requirements</span>
            </div>

            {/* Row 1: Role title + experience + degree */}
            <div className="jd-row jd-row-top">
              <div className="jd-field jd-field-role">
                <label className="jd-label">Role Title</label>
                <div className="jd-role-wrap">
                  <input
                    id="jd-role-input"
                    type="text"
                    className="jd-input"
                    placeholder="e.g. Backend Engineer, Data Scientist…"
                    value={roleTitle}
                    onChange={e => setRoleTitle(e.target.value)}
                  />
                  {isFetchingSkills && (
                    <div className="jd-spinner" title="AI generating skills…"></div>
                  )}
                </div>
              </div>
              <div className="jd-field jd-field-exp">
                <label className="jd-label">Min Experience</label>
                <div className="jd-exp-wrap">
                  <input
                    id="jd-exp-input"
                    type="number"
                    min="0"
                    max="30"
                    className="jd-input jd-input-sm"
                    placeholder="3"
                    value={experienceYears}
                    onChange={e => setExperienceYears(e.target.value)}
                  />
                  <span className="jd-exp-unit">yrs</span>
                </div>
              </div>
              <div className="jd-field jd-field-degree">
                <label className="jd-label">Degree</label>
                <select
                  id="jd-degree-select"
                  className="jd-input jd-select"
                  value={degreeRequired}
                  onChange={e => setDegreeRequired(e.target.value)}
                >
                  <option value="Any">Any</option>
                  <option value="High School">High School</option>
                  <option value="Associate">Associate</option>
                  <option value="Bachelor">Bachelor</option>
                  <option value="Master">Master</option>
                  <option value="PhD">PhD</option>
                </select>
              </div>
            </div>

            {/* Row 2: Skill checkboxes */}
            {(suggestedSkills.length > 0 || customSkills.length > 0) && (
              <div className="jd-skills-section">
                <div className="jd-skills-header">
                  <span className="jd-label">
                    ✨ AI-Suggested Requirements
                    {checkedCount > 0 && (
                      <span className="jd-checked-count">{checkedCount} selected</span>
                    )}
                  </span>
                  <button
                    className="jd-toggle-all"
                    onClick={() => {
                      const allSkills = [...suggestedSkills, ...customSkills];
                      const allChecked = allSkills.every(s => checkedSkills[s]);
                      const next = {};
                      allSkills.forEach(s => { next[s] = !allChecked; });
                      setCheckedSkills(next);
                    }}
                  >
                    {[...suggestedSkills, ...customSkills].every(s => checkedSkills[s])
                      ? 'Deselect All'
                      : 'Select All'}
                  </button>
                </div>

                <div className="jd-skill-chips">
                  {suggestedSkills.map(skill => (
                    <button
                      key={skill}
                      className={`jd-skill-chip ${checkedSkills[skill] ? 'checked' : 'unchecked'}`}
                      onClick={() => toggleSkill(skill)}
                    >
                      <span className="jd-chip-check">{checkedSkills[skill] ? '✓' : '+'}</span>
                      {skill}
                    </button>
                  ))}
                  {customSkills.map(skill => (
                    <button
                      key={skill}
                      className="jd-skill-chip checked custom"
                      onClick={() => removeCustomSkill(skill)}
                      title="Click to remove"
                    >
                      <span className="jd-chip-check">✓</span>
                      {skill}
                      <span className="jd-chip-remove">✕</span>
                    </button>
                  ))}
                </div>

                {/* Custom skill input */}
                <div className="jd-custom-row">
                  <input
                    type="text"
                    className="jd-input jd-custom-input"
                    placeholder="Add custom requirement…"
                    value={customSkill}
                    onChange={e => setCustomSkill(e.target.value)}
                    onKeyDown={e => e.key === 'Enter' && addCustomSkill()}
                  />
                  <button className="jd-add-btn" onClick={addCustomSkill}>+ Add</button>
                </div>
              </div>
            )}

            {/* Empty state while waiting for role */}
            {roleTitle.trim().length > 0 && roleTitle.trim().length < 3 && (
              <div className="jd-hint">Keep typing the role title…</div>
            )}
            {!roleTitle.trim() && (
              <div className="jd-hint">Type a role title to auto-generate required skills ↑</div>
            )}
          </div>

          {/* ── Drop Zone ──────────────────────────────────── */}
          <div
            className={`drop-zone ${file ? 'has-file' : ''}`}
            onDragOver={handleDragOver}
            onDrop={handleDrop}
            onClick={() => fileInputRef.current.click()}
          >
            <input
              type="file"
              ref={fileInputRef}
              style={{ display: 'none' }}
              accept="application/pdf,image/png,image/jpeg,image/jpg"
              onChange={handleFileChange}
            />
            {!file ? (
              <div className="drop-content">
                <div className="upload-icon">📄</div>
                <h3>Drag &amp; Drop your resume here</h3>
                <p>or click to browse from your computer</p>
              </div>
            ) : (
              <div className="file-info">
                <div className="file-icon">✅</div>
                <div className="file-details">
                  <h3>{file.name}</h3>
                  <p>{(file.size / 1024 / 1024).toFixed(2)} MB</p>
                </div>
                <button
                  className="remove-btn"
                  onClick={e => { e.stopPropagation(); setFile(null); setResult(null); }}
                >✕</button>
              </div>
            )}
          </div>

          {error && <div className="error-message">⚠️ {error}</div>}

          <button
            id="scan-resume-btn"
            className={`scan-btn ${isScanning ? 'scanning' : ''} ${!file ? 'disabled' : ''}`}
            onClick={handleScan}
            disabled={!file || isScanning}
          >
            {isScanning ? (
              <span className="btn-content">
                <div className="spinner"></div>
                Analyzing Document...
              </span>
            ) : (
              <span className="btn-content">✨ Scan Resume ✨</span>
            )}
          </button>
        </div>

        {/* ═══ Structured Analysis Panel ═════════════════════ */}
        {sd && (
          <div className="result-panel structured-panel">
            <div className="panel-header">
              <h2>📊 Structured Analysis</h2>
              <span className="bias-badge">Bias-Free</span>
            </div>

            {/* Role Comparison Banner */}
            <div className="role-comparison-banner">
              <div className="role-card candidate-role-card">
                <div className="role-card-label">👤 Candidate Specialises In</div>
                <div className="role-card-value">{result.candidateRole || 'Software Engineer'}</div>
              </div>
              {result.roleTitle && (
                <>
                  <div className="role-vs">vs</div>
                  <div className="role-card hiring-role-card">
                    <div className="role-card-label">🎯 You're Hiring For</div>
                    <div className="role-card-value">{result.roleTitle}</div>
                  </div>
                </>
              )}
            </div>

            <div className="structured-grid">
              {/* Work Experience Summary */}
              {sd.work_experience_summary ? (
                <div className="stat-card wide section-card section-work">
                  <div className="stat-label">⏱ Work Experience Summary</div>
                  <div className="work-summary-header">
                    <div className="work-total">
                      <span className="work-years">{sd.work_experience_summary.total_years}</span>
                      <span className="work-years-label">yrs total</span>
                    </div>
                    <div className="work-breakdown">
                      {sd.work_experience_summary.jobs_count > 0 && (
                        <div className="work-breakdown-item">
                          <span className="type-badge type-job">💼 Jobs</span>
                          <span className="breakdown-detail">
                            {sd.work_experience_summary.jobs_count} role{sd.work_experience_summary.jobs_count !== 1 ? 's' : ''}
                            · {sd.work_experience_summary.jobs_months} mo
                          </span>
                        </div>
                      )}
                      {sd.work_experience_summary.internships_count > 0 && (
                        <div className="work-breakdown-item">
                          <span className="type-badge type-intern">📍 Internships</span>
                          <span className="breakdown-detail">
                            {sd.work_experience_summary.internships_count} role{sd.work_experience_summary.internships_count !== 1 ? 's' : ''}
                            · {sd.work_experience_summary.internships_months} mo
                          </span>
                        </div>
                      )}
                    </div>
                  </div>
                  <div className="section-list" style={{ marginTop: '0.75rem' }}>
                    {sd.work_experience_summary.roles.map((role, i) => (
                      <div key={i} className="section-row exp-row">
                        <div className="section-row-left">
                          <div className="section-title-row">
                            <span className="section-title">{role.title || 'Role'}</span>
                            <span className={`type-badge ${role.type === 'Internship' ? 'type-intern' : 'type-job'}`}>
                              {role.type === 'Internship' ? '📍 Intern' : '💼 Job'}
                            </span>
                          </div>
                        </div>
                        <div className="section-row-right">
                          {role.date_range && <span className="section-date">{role.date_range}</span>}
                          <span className="job-duration">{role.duration_months} mo</span>
                        </div>
                      </div>
                    ))}
                  </div>
                </div>
              ) : (
                <div className="stat-card">
                  <div className="stat-label">Total Experience</div>
                  <div className="stat-value">
                    {sd.total_years_experience != null ? `${sd.total_years_experience} yrs` : 'N/A'}
                  </div>
                </div>
              )}

              {/* Highest Degree */}
              <div className="stat-card">
                <div className="stat-label">Highest Degree</div>
                <div className="stat-value degree">{sd.highest_degree || 'N/A'}</div>
              </div>

              {/* Technical Skills */}
              <div className="stat-card wide">
                <div className="stat-label">🛠 Technical Skills ({sd.technical_skills?.length || 0})</div>
                <div className="skills-list">
                  {sd.technical_skills?.length > 0
                    ? sd.technical_skills.map((s, i) => <span key={i} className="skill-chip">{s}</span>)
                    : <span className="text-muted">None detected</span>
                  }
                </div>
              </div>

              {/* Education */}
              {sd.education?.length > 0 && (
                <div className="stat-card wide section-card section-edu">
                  <div className="stat-label">🎓 Education ({sd.education.length})</div>
                  <div className="section-list">
                    {sd.education.map((edu, i) => (
                      <div key={i} className="section-row edu-row">
                        <div className="section-row-left">
                          <div className="section-title-row">
                            <span className="section-title">{edu.degree}</span>
                            {edu.gpa && <span className="score-chip gpa-chip">GPA {edu.gpa}</span>}
                            {edu.score && <span className="score-chip score-chip-pct">{edu.score}</span>}
                          </div>
                          {edu.field && <span className="section-subtitle">{edu.field}</span>}
                        </div>
                        {edu.institution && <span className="section-meta">{edu.institution}</span>}
                      </div>
                    ))}
                  </div>
                </div>
              )}

              {/* Experience */}
              {sd.experience?.length > 0 ? (
                <div className="stat-card wide section-card section-exp">
                  <div className="stat-label">💼 Experience ({sd.experience.length} roles)</div>
                  <div className="section-list">
                    {sd.experience.map((exp, i) => (
                      <div key={i} className="section-row exp-row">
                        <div className="section-row-left">
                          <div className="section-title-row">
                            <span className="section-title">{exp.title || 'Role'}</span>
                            <span className={`type-badge ${exp.type === 'Internship' ? 'type-intern' : 'type-job'}`}>
                              {exp.type === 'Internship' ? '📍 Intern' : '💼 Job'}
                            </span>
                          </div>
                          {exp.company && <span className="section-subtitle">{exp.company}</span>}
                        </div>
                        <div className="section-row-right">
                          {exp.date_range && <span className="section-date">{exp.date_range}</span>}
                          <span className="job-duration">{exp.duration_months} mo</span>
                        </div>
                      </div>
                    ))}
                  </div>
                </div>
              ) : sd.job_history?.length > 0 && (
                <div className="stat-card wide section-card section-exp">
                  <div className="stat-label">💼 Job History ({sd.job_history.length} roles)</div>
                  <div className="section-list">
                    {sd.job_history.map((job, i) => (
                      <div key={i} className="section-row exp-row">
                        <div className="section-row-left">
                          <span className="section-title">{job.title}</span>
                        </div>
                        <span className="job-duration">{job.duration_months} mo</span>
                      </div>
                    ))}
                  </div>
                </div>
              )}
            </div>
          </div>
        )}

        {/* ═══ Evaluator Scorecard Panel ══════════════════════ */}
        {sd && result?.jdText && (
          <div className="result-panel evaluator-panel">
            <div className="panel-header">
              <h2>🏆 Evaluator Scorecard</h2>
              <span className="bias-badge">Bot 4 — Phi-3.5</span>
            </div>

            {!evaluation && !isEvaluating && (
              <div className="eval-trigger">
                <p className="eval-trigger-desc">
                  Ready to score this candidate against your Job Description.
                  The evaluator will assign 0–10 ratings per skill with factual justifications.
                </p>
                <button
                  id="evaluate-btn"
                  className={`scan-btn eval-btn ${isEvaluating ? 'scanning' : ''}`}
                  onClick={handleEvaluate}
                  disabled={isEvaluating}
                >
                  <span className="btn-content">🎯 Evaluate Candidate</span>
                </button>
                {evalError && <div className="error-message">⚠️ {evalError}</div>}
              </div>
            )}

            {isEvaluating && (
              <div className="eval-loading">
                <div className="eval-loading-spinner"></div>
                <p>Evaluating candidate against JD requirements…</p>
              </div>
            )}

            {evaluation && (
              <div className="eval-results">
                {/* Overall Score Ring */}
                <div className="eval-overview">
                  <div className={`eval-score-ring ${
                    evaluation.overall_score >= 7.5 ? 'ring-strong' :
                    evaluation.overall_score >= 5.0 ? 'ring-moderate' :
                    evaluation.overall_score >= 2.5 ? 'ring-weak' : 'ring-none'
                  }`}>
                    <div className="eval-score-value">{evaluation.overall_score}</div>
                    <div className="eval-score-label">/ 10</div>
                  </div>
                  <div className={`eval-recommendation ${
                    evaluation.recommendation === 'Strong Match' ? 'rec-strong' :
                    evaluation.recommendation === 'Moderate Match' ? 'rec-moderate' :
                    evaluation.recommendation === 'Weak Match' ? 'rec-weak' : 'rec-none'
                  }`}>
                    {evaluation.recommendation}
                  </div>
                </div>

                {/* Skill-by-Skill Breakdown */}
                {evaluation.scorecard?.length > 0 && (
                  <div className="eval-skills">
                    <h3 className="eval-skills-title">Skill Breakdown</h3>
                    {evaluation.scorecard.map((item, i) => (
                      <div key={i} className="eval-skill-row">
                        <div className="eval-skill-header">
                          <span className="eval-skill-name">{item.skill}</span>
                          <span className={`eval-skill-score ${
                            item.score >= 8 ? 'score-high' :
                            item.score >= 5 ? 'score-mid' : 'score-low'
                          }`}>{item.score}/10</span>
                        </div>
                        <div className="eval-skill-bar-track">
                          <div
                            className={`eval-skill-bar-fill ${
                              item.score >= 8 ? 'bar-high' :
                              item.score >= 5 ? 'bar-mid' : 'bar-low'
                            }`}
                            style={{ width: `${item.score * 10}%` }}
                          ></div>
                        </div>
                        <p className="eval-skill-justification">{item.justification}</p>
                      </div>
                    ))}
                  </div>
                )}
              </div>
            )}
          </div>
        )}

        {/* ═══ Sanitization Panel ════════════════════════════ */}
        {result?.sanitizedText && (
          <div className="result-panel step3-info-panel">
            <div className="panel-header">
              <h2>🛡️ Sanitization Agent</h2>
              <span className="bias-badge">GLiNER Active</span>
            </div>
            <div className="step3-body">
              <p className="step3-description">
                Personally identifiable information — names, emails, phone numbers,
                addresses, dates, and organisations — has been detected and redacted
                using the <strong>GLiNER</strong> NLP model before being passed to
                the structuring agent.
              </p>
              <details className="sanitized-details">
                <summary>View anonymised resume text ↓</summary>
                <pre className="sanitized-pre">{result.sanitizedText}</pre>
              </details>
            </div>
          </div>
        )}
      </main>
    </div>
  );
}

export default App;
